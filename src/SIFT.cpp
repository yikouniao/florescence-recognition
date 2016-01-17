#include "SIFT.h"
#include <fstream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;
using namespace std;

static void Help() {
  cout << "";
}

// Write vocabulary
// INPUT:
//   filename: vocabulary file name
//   vocabulary: vocabulary
// RETURN:
//   true if succeeded
static bool WriteVocabulary(const string& filename, const Mat& vocabulary) {
  cout << "Saving vocabulary...\n";
  FileStorage fs(filename, FileStorage::WRITE);
  if (fs.isOpened()) {
    fs << "vocabulary" << vocabulary;
    return true;
  }
  return false;
}

// Train vocabulary
// INPUT:
//   filename: vocabulary file name
//   fdetector: feature detector
//   dextractor: descriptor extractor
//   images: a vector of Image for training
// RETURN:
//   vocabulary
static Mat TrainVocabulary(const string& filename,
                           const Ptr<FeatureDetector>& fdetector,
                           const Ptr<DescriptorExtractor>& dextractor,
                           const vector<Image>& images) {
  Mat vocabulary;
  CV_Assert(dextractor->descriptorType() == CV_32FC1);

  cout << "Computing descriptors...\n";
  TermCriteria terminate_criterion;
  terminate_criterion.epsilon = FLT_EPSILON;
  BOWKMeansTrainer bow_trainer(vocab_size, terminate_criterion,
                               3, KMEANS_PP_CENTERS);
  size_t i = images.size();
  while (i-- > 0) {
    // Compute the descriptors from train image.
    Mat color_img = imread(images[i].f_name);
    if (!color_img.data) {
      cerr << images[i].f_name << "can not be read.\n";
      exit(1);
    }
    vector<KeyPoint> img_keypoints;
    fdetector->detect(color_img, img_keypoints);
    Mat img_descriptors;
    dextractor->compute(color_img, img_keypoints, img_descriptors);

    // Check that there were descriptors calculated for the current image
    if (!img_descriptors.empty()) {
      for (int i = 0; i < img_descriptors.rows; i++) {
          bow_trainer.add(img_descriptors.row(i));
      }
    }
  }
  cout << "Actual descriptor count: " << bow_trainer.descriptorsCount() << "\n";

  cout << "Training vocabulary...\n";
  vocabulary = bow_trainer.cluster();

  if (!WriteVocabulary(filename, vocabulary)) {
    cout << filename << " can not be opened to write.\n";
    exit(-1);
  }
  return vocabulary;
}

// Read BOW image descriptor
// INPUT:
//   file: file name
// OUTPUT:
//   bow_img_descriptor: BOW descriptor
// RETURN:
//   true if succeeded
static bool ReadBowImgDescriptor(const string& file, Mat& bow_img_descriptor)
{
  FileStorage fs(file, FileStorage::READ);
  if (fs.isOpened()) {
    fs["imageDescriptor"] >> bow_img_descriptor;
    return true;
  }
  return false;
}

// Write BOW image descriptor
// INPUT:
//   file: file name
//   bow_img_descriptor: BOW descriptor
static bool WriteBowImgDescriptor(const string& file, const Mat& bow_img_descriptor)
{
  FileStorage fs(file, FileStorage::WRITE);
  if (fs.isOpened()) {
    fs << "imageDescriptor" << bow_img_descriptor;
    return true;
  }
  return false;
}

// Load in the bag of words vectors for a set of images, from file if possible
// INPUT:
//   images: a vector of Image
//   bow_extractor: BOW image descriptor extractor
//   fdetector: feature detector
// OUTPUT:
//   img_descriptors: image descriptors
static void CalculateImageDescriptors(
    const vector<Image>& images,
    vector<Mat>& img_descriptors, const Ptr<BOWImgDescriptorExtractor>& bow_extractor,
    const Ptr<FeatureDetector>& fdetector) {
  CV_Assert(!bow_extractor->getVocabulary().empty());
  img_descriptors.resize(images.size());

  for (size_t i = 0; i < images.size(); ++i) {
    string filename = bow_img_descriptors_dir +
                          images[i].f_name.substr(4) + ".xml.gz";

    Mat color_img = imread(images[i].f_name);
    vector<KeyPoint> keypoints;
    fdetector->detect(color_img, keypoints);
    bow_extractor->compute(color_img, keypoints, img_descriptors[i]);
    if (!img_descriptors[i].empty()) {
      if (!WriteBowImgDescriptor(filename, img_descriptors[i])) {
        cout << filename << " can not be opened.\n";
        exit(-1);
      }
    }
  }
}

// set SVM parameters
// INPUT:
//   responses: response
//   balance_classes: SVMTrainParamsExt::balance_classes
// OUTPUT:
//   svm: SVM parameters
static void setSVMParams(Ptr<SVM> & svm, const Mat& responses, bool balance_classes) {
  int pos_ex = countNonZero(responses == 1);
  int neg_ex = countNonZero(responses == -1);
  cout << pos_ex << " positive training samples; "
       << neg_ex << " negative training samples.\n";

  svm->setType(SVM::C_SVC);
  svm->setKernel(SVM::RBF);
  if (balance_classes) {
    Mat class_wts(2, 1, CV_32FC1);
    // The first training sample determines the '+1' class internally,
    // even if it is negative, so store whether this is the case
    // so that the class weights can be reversed accordingly.
    bool reversed_classes = (responses.at<float>(0) < 0.f);
    if (reversed_classes == false) {
      // weighting for costs of positive class + 1
      // (i.e. cost of false positive - larger gives greater cost)
      class_wts.at<float>(0) = static_cast<float>(pos_ex) /
                                   static_cast<float>(pos_ex + neg_ex);
      // weighting for costs of negative class - 1
      // (i.e. cost of false negative)
      class_wts.at<float>(1) = static_cast<float>(neg_ex) /
                                   static_cast<float>(pos_ex + neg_ex);
    } else {
      class_wts.at<float>(0) = static_cast<float>(neg_ex) /
                                   static_cast<float>(pos_ex + neg_ex);
      class_wts.at<float>(1) = static_cast<float>(pos_ex) /
                                   static_cast<float>(pos_ex + neg_ex);
    }
    svm->setClassWeights(class_wts);
  }
}

// Set SVM train default parameters
static void SetSVMTrainAutoParams(ParamGrid& c_grid, ParamGrid& gamma_grid,
                                  ParamGrid& p_grid, ParamGrid& nu_grid,
                                  ParamGrid& coef_grid, ParamGrid& degree_grid) {
  c_grid = SVM::getDefaultGrid(SVM::C);

  gamma_grid = SVM::getDefaultGrid(SVM::GAMMA);

  p_grid = SVM::getDefaultGrid(SVM::P);
  p_grid.logStep = 0;

  nu_grid = SVM::getDefaultGrid(SVM::NU);
  nu_grid.logStep = 0;

  coef_grid = SVM::getDefaultGrid(SVM::COEF);
  coef_grid.logStep = 0;

  degree_grid = SVM::getDefaultGrid(SVM::DEGREE);
  degree_grid.logStep = 0;
}

// Train SVM classifier
// INPUT:
//   svm_params_ext: svm parameters
//   class_name: class name
//   fdetector: feature detector
// INPUT&OUTPUT:
//   images: a vector of Image for training
//   obj_present: an array of bools specifying whether the object
//                defined by obj_classes is present in each image or not
//bow_extractor?????????????????????
// RETURN:
//   svm
static Ptr<SVM> TrainSVMClassifier(
    const SVMTrainParamsExt& svm_params_ext, const string& class_name,
    Ptr<BOWImgDescriptorExtractor>& bow_extractor,
    const Ptr<FeatureDetector>& fdetector, vector<Image>& images,
    vector<char>& obj_present) {
  // first check if a previously trained svm for the current class has been saved to file
  string svm_file_name = svms_dir + "/" + class_name + ".xml.gz";
  Ptr<SVM> svm;
  FileStorage fs(svm_file_name, FileStorage::READ);

  cout << "Training classifier for class " << class_name << "\n";
  cout << "Calculating BOW vectors for training set of " << class_name << "\n";
  vector<Mat> bow_img_descrs;
  
  // Compute the bag of words vector for each image in the training set.
  CalculateImageDescriptors(images, bow_img_descrs, bow_extractor, fdetector);

  // Remove any images for which descriptors could not be calculated
  RemoveEmptyBowImageDescriptors(images, bow_img_descrs, obj_present);

  // Prepare the input matrices for SVM training.
  Mat train_data((int)images.size(),
                 bow_extractor->getVocabulary().rows, CV_32FC1);
  Mat responses((int)images.size(), 1, CV_32SC1);

  // Transfer BOWs vectors and responses across to the training data matrices
  for (size_t img_idx = 0; img_idx < images.size(); ++img_idx) {
    // Transfer image descriptor (bag of words vector) to training data matrix
    Mat submat = train_data.row((int)img_idx);
    if (bow_img_descrs[img_idx].cols != bow_extractor->descriptorSize()) {
      cout << "Error: computed bow image descriptor size "
           << bow_img_descrs[img_idx].cols << " differs from vocabulary size"
           << bow_extractor->getVocabulary().cols << ".\n";
      exit(-1);
    }
    bow_img_descrs[img_idx].copyTo(submat);

    // Set response value
    responses.at<int>((int)img_idx) = obj_present[img_idx] ? 1 : -1;
  }

  cout << "Training SVM for class " << class_name << "...\n";
  svm = SVM::create();
  setSVMParams(svm, responses, svm_params_ext.balance_classes);
  ParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
  SetSVMTrainAutoParams(c_grid, gamma_grid, p_grid,
                        nu_grid, coef_grid, degree_grid);

  svm->trainAuto(TrainData::create(train_data, ROW_SAMPLE, responses), 10,
                 c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);
  cout << "SVM Training for class " << class_name << " completed.\n";

  svm->save(svm_file_name);
  cout << "saved classifier to file.\n";

  return svm;
}

// Write SVM parameters into file
void SVMTrainParamsExt::SVMTrainParamsExtFile() {
  string svm_file_name = svms_dir + "/svm.xml";
  FileStorage paramsFS(svm_file_name, FileStorage::WRITE);
  if (paramsFS.isOpened()) {
    paramsFS << "svm_train_params_ext" << "{";
    Write(paramsFS);
    paramsFS << "}";
    paramsFS.release();
  } else {
    cout << svm_file_name << "can not be opened to write.\n";
    exit(1);
  }
}

// Compute confidences
// INPUT:
//   svm: svm data
//   class_idx: index of classes
//   bow_extractor: BOW descriptor extractor
//   fdetector: feature detector
// INPUT&OUTPUT:
//   images: a vector of Image to be classified
// OUTPUT:
//   confidences: confidences of each object for all classes
static void ComputeConfidences(const Ptr<SVM>& svm, const size_t class_idx,
                               const Ptr<BOWImgDescriptorExtractor>& bow_extractor,
                               const Ptr<FeatureDetector>& fdetector,
                               vector<Image>& images,
                               vector<array<float, CLASS_CNT>>& confidences) {
  vector<Mat> bow_img_descrs;// BOW image descriptors

  // Compute the bag of words vector for each image in the test set
  cout << "Calculating BOW vectors for TEST set of " << obj_classes[class_idx] << " .\n";
  CalculateImageDescriptors(images, bow_img_descrs, bow_extractor, fdetector);
  // Remove any images for which descriptors could not be calculated
  RemoveEmptyBowImageDescriptors(images, bow_img_descrs);

  // Use the bag of words vectors to calculate classifier output for each image in test set
  cout << "Calculating confidences for class " << obj_classes[class_idx] << " .\n";
  float sign_mul = 1.f;
  for (size_t img_idx = 0; img_idx < images.size(); img_idx++) {
    if (img_idx == 0) {
      // In the first iteration, determine the sign of the positive class
      float class_val = confidences[img_idx][class_idx]
                      = svm->predict(bow_img_descrs[img_idx], noArray(), 0);
      float score_val = confidences[img_idx][class_idx]
                      = svm->predict(bow_img_descrs[img_idx], noArray(),
                                     StatModel::RAW_OUTPUT);
      sign_mul = (class_val < 0) == (score_val < 0) ? 1.f : -1.f;
    }
    // svm output of decision function
    confidences[img_idx][class_idx]
        = sign_mul * svm->predict(bow_img_descrs[img_idx], noArray(),
                                  StatModel::RAW_OUTPUT);
  }
  cout << obj_classes[class_idx] << " DONE.\n\n\n";
}

// Remove empty BOW image descriptors
// INPUT&OUTPUT:
//   images: a vector of Image to be classified
//   bow_img_descrs: BOW image descriptors
//   obj_present: An array of bools specifying whether the object
//                   defined by obj_classes is present in each image or not
static void RemoveEmptyBowImageDescriptors(vector<Image>& images,
                                           vector<Mat>& bow_img_descrs,
                                           vector<char>& obj_present) {
  CV_Assert(!images.empty());
  for (int i = static_cast<int>(images.size()) - 1; i >= 0; --i) {
    if (bow_img_descrs[i].empty()) {
      cout << "Removing " << images[i].f_name << " due to no descriptors.\n";
      images.erase(images.begin() + i);
      bow_img_descrs.erase(bow_img_descrs.begin() + i);
      obj_present.erase(obj_present.begin() + i);
    }
  }
}

// Remove empty BOW image descriptors
// INPUT&OUTPUT:
//   images: a vector of Image to be classified
//   bow_img_descrs: BOW image descriptors
static void RemoveEmptyBowImageDescriptors(vector<Image>& images,
                                           vector<Mat>& bow_img_descrs) {
  CV_Assert(!images.empty());
  for (int i = static_cast<int>(images.size()) - 1; i >= 0; --i) {
    if (bow_img_descrs[i].empty()) {
      cout << "Removing " << images[i].f_name << " due to no descriptors.\n";
      images.erase(images.begin() + i);
      bow_img_descrs.erase(bow_img_descrs.begin() + i);
    }
  }
}

// Calculate the classification result
// INPUT:
//   confidences: confidences of each object for all classes
// OUTPUT:
//   florescences: which class each object belongs to
void CalculateResult(const vector<array<float, CLASS_CNT>>& confidences,
                     vector<Florescence>& florescences) {
  for (size_t i = 0; i < florescences.size(); ++i) {
    size_t max_conf = 0;
    for (size_t j = 1; j < CLASS_CNT; ++j) {
      if (confidences[i][max_conf] < confidences[i][j])
        max_conf = j;
    }
    florescences[i] = static_cast<Florescence>(max_conf);
  }
}

// Write classifier results file
// INPUT:
//   images: a vector of Image that has been classified
//   confidences: confidences of each object for all classes
//   florescences: which class each object belongs to
void WriteClassifierResultsFile(const vector<Image>& images,
                                const vector<array<float, CLASS_CNT>>& confidences,
                                const vector<Florescence>& florescences) {
  CV_Assert(images.size() == confidences.size());

  string output_file = results_dir;
  cout << "Writing test results file " << output_file << " \n";
  // Output data to file
  ofstream result_file(output_file.c_str());
  if (result_file.is_open()) {
    array<size_t, CLASS_CNT> n_correct{0};
    array<size_t, CLASS_CNT> n_all{0};
    array<double, CLASS_CNT> accuracy;
    size_t n_correct_total{0};
    double accuracy_total;
    
    result_file << "file name                      confidences of ";
    for (size_t i = 0; i < CLASS_CNT; ++i) {
      result_file << obj_classes[i] << " ";
    }
    result_file << "result\n";

    for (size_t i = 0; i < images.size(); ++i) {
      result_file << images[i].f_name << " ";
      for (size_t j = 0; j < CLASS_CNT; ++j) {
        result_file << confidences[i][j] << " ";
      }
      result_file << obj_classes[florescences[i]] << "\n";
      // Prepare for calculation of precision and recall
      if (florescences[i] == images[i].florescence) {
        ++n_correct[images[i].florescence];
      }
      ++n_all[images[i].florescence];
    }
    // Calculate precision and recall for each class, write into file
    for (size_t i = 0; i < CLASS_CNT; ++i) {
      n_correct_total += n_correct[i];
      accuracy[i] = static_cast<double>(n_correct[i]) / n_all[i];
      cout << "Accuracy of class " << obj_classes[i] << "is: " << accuracy[i]
           << " (" << n_correct[i] << "/" << n_all[i] << ")\n";
      result_file << "recall of class " << obj_classes[i] << "is: "
                  << accuracy[i] << " (" << n_correct[i] << "/" << n_all[i]
                  << ")\n";
    }
    // Calculate total precision and recall, write into file
    accuracy_total = static_cast<double>(n_correct_total) / images.size();
    cout << "Accuracy of all data is: " << accuracy_total
         << " (" << n_correct_total << "/" << images.size() << ")\n";
    result_file << "Accuracy of all data is: " << accuracy_total
         << " (" << n_correct_total << "/" << images.size() << ")\n";
    result_file.close();
  } else {
    string err_msg = "could not open " + output_file + "\n";
    CV_Error(Error::StsError, err_msg.c_str());
  }
}

// Train and test dataset
void TrainTest() {
  Help();
  MakeUsedDirs();
  Ptr<Feature2D> feature_detector = SIFT::create();
  Ptr<DescriptorExtractor> desc_extractor = feature_detector;
  Ptr<BOWImgDescriptorExtractor> bow_extractor;
  if (!feature_detector || !desc_extractor) {
    cout << "feature_detector or desc_extractor was not created" << endl;
    exit(1);
  }
  {
    Ptr<DescriptorMatcher> desc_matcher =
        DescriptorMatcher::create("BruteForce");
    if (!feature_detector || !desc_extractor || !desc_matcher) {
      cout << "desc_matcher was not created" << endl;
      exit(1);
    }
    bow_extractor =
        makePtr<BOWImgDescriptorExtractor>(desc_extractor, desc_matcher);
  }

  vector<Image> images_train, images_test;
  InitImages(images_train, images_test);
  SaveImages(images_path, images_train, images_test);

  // 1. Train visual word vocabulary
  Mat vocabulary = TrainVocabulary(train_vocabulary_path, feature_detector,
                                   desc_extractor, images_train);
  bow_extractor->setVocabulary(vocabulary);

  // 2. Train a classifier and run a sample query for each object class
  SVMTrainParamsExt svm_train_params_ext;
  svm_train_params_ext.SVMTrainParamsExtFile();
  vector<array<float, CLASS_CNT>> confidences(images_test.size());
  vector<Florescence> florescences(images_test.size());
  for (size_t class_idx = 0; class_idx < obj_classes.size(); ++class_idx)
  {
    // An array of bools specifying whether the object defined by obj_classes
    // is present in each image or not
    vector<char> obj_present;

    // Init obj_present
    for (size_t img_idx = 0; img_idx < images_train.size(); ++ img_idx) {
      obj_present.push_back(class_idx == images_train[img_idx].florescence);
    }

    // Train a classifier on train dataset
    Ptr<SVM> svm = TrainSVMClassifier(
                       svm_train_params_ext, obj_classes[class_idx],
                       bow_extractor, feature_detector, images_train,
                       obj_present);

    // Use the classifier over all images on the test dataset
    ComputeConfidences(svm, class_idx, bow_extractor,
                       feature_detector, images_test, confidences);
  }

  // 3. Calculate and save the result and precision-recall
  CalculateResult(confidences, florescences);
  WriteClassifierResultsFile(images_test, confidences, florescences);
}