#include "SIFT.h"
#include <fstream>

#if defined WIN32 || defined _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max
#include "sys/types.h"
#endif
#include <sys/stat.h>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;
using namespace std;

static void Help() {
  cout << "";
}

static void MakeDir(const string& dir)
{
#if defined WIN32 || defined _WIN32
  CreateDirectoryA(dir.c_str(), 0);
#else
  mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
}

static void MakeUsedDirs()
{
  MakeDir(svms_dir);
  MakeDir(bowImageDescriptorsDir);
  for (size_t i = 0; i < CLASS_CNT; ++i) {
    MakeDir(bowImageDescriptorsDir + "/" + obj_classes[i]);
  }
}

static void DivideImagesIntoTrainTest(vector<Image>& images_train,
                                      vector<Image>& images_test) {
  RNG& rng = theRNG();
  size_t i = train_pic_num;
  while (i-- > 0)
  {
    // Randomly pick an image from the dataset for training
    int randImgIdx = rng((unsigned)images_test.size());
    images_train.push_back(images_test[randImgIdx]);
    images_test.erase(images_test.begin() + randImgIdx);
  }
}

static void InitImages(vector<Image>& images_train, vector<Image>& images_test) {
  for (int i = FULLY_BLOOMED; i < CLASS_CNT; ++i) {
    const string dir = data_dir + obj_classes[i];
    vector<String> filenames;
    glob(dir, filenames); // read a sequence of files within a folder
    for (size_t j = 0; j < filenames.size(); ++j) {
      images_test.push_back({filenames[j], static_cast<Florescence>(i)});
    }
  }
  DivideImagesIntoTrainTest(images_train, images_test);
}

static void SaveImages(const string& filename, const vector<Image>& images_train,
                       const vector<Image>& images_test)
{
  cout << "Saving images...";
  FileStorage fs(filename, FileStorage::WRITE);
  fs << "strings" << "["; // text - string sequence
  fs << "images for training";
  for (const auto& e : images_train) {
      fs << e.f_name;
  }
  fs << "]"; // close sequence
  fs << "strings" << "["; // text - string sequence
  fs << "images for testing"; // text - string sequence
  for (const auto& e : images_test) {
    fs << e.f_name;
  }
  fs << "]"; // close sequence
}

static bool WriteVocabulary(const string& filename, const Mat& vocabulary)
{
  cout << "Saving vocabulary..." << endl;
  FileStorage fs(filename, FileStorage::WRITE);
  if (fs.isOpened())
  {
    fs << "vocabulary" << vocabulary;
    return true;
  }
  return false;
}

static Mat TrainVocabulary(const string& filename,
                           const Ptr<FeatureDetector>& fdetector,
                           const Ptr<DescriptorExtractor>& dextractor,
                           vector<Image>& images) {
  Mat vocabulary;
  CV_Assert(dextractor->descriptorType() == CV_32FC1);

  cout << "Computing descriptors..." << endl;
  TermCriteria terminate_criterion;
  terminate_criterion.epsilon = FLT_EPSILON;
  BOWKMeansTrainer bowTrainer(vocab_size, terminate_criterion, 3, KMEANS_PP_CENTERS);

  size_t i = images.size();
  while (i-- > 0)
  {
    // compute the descriptors from train image.
    Mat colorImage = imread(images[i].f_name);
    if (!colorImage.data) {
      cerr << images[i].f_name << "can not be read.\n";
      exit(1);
    }
    vector<KeyPoint> imageKeypoints;
    fdetector->detect(colorImage, imageKeypoints);
    Mat imageDescriptors;
    dextractor->compute(colorImage, imageKeypoints, imageDescriptors);

    //check that there were descriptors calculated for the current image
    if (!imageDescriptors.empty())
    {
      for (int i = 0; i < imageDescriptors.rows; i++)
      {
          bowTrainer.add(imageDescriptors.row(i));
      }
    }
  }
  cout << "Actual descriptor count: " << bowTrainer.descriptorsCount() << endl;

  cout << "Training vocabulary..." << endl;
  vocabulary = bowTrainer.cluster();

  if (!WriteVocabulary(filename, vocabulary))
  {
    cout << "Error: file " << filename << " can not be opened to write" << endl;
    exit(-1);
  }
  return vocabulary;
}

static bool readBowImageDescriptor(const string& file, Mat& bowImageDescriptor)
{
  FileStorage fs(file, FileStorage::READ);
  if (fs.isOpened())
  {
    fs["imageDescriptor"] >> bowImageDescriptor;
    return true;
  }
  return false;
}

static bool writeBowImageDescriptor(const string& file, const Mat& bowImageDescriptor)
{
  FileStorage fs(file, FileStorage::WRITE);
  if (fs.isOpened())
  {
    fs << "imageDescriptor" << bowImageDescriptor;
    return true;
  }
  return false;
}

// Load in the bag of words vectors for a set of images, from file if possible
static void calculateImageDescriptors(const vector<Image>& images, vector<Mat>& imageDescriptors,
  const Ptr<BOWImgDescriptorExtractor>& bow_extractor, const Ptr<FeatureDetector>& fdetector)
{
  CV_Assert(!bow_extractor->getVocabulary().empty());
  imageDescriptors.resize(images.size());

  for (size_t i = 0; i < images.size(); i++)
  {
    string filename = bowImageDescriptorsDir + images[i].f_name.substr(4) + ".xml.gz";

    Mat colorImage = imread(images[i].f_name);
    vector<KeyPoint> keypoints;
    fdetector->detect(colorImage, keypoints);
    bow_extractor->compute(colorImage, keypoints, imageDescriptors[i]);
    if (!imageDescriptors[i].empty())
    {
      if (!writeBowImageDescriptor(filename, imageDescriptors[i]))
      {
        cout << "Error: file " << filename << "can not be opened to write bow image descriptor" << endl;
        exit(-1);
      }
    }
  }
}

static void setSVMParams(Ptr<SVM> & svm, const Mat& responses, bool balanceClasses)
{
  int pos_ex = countNonZero(responses == 1);
  int neg_ex = countNonZero(responses == -1);
  cout << pos_ex << " positive training samples; " << neg_ex << " negative training samples" << endl;

  svm->setType(SVM::C_SVC);
  svm->setKernel(SVM::RBF);
  if (balanceClasses)
  {
    Mat class_wts(2, 1, CV_32FC1);
    // The first training sample determines the '+1' class internally, even if it is negative,
    // so store whether this is the case so that the class weights can be reversed accordingly.
    bool reversed_classes = (responses.at<float>(0) < 0.f);
    if (reversed_classes == false)
    {
      class_wts.at<float>(0) = static_cast<float>(pos_ex) / static_cast<float>(pos_ex + neg_ex); // weighting for costs of positive class + 1 (i.e. cost of false positive - larger gives greater cost)
      class_wts.at<float>(1) = static_cast<float>(neg_ex) / static_cast<float>(pos_ex + neg_ex); // weighting for costs of negative class - 1 (i.e. cost of false negative)
    }
    else
    {
      class_wts.at<float>(0) = static_cast<float>(neg_ex) / static_cast<float>(pos_ex + neg_ex);
      class_wts.at<float>(1) = static_cast<float>(pos_ex) / static_cast<float>(pos_ex + neg_ex);
    }
    svm->setClassWeights(class_wts);
  }
}

static void setSVMTrainAutoParams(ParamGrid& c_grid, ParamGrid& gamma_grid,
  ParamGrid& p_grid, ParamGrid& nu_grid,
  ParamGrid& coef_grid, ParamGrid& degree_grid)
{
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

static Ptr<SVM> trainSVMClassifier(const SVMTrainParamsExt& svmParamsExt, const string& objClassName,
  Ptr<BOWImgDescriptorExtractor>& bow_extractor, const Ptr<FeatureDetector>& fdetector,
  vector<Image>& images, vector<char>& objectPresent)
{
  /* first check if a previously trained svm for the current class has been saved to file */
  string svmFilename = svms_dir + "/" + objClassName + ".xml.gz";
  Ptr<SVM> svm;

  FileStorage fs(svmFilename, FileStorage::READ);

  cout << "TRAINING CLASSIFIER FOR CLASS " << objClassName << "\n";
  cout << "CALCULATING BOW VECTORS FOR TRAINING SET OF " << objClassName << "\n";

  // Get classification ground truth for images in the training set
  vector<Mat> bow_img_descrs;
  
  // Compute the bag of words vector for each image in the training set.
  calculateImageDescriptors(images, bow_img_descrs, bow_extractor, fdetector);

  // Remove any images for which descriptors could not be calculated
  removeEmptyBowImageDescriptors(images, bow_img_descrs, objectPresent);

  // Prepare the input matrices for SVM training.
  Mat trainData((int)images.size(), bow_extractor->getVocabulary().rows, CV_32FC1);
  Mat responses((int)images.size(), 1, CV_32SC1);

  // Transfer bag of words vectors and responses across to the training data matrices
  for (size_t img_idx = 0; img_idx < images.size(); img_idx++)
  {
    // Transfer image descriptor (bag of words vector) to training data matrix
    Mat submat = trainData.row((int)img_idx);
    if (bow_img_descrs[img_idx].cols != bow_extractor->descriptorSize())
    {
      cout << "Error: computed bow image descriptor size " << bow_img_descrs[img_idx].cols
        << " differs from vocabulary size" << bow_extractor->getVocabulary().cols << endl;
      exit(-1);
    }
    bow_img_descrs[img_idx].copyTo(submat);

    // Set response value
    responses.at<int>((int)img_idx) = objectPresent[img_idx] ? 1 : -1;
  }

  cout << "TRAINING SVM FOR CLASS ..." << objClassName << "..." << endl;
  svm = SVM::create();
  setSVMParams(svm, responses, svmParamsExt.balanceClasses);
  ParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
  setSVMTrainAutoParams(c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);

  svm->trainAuto(TrainData::create(trainData, ROW_SAMPLE, responses), 10,
    c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);
  cout << "SVM TRAINING FOR CLASS " << objClassName << " COMPLETED" << endl;

  svm->save(svmFilename);
  cout << "SAVED CLASSIFIER TO FILE" << endl;

  return svm;
}

void SVMTrainParamsExt::SVMTrainParamsExtFile() {
  string svmFilename = svms_dir + "/svm.xml";

  FileStorage paramsFS(svmFilename, FileStorage::READ);
  if (paramsFS.isOpened())
  {
    FileNode fn = paramsFS.root();
    FileNode currFn = fn;
    currFn = fn["svm_train_params_ext"];
    read(currFn);
  }
  else
  {
    paramsFS.open(svmFilename, FileStorage::WRITE);
    if (paramsFS.isOpened())
    {
      paramsFS << "svm_train_params_ext" << "{";
      write(paramsFS);
      paramsFS << "}";
      paramsFS.release();
    }
    else
    {
      cout << "File " << svmFilename << "can not be opened to write" << endl;
      exit(1);
    }
  }
}

static void removeEmptyBowImageDescriptors(vector<Image>& images, vector<Mat>& bow_img_descrs,
  vector<char>& objectPresent)
{
  CV_Assert(!images.empty());
  for (int i = (int)images.size() - 1; i >= 0; i--)
  {
    if (bow_img_descrs[i].empty())
    {
      cout << "Removing image " << images[i].f_name << " due to no descriptors..." << endl;
      images.erase(images.begin() + i);
      bow_img_descrs.erase(bow_img_descrs.begin() + i);
      objectPresent.erase(objectPresent.begin() + i);
    }
  }
}

static void removeEmptyBowImageDescriptors(vector<Image>& images, vector<Mat>& bow_img_descrs)
{
  CV_Assert(!images.empty());
  for (int i = (int)images.size() - 1; i >= 0; i--)
  {
    if (bow_img_descrs[i].empty())
    {
      cout << "Removing image " << images[i].f_name << " due to no descriptors..." << endl;
      images.erase(images.begin() + i);
      bow_img_descrs.erase(bow_img_descrs.begin() + i);
    }
  }
}

// compute confidences
// INPUT:
//   svm: svm data
//   class_idx: index of classes
//   bow_extractor: BOW descriptor extractor
//   fdetector: feature detector
// INPUT&OUTPUT:
//   images: a vector of Image to be classified
// OUTPUT:
//   confidences: confidences of each object for all classes
static void computeConfidences(const Ptr<SVM>& svm, const size_t class_idx,
                               const Ptr<BOWImgDescriptorExtractor>& bow_extractor,
                               const Ptr<FeatureDetector>& fdetector,
                               vector<Image>& images,
                               vector<array<float, CLASS_CNT>>& confidences) {
  vector<Mat> bow_img_descrs;// BOW image descriptors

  // Compute the bag of words vector for each image in the test set
  cout << "Calculating BOW vectors for TEST set of " << obj_classes[class_idx] << " .\n";
  calculateImageDescriptors(images, bow_img_descrs, bow_extractor, fdetector);
  // Remove any images for which descriptors could not be calculated
  removeEmptyBowImageDescriptors(images, bow_img_descrs);

  // Use the bag of words vectors to calculate classifier output for each image in test set
  cout << "Calculating confidences for class " << obj_classes[class_idx] << " .\n";
  float sign_mul = 1.f;
  for (size_t img_idx = 0; img_idx < images.size(); img_idx++) {
    if (img_idx == 0) {
      // In the first iteration, determine the sign of the positive class
      float class_val = confidences[img_idx][class_idx]
                      = svm->predict(bow_img_descrs[img_idx], noArray(), 0);
      float score_val = confidences[img_idx][class_idx]
                      = svm->predict(bow_img_descrs[img_idx], noArray(), StatModel::RAW_OUTPUT);
      sign_mul = (class_val < 0) == (score_val < 0) ? 1.f : -1.f;
    }
    // svm output of decision function
    confidences[img_idx][class_idx] = sign_mul * svm->predict(bow_img_descrs[img_idx], noArray(), StatModel::RAW_OUTPUT);
  }
  cout << "DONE - " << obj_classes[class_idx] << "\n\n\n";
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

// write classifier results file
// INPUT:
//   images: a vector of Image that has been classified
//   confidences: confidences of each object for all classes
//   florescences: which class each object belongs to
void writeClassifierResultsFile(const vector<Image>& images,
                                const vector<array<float, CLASS_CNT>>& confidences,
                                const vector<Florescence>& florescences) {
  CV_Assert(images.size() == confidences.size());

  string output_file = results_dir;
  cout << "Writing test results file " << output_file << " \n";
  //output data to file
  ofstream result_file(output_file.c_str());
  if (result_file.is_open()) {
    array<size_t, CLASS_CNT> n_correct{0};
    array<size_t, CLASS_CNT> n_all{0};
    array<double, CLASS_CNT> accuracy;
    size_t n_correct_total{0};
    double accuracy_total;
    
    result_file << "file name     confidences of ";
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
      // prepare for calculation of precision and recall
      if (florescences[i] == images[i].florescence) {
        ++n_correct[images[i].florescence];
      }
      ++n_all[images[i].florescence];
    }
    // calculate precision and recall for each class, write into file
    for (size_t i = 0; i < CLASS_CNT; ++i) {
      n_correct_total += n_correct[i];
      accuracy[i] = static_cast<double>(n_correct[i]) / n_all[i];
      cout << "Accuracy of class " << obj_classes[i] << "is: " << accuracy[i]
           << " (" << n_correct[i] << "/" << n_all[i] << ")\n";
      result_file << "recall of class " << obj_classes[i] << "is: "
                  << accuracy[i] << " (" << n_correct[i] << "/" << n_all[i]
                  << ")\n";
    }
    // calculate total precision and recall, write into file
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

// train and test dataset
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
    bow_extractor
        = makePtr<BOWImgDescriptorExtractor>(desc_extractor, desc_matcher);
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
    vector<char> object_present;
    // init objectPresent
    for (size_t img_idx = 0; img_idx < images_train.size(); ++ img_idx) {
      object_present.push_back(class_idx == images_train[img_idx].florescence);
    }

    // Train a classifier on train dataset
    Ptr<SVM> svm = trainSVMClassifier(svm_train_params_ext, obj_classes[class_idx],
                                      bow_extractor, feature_detector,
                                      images_train, object_present);

    // Use the classifier over all images on the test dataset
    computeConfidences(svm, class_idx, bow_extractor,
                       feature_detector, images_test, confidences);
  }

  // 3. Calculate and save the result and precision-recall
  CalculateResult(confidences, florescences);
  writeClassifierResultsFile(images_test, confidences, florescences);
}