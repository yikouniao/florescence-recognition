#include "train-test.h"
#include <fstream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;
using namespace std;

static void Help() {
  cout << "";
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
    array<size_t, CLASS_CNT> n_tp{0};
    array<size_t, CLASS_CNT> n_relevant{0};
    array<double, CLASS_CNT> recall;
    array<size_t, CLASS_CNT> n_selected{0};
    array<double, CLASS_CNT> precision;
    size_t n_tp_total{0};
    double precision_recall_total;
    
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
      // Relevant elements include true positives(TP) and false negatives(FN)
      // Seleted elements include true positives and false positives(FP)
      if (florescences[i] == images[i].florescence) {
        ++n_tp[images[i].florescence];
      }
      ++n_relevant[images[i].florescence];
      ++n_selected[florescences[i]];
    }

    // Calculate precision for each class, write into file
    // recall = TP / (TP + FN) = TP / relevant elements
    for (size_t i = 0; i < CLASS_CNT; ++i) {
      n_tp_total += n_tp[i];
      recall[i] = static_cast<double>(n_tp[i]) / n_relevant[i];
      cout << "Recall of class " << obj_classes[i] << "is: " << recall[i]
           << " (" << n_tp[i] << "/" << n_relevant[i] << ")\n";
      result_file << "Recall of class " << obj_classes[i] << "is: "
                  << recall[i] << " (" << n_tp[i] << "/" << n_relevant[i]
                  << ")\n";
    }

    // Calculate recall for each class, write into file
    // precision = TP / (TP + FP) = TP / seleted elements
    for (size_t i = 0; i < CLASS_CNT; ++i) {
      precision[i] = static_cast<double>(n_tp[i]) / n_selected[i];
      cout << "Precision of class " << obj_classes[i] << "is: " << precision[i]
           << " (" << n_tp[i] << "/" << n_selected[i] << ")\n";
      result_file << "Precision of class " << obj_classes[i] << "is: "
                  << precision[i] << " (" << n_tp[i] << "/" << n_selected[i]
                  << ")\n";
    }

    // Calculate total precision and recall, write into file
    precision_recall_total = static_cast<double>(n_tp_total) / images.size();
    cout << "Precision and recall of all data is: " << precision_recall_total
         << " (" << n_tp_total << "/" << images.size() << ")\n";
    result_file << "Precision and recall of all data is: "
                << precision_recall_total << " (" << n_tp_total << "/"
                << images.size() << ")\n";
    result_file.close();
  } else {
    string err_msg = "can not open " + output_file + "\n";
    CV_Error(Error::StsError, err_msg.c_str());
  }
}

// Train and test dataset
// INPUT&OUTPUT:
//   images_train, images_test: vectors of train data and test data
//                              some images may be erased due to no descriptor
void TrainTest(vector<Image>& images_train, vector<Image>& images_test) {
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

  // 1. Train visual word vocabulary
  Mat vocabulary = TrainVocabulary(train_vocabulary_path, feature_detector,
                                   desc_extractor, images_train);
  bow_extractor->setVocabulary(vocabulary);
  cout << "\n\n";

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
                       obj_present, confidences);

    // Use the classifier over all images on the test dataset
    ComputeConfidences(svm, class_idx, bow_extractor,
                       feature_detector, images_test, confidences);
  }

  // 3. Calculate and save the result and precision-recall
  CalculateResult(confidences, florescences);
  WriteClassifierResultsFile(images_test, confidences, florescences);
}