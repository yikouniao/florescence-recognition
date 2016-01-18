#include "svm.h"
#include "directory.h"
#include "bow.h"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;

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

// set SVM parameters
// INPUT:
//   responses: response
//   balance_classes: SVMTrainParamsExt::balance_classes
// OUTPUT:
//   svm: SVM parameters
void setSVMParams(Ptr<SVM> & svm, const Mat& responses, bool balance_classes) {
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
void SetSVMTrainAutoParams(ParamGrid& c_grid, ParamGrid& gamma_grid,
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
Ptr<SVM> TrainSVMClassifier(
    const SVMTrainParamsExt& svm_params_ext, const string& class_name,
    Ptr<BOWImgDescriptorExtractor>& bow_extractor,
    const Ptr<FeatureDetector>& fdetector, vector<Image>& images,
    vector<char>& obj_present) {
  string svm_file_name = svms_dir + "/" + class_name + ".xml.gz";
  Ptr<SVM> svm;
  FileStorage fs(svm_file_name, FileStorage::READ);

  cout << "Training classifier for class " << class_name << "...\n";
  cout << "Calculating BoW vectors for training set of "
       << class_name << "...\n";
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