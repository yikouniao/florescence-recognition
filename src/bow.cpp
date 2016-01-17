#include "bow.h"
#include "directory.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;
using namespace std;

// Read BOW image descriptor
// INPUT:
//   file: file name
// OUTPUT:
//   bow_img_descriptor: BOW descriptor
// RETURN:
//   true if succeeded
bool ReadBowImgDescriptor(const string& file, Mat& bow_img_descriptor) {
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
bool WriteBowImgDescriptor(const string& file, const Mat& bow_img_descriptor) {
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
void CalculateImageDescriptors(
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
void ComputeConfidences(const Ptr<SVM>& svm, const size_t class_idx,
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
void RemoveEmptyBowImageDescriptors(vector<Image>& images,
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
void RemoveEmptyBowImageDescriptors(vector<Image>& images,
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