#include "vocabulary.h"

using namespace std;
using namespace cv;

// Write vocabulary
// INPUT:
//   filename: vocabulary file name
//   vocabulary: vocabulary
// RETURN:
//   true if succeeded
bool WriteVocabulary(const string& filename, const Mat& vocabulary) {
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
Mat TrainVocabulary(const string& filename,
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