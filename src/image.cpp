#include "image.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "directory.h"

using namespace cv;
using namespace std;

// Initialize the vector of train images and test images
// INPUT&OUTPUT:
//   images_train: images for training
//   images_test: images for testing
void InitImages(vector<Image>& images_train, vector<Image>& images_test) {
  for (int i = FULLY_BLOOMED; i < CLASS_CNT; ++i) {
    // First read in all images and set them as test images
    const string dir = data_dir + obj_classes[i];
    vector<String> filenames;
    glob(dir, filenames); // Read a sequence of files within a folder

    // Randomly pick images for training
    CV_Assert(filenames.size() > train_pic_num_each_class);
    RNG& rng = theRNG();
    rng.state = getTickCount();
    size_t j = train_pic_num_each_class;
    while (j-- > 0) {
      int rand_img_idx = rng((unsigned)filenames.size());
      images_train.push_back({filenames[rand_img_idx],
                              static_cast<Florescence>(i)});
      filenames.erase(filenames.begin() + rand_img_idx);
    }

    // Move the rest images into train set
    for (size_t k = 0; k < filenames.size(); ++k) {
      images_test.push_back({filenames[k], static_cast<Florescence>(i)});
    }
  }
}

// Save all train/test Image into file
// INPUT:
//   filename: vocabulary file name
//   images_train: images for training
//   images_test: images for testing
void SaveImages(const string& filename, const vector<Image>& images_train,
                       const vector<Image>& images_test) {
  cout << "Saving images...\n";
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