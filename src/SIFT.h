#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

const std::string data_dir = "data/";
const std::vector<std::string> pic_dir {
  "Fully-bloomed/",
  "Non-bloomed/",
  "Partially-bloomed/"
};
const std::string images_path = "data/images.xml";
const std::string train_vocabulary_path = "data/train.xml";

const int vocab_size = 50;
const size_t train_pic_num = 20;

enum Florescence {
  FULLY_BLOOMED, NON_BLOOMED, PARTIALLY_BLOOMED,
  CLASS_CNT, CLASS_UNKNOWN = CLASS_CNT
};
enum DatasetType { TRAIN, TEST };

struct Image {
  Image() : f_name(""), florescence(CLASS_UNKNOWN), datatype(TEST) {}
  Image(std::string p_f, Florescence p_florescence, DatasetType p_datatype)
      : f_name(p_f), florescence(p_florescence), datatype(p_datatype) {}
  std::string f_name; // file name
  Florescence florescence; // a flag for different classes
  DatasetType datatype; // train or test
};

void InitImages(std::vector<Image>& images);
static void SaveImages(const std::string& filename,
                       const std::vector<Image>& images);
static bool readVocabulary(const std::string& filename, cv::Mat& vocabulary);
static bool writeVocabulary(const std::string& filename,
                            const cv::Mat& vocabulary);
static cv::Mat trainVocabulary(
    const std::string& filename,
    const cv::Ptr<cv::FeatureDetector>& fdetector,
    const cv::Ptr<cv::DescriptorExtractor>& dextractor);
void test0();
