#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

const std::string data_dir = "data/";
const std::vector<std::string> pic_dir {
  "Fully-bloomed",
  "Non-bloomed",
  "Partially-bloomed"
};
const std::string images_path = data_dir + "images.xml";
const std::string train_vocabulary_path = data_dir + "vocabulary.xml.gz";
const std::string svms_dir = data_dir + "svms";

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

struct SVMTrainParamsExt
{
  SVMTrainParamsExt() : descPercent(0.5f), targetRatio(0.4f), balanceClasses(true) {}
  SVMTrainParamsExt(float _descPercent, float _targetRatio, bool _balanceClasses) :
    descPercent(_descPercent), targetRatio(_targetRatio), balanceClasses(_balanceClasses) {}
  void read(const cv::FileNode& fn)
  {
    fn["descPercent"] >> descPercent;
    fn["targetRatio"] >> targetRatio;
    fn["balanceClasses"] >> balanceClasses;
  }
  void write(cv::FileStorage& fs) const
  {
    fs << "descPercent" << descPercent;
    fs << "targetRatio" << targetRatio;
    fs << "balanceClasses" << balanceClasses;
  }
  void print() const
  {
    std::cout << "descPercent: " << descPercent << "\n";
    std::cout << "targetRatio: " << targetRatio << "\n";
    std::cout << "balanceClasses: " << balanceClasses << "\n";
  }
  // If the file storing parameters of SVM trainer, read it, else creat it.
  void SVMTrainParamsExtFile();
  // Percentage of extracted descriptors to use for training.
  float descPercent;
  // Try to get this ratio of positive to negative samples (minimum).
  float targetRatio;
  // Balance class weights by number of samples in each
  // (if true cSvmTrainTargetRatio is ignored).
  bool balanceClasses;
};

void test0();
