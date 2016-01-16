#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <vector>
#include <array>

const std::string data_dir = "data/";
const std::vector<std::string> obj_classes {
  "Fully-bloomed",
  "Non-bloomed",
  "Partially-bloomed"
};
const std::string images_path = data_dir + "images.xml";
const std::string train_vocabulary_path = data_dir + "vocabulary.xml.gz";
const std::string svms_dir = data_dir + "svms";
const std::string bowImageDescriptorsDir = data_dir + "bow_img_descrs";
const std::string results_dir = data_dir + "results.txt";

const int vocab_size = 50;
const size_t train_pic_num = 20;

static void Help();
static void MakeDir(const std::string& dir);
static void MakeUsedDirs();

enum Florescence {
  FULLY_BLOOMED, NON_BLOOMED, PARTIALLY_BLOOMED,
  CLASS_CNT, CLASS_UNKNOWN = CLASS_CNT
};

struct Image {
  Image() : f_name(""), florescence(CLASS_UNKNOWN) {}
  Image(std::string p_f, Florescence p_florescence)
      : f_name(p_f), florescence(p_florescence) {}
  std::string f_name; // file name
  Florescence florescence; // a flag for different classes
};

static void DivideImagesIntoTrainTest(std::vector<Image>& images_train,
  std::vector<Image>& images_test);
static void InitImages(std::vector<Image>& images_train, std::vector<Image>& images_test);
static void SaveImages(const std::string& filename,
                       const std::vector<Image>& images_train,
                       const std::vector<Image>& images_test);
static bool readVocabulary(const std::string& filename, cv::Mat& vocabulary);
static bool WriteVocabulary(const std::string& filename,
                            const cv::Mat& vocabulary);
static cv::Mat TrainVocabulary(
    const std::string& filename,
    const cv::Ptr<cv::FeatureDetector>& fdetector,
    const cv::Ptr<cv::DescriptorExtractor>& dextractor,
    std::vector<Image>& images);

struct SVMTrainParamsExt
{
  SVMTrainParamsExt() : descPercent(1.f), targetRatio(0.4f), balanceClasses(true) {}
  SVMTrainParamsExt(float _descPercent, float _targetRatio, bool _balanceClasses) :
    descPercent(_descPercent), targetRatio(_targetRatio), balanceClasses(_balanceClasses) {}
  void Read(const cv::FileNode& fn)
  {
    fn["descPercent"] >> descPercent;
    fn["targetRatio"] >> targetRatio;
    fn["balanceClasses"] >> balanceClasses;
  }
  void Write(cv::FileStorage& fs) const
  {
    fs << "descPercent" << descPercent;
    fs << "targetRatio" << targetRatio;
    fs << "balanceClasses" << balanceClasses;
  }
  void Print() const
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

static bool readBowImageDescriptor(const std::string& file, cv::Mat& bowImageDescriptor);
static bool WriteBowImageDescriptor(const std::string& file, const cv::Mat& bowImageDescriptor);
static void CalculateImageDescriptors(const std::vector<Image>& images, std::vector<cv::Mat>& imageDescriptors,
  const cv::Ptr<cv::BOWImgDescriptorExtractor>& bow_extractor, const cv::Ptr<cv::FeatureDetector>& fdetector);
static void RemoveEmptyBowImageDescriptors(std::vector<Image>& images, std::vector<cv::Mat>& bow_img_descrs,
  std::vector<char>& obj_present);
static void RemoveEmptyBowImageDescriptors(std::vector<Image>& images, std::vector<cv::Mat>& bow_img_descrs);
static void setSVMParams(cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& responses, bool balanceClasses);
static void setSVMTrainAutoParams(cv::ml::ParamGrid& c_grid, cv::ml::ParamGrid& gamma_grid,
  cv::ml::ParamGrid& p_grid, cv::ml::ParamGrid& nu_grid,
  cv::ml::ParamGrid& coef_grid, cv::ml::ParamGrid& degree_grid);
static cv::Ptr<cv::ml::SVM> TrainSVMClassifier(const SVMTrainParamsExt& svmParamsExt, const std::string& objClassName,
  cv::Ptr<cv::BOWImgDescriptorExtractor>& bow_extractor, const cv::Ptr<cv::FeatureDetector>& fdetector,
  std::vector<Image>& images, std::vector<char>& obj_present);
static void ComputeConfidences(const cv::Ptr<cv::ml::SVM>& svm, const size_t class_idx,
  const cv::Ptr<cv::BOWImgDescriptorExtractor>& bow_extractor, const cv::Ptr<cv::FeatureDetector>& fdetector,
  std::vector<Image>& images, std::vector<std::array<float, CLASS_CNT>>& confidences);
//Write classifier results file
//-------------------------------------------
//INPUTS:
// - obj_class          The object class identifier string
// - images             An array of ObdImage containing the images for which data will be saved to the result file
// - scores             A corresponding array of confidence scores given a query
//NOTES:
// The result file path and filename are determined automatically using m_results_directory as a base
void WriteClassifierResultsFile(const std::vector<Image>& images,
  const std::vector<std::array<float, CLASS_CNT>>& confidences,
  const std::vector<Florescence>& florescences);
void CalculateResult(const std::vector<std::array<float, CLASS_CNT>>& confidences, std::vector<Florescence>& florescences);
void TrainTest();