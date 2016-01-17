#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <vector>
#include <array>
#include "image.h"
#include "florescence.h"
#include "directory.h"

const int vocab_size = 50;

static void Help();

static bool readVocabulary(const std::string& filename, cv::Mat& vocabulary);
static bool WriteVocabulary(const std::string& filename,
                            const cv::Mat& vocabulary);
static cv::Mat TrainVocabulary(
    const std::string& filename,
    const cv::Ptr<cv::FeatureDetector>& fdetector,
    const cv::Ptr<cv::DescriptorExtractor>& dextractor,
    const std::vector<Image>& images);

struct SVMTrainParamsExt
{
  SVMTrainParamsExt() : desc_percent(1.f), target_ratio(0.4f), balance_classes(true) {}
  SVMTrainParamsExt(float _descPercent, float _targetRatio, bool _balanceClasses) :
    desc_percent(_descPercent), target_ratio(_targetRatio), balance_classes(_balanceClasses) {}
  void Read(const cv::FileNode& fn)
  {
    fn["desc_percent"] >> desc_percent;
    fn["target_ratio"] >> target_ratio;
    fn["balance_classes"] >> balance_classes;
  }
  void Write(cv::FileStorage& fs) const
  {
    fs << "desc_percent" << desc_percent;
    fs << "target_ratio" << target_ratio;
    fs << "balance_classes" << balance_classes;
  }
  void Print() const
  {
    std::cout << "desc_percent: " << desc_percent << "\n";
    std::cout << "target_ratio: " << target_ratio << "\n";
    std::cout << "balance_classes: " << balance_classes << "\n";
  }
  // If the file storing parameters of SVM trainer, read it, else creat it.
  void SVMTrainParamsExtFile();
  // Percentage of extracted descriptors to use for training.
  float desc_percent;
  // Try to get this ratio of positive to negative samples (minimum).
  float target_ratio;
  // Balance class weights by number of samples in each
  // (if true cSvmTrainTargetRatio is ignored).
  bool balance_classes;
};

static bool ReadBowImgDescriptor(const std::string& file, cv::Mat& bow_img_descriptor);
static bool WriteBowImageDescriptor(const std::string& file, const cv::Mat& bow_img_descriptor);
static void CalculateImageDescriptors(const std::vector<Image>& images, std::vector<cv::Mat>& img_descriptors,
  const cv::Ptr<cv::BOWImgDescriptorExtractor>& bow_extractor, const cv::Ptr<cv::FeatureDetector>& fdetector);
static void RemoveEmptyBowImageDescriptors(std::vector<Image>& images, std::vector<cv::Mat>& bow_img_descrs,
  std::vector<char>& obj_present);
static void RemoveEmptyBowImageDescriptors(std::vector<Image>& images, std::vector<cv::Mat>& bow_img_descrs);
static void setSVMParams(cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& responses, bool balance_classes);
static void SetSVMTrainAutoParams(cv::ml::ParamGrid& c_grid, cv::ml::ParamGrid& gamma_grid,
  cv::ml::ParamGrid& p_grid, cv::ml::ParamGrid& nu_grid,
  cv::ml::ParamGrid& coef_grid, cv::ml::ParamGrid& degree_grid);
static cv::Ptr<cv::ml::SVM> TrainSVMClassifier(const SVMTrainParamsExt& svm_params_ext, const std::string& class_name,
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