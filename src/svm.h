#pragma once
#include "opencv2/opencv.hpp"
#include "image.h"

struct SVMTrainParamsExt {
  SVMTrainParamsExt() : desc_percent(1.f), target_ratio(0.4f),
                        balance_classes(true) {}
  SVMTrainParamsExt(float _descPercent, float _targetRatio,
                    bool _balanceClasses) :
      desc_percent(_descPercent), target_ratio(_targetRatio),
      balance_classes(_balanceClasses) {}
  void Read(const cv::FileNode& fn) {
    fn["desc_percent"] >> desc_percent;
    fn["target_ratio"] >> target_ratio;
    fn["balance_classes"] >> balance_classes;
  }
  void Write(cv::FileStorage& fs) const {
    fs << "desc_percent" << desc_percent;
    fs << "target_ratio" << target_ratio;
    fs << "balance_classes" << balance_classes;
  }
  void Print() const {
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

void setSVMParams(cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& responses,
                  bool balance_classes);
void SetSVMTrainAutoParams(
    cv::ml::ParamGrid& c_grid, cv::ml::ParamGrid& gamma_grid,
    cv::ml::ParamGrid& p_grid, cv::ml::ParamGrid& nu_grid,
    cv::ml::ParamGrid& coef_grid, cv::ml::ParamGrid& degree_grid);
cv::Ptr<cv::ml::SVM> TrainSVMClassifier(
    const SVMTrainParamsExt& svm_params_ext, const std::string& class_name,
    const cv::Ptr<cv::BOWImgDescriptorExtractor>& bow_extractor,
    const cv::Ptr<cv::FeatureDetector>& fdetector, std::vector<Image>& images,
    std::vector<char>& obj_present,
    std::vector<std::array<float, CLASS_CNT>>& confidences);