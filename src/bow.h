#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "image.h"
#include <vector>
#include <array>

bool ReadBowImgDescriptor(const std::string& file,
                          cv::Mat& bow_img_descriptor);
bool WriteBowImageDescriptor(const std::string& file,
                             const cv::Mat& bow_img_descriptor);
void CalculateImageDescriptors(
    const std::vector<Image>& images, std::vector<cv::Mat>& img_descriptors,
    const cv::Ptr<cv::BOWImgDescriptorExtractor>& bow_extractor,
    const cv::Ptr<cv::FeatureDetector>& fdetector);
void RemoveEmptyBowImageDescriptors(
    std::vector<Image>& images, std::vector<cv::Mat>& bow_img_descrs,
    std::vector<char>& obj_present,
    std::vector<std::array<float, CLASS_CNT>>& confidences);
void RemoveEmptyBowImageDescriptors(
    std::vector<Image>& images, std::vector<cv::Mat>& bow_img_descrs,
    std::vector<std::array<float, CLASS_CNT>>& confidences);
void ComputeConfidences(
    const cv::Ptr<cv::ml::SVM>& svm, const size_t class_idx,
    const cv::Ptr<cv::BOWImgDescriptorExtractor>& bow_extractor,
    const cv::Ptr<cv::FeatureDetector>& fdetector, std::vector<Image>& images,
    std::vector<std::array<float, CLASS_CNT>>& confidences);