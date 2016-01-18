#pragma once
#include "opencv2/opencv.hpp"
#include "image.h"

const int vocab_size = 20;

bool WriteVocabulary(const std::string& filename,
                     const cv::Mat& vocabulary);
cv::Mat TrainVocabulary(
    const std::string& filename,
    const cv::Ptr<cv::FeatureDetector>& fdetector,
    const cv::Ptr<cv::DescriptorExtractor>& dextractor,
    const std::vector<Image>& images);