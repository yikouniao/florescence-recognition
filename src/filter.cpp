#include "filter.h"

using namespace cv;

void ImgPreProc(Mat& img) {
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      int b = img.at<Vec3b>(i, j)[0];
      int g = img.at<cv::Vec3b>(i, j)[1];
      int r = img.at<cv::Vec3b>(i, j)[2];
      if ((g * 0.78 > r && g * 0.78 > b) || // green leaves background
          r < b * 1.15 || // but not yellow flowers
          (b < 70 && g < 70 && r < 70) || // dark points, may be soil
          r * 0.6 > g) { // light red points, may be sticks
        img.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
      }
    }
  }
}