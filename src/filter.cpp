#include "filter.h"

using namespace cv;

void RemoveGreen(Mat& img) {
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      int b = img.at<Vec3b>(i, j)[0];
      int g = img.at<cv::Vec3b>(i, j)[1];
      int r = img.at<cv::Vec3b>(i, j)[2];
      //
      if ((g * 0.8 > r && g * 0.8 > b)||(r<b*1.15)||(b<70&&g<70&&r<70)||(r*0.55>g)) {
        img.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
      }
    }
  }
}