#include "SIFT.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void MySIFT() {
  char* f = "data/Fully-bloomed/T0002_YM_20100818150220_01_9.jpg";
  Mat img = imread(f);
  if (!img.data) {
    cerr << f << "can not be read.\n";
    exit(1);
  }
  Mat gray;
  cvtColor(img, gray, COLOR_BGR2GRAY);
  namedWindow(f);
  imshow(f, gray);
  waitKey(0);
  Mat img1 = gray;
  Mat img2 = gray;
  //SIFT a();
  //std::vector<cv::KeyPoint> keypoints;
  //Mat descriptors;
  //a(gray, keypoints, descriptors, false);

  //cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

  //// Detect the keypoints:
  //std::vector<KeyPoint> keypoints;
  //f2d->detect(gray, keypoints);

  //// Calculate descriptors (feature vectors)    
  //Mat descriptors;
  //f2d->compute(gray, keypoints, descriptors);

  cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
  // Detect the keypoints:
  std::vector<KeyPoint> keypoints1, keypoints2;
  f2d->detect(img1, keypoints1);
  f2d->detect(img1, keypoints2);
  // Calculate descriptors (feature vectors)    
  Mat descriptors1, descriptors2;
  f2d->compute(gray, keypoints1, descriptors1);
  f2d->compute(gray, keypoints2, descriptors2);
  // Matching descriptor vectors using BFMatcher :
  BFMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match(descriptors1, descriptors2, matches);

  Mat outImg;
  drawMatches(img1, keypoints1, img2, keypoints2, matches, outImg);
  namedWindow("Match");
  imshow("Match", outImg);
  waitKey(0);
}

void test0() {
  Ptr<Feature2D> featureDetector = SIFT::create();
  Ptr<DescriptorExtractor> descExtractor = featureDetector;
  Ptr<BOWImgDescriptorExtractor> bowExtractor;
  if (!featureDetector || !descExtractor)
  {
    cout << "featureDetector or descExtractor was not created" << endl;
    exit(1);
  }
}