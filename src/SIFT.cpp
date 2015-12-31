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

static bool readVocabulary(const string& filename, Mat& vocabulary)
{
  cout << "Reading vocabulary...";
  FileStorage fs(filename, FileStorage::READ);
  if (fs.isOpened())
  {
    fs["vocabulary"] >> vocabulary;
    cout << "done" << endl;
    return true;
  }
  return false;
}

static bool writeVocabulary(const string& filename, const Mat& vocabulary)
{
  cout << "Saving vocabulary..." << endl;
  FileStorage fs(filename, FileStorage::WRITE);
  if (fs.isOpened())
  {
    fs << "vocabulary" << vocabulary;
    return true;
  }
  return false;
}

static Mat trainVocabulary(const string& filename, /*VocData& vocData,*/ /*const VocabTrainParams& trainParams,*/
  const Ptr<FeatureDetector>& fdetector, const Ptr<DescriptorExtractor>& dextractor)
{
  Mat vocabulary;
  if (!readVocabulary(filename, vocabulary))
  {
    CV_Assert(dextractor->descriptorType() == CV_32FC1);
    //const int elemSize = CV_ELEM_SIZE(dextractor->descriptorType());
    //const int descByteSize = dextractor->descriptorSize() * elemSize;
    //const int bytesInMB = 1048576;
    //const int maxDescCount = (trainParams.memoryUse * bytesInMB) / descByteSize; // Total number of descs to use for training.

    cout << "Extracting VOC data..." << endl;
    vector<string> images;
    //vector<char> objectPresent;
    //vocData.getClassImages(trainParams.trainObjClass, CV_OBD_TRAIN, images, objectPresent);

    cout << "Computing descriptors..." << endl;
    RNG& rng = theRNG();
    TermCriteria terminate_criterion;
    terminate_criterion.epsilon = FLT_EPSILON;
    const int vocabSize = 50;
    BOWKMeansTrainer bowTrainer(vocabSize, terminate_criterion, 3, KMEANS_PP_CENTERS);

    while (images.size() > 0)
    {
//      if (bowTrainer.descriptorsCount() > maxDescCount)
//      {
//#ifdef DEBUG_DESC_PROGRESS
//        cout << "Breaking due to full memory ( descriptors count = " << bowTrainer.descriptorsCount()
//          << "; descriptor size in bytes = " << descByteSize << "; all used memory = "
//          << bowTrainer.descriptorsCount()*descByteSize << endl;
//#endif
//        break;
//      }

      // Randomly pick an image from the dataset which hasn't yet been seen
      // and compute the descriptors from that image.
      int randImgIdx = rng((unsigned)images.size());
      Mat colorImage = imread(images[randImgIdx]/*.path*/);
      vector<KeyPoint> imageKeypoints;
      fdetector->detect(colorImage, imageKeypoints);
      Mat imageDescriptors;
      dextractor->compute(colorImage, imageKeypoints, imageDescriptors);

      //check that there were descriptors calculated for the current image
      if (!imageDescriptors.empty())
      {
        int descCount = imageDescriptors.rows;
        // Extract trainParams.descProportion descriptors from the image, breaking if the 'allDescriptors' matrix becomes full
        //int descsToExtract = static_cast<int>(trainParams.descProportion * static_cast<float>(descCount));
        // Fill mask of used descriptors
        //vector<char> usedMask(descCount, false);
        //fill(usedMask.begin(), usedMask.begin() + descCount, true);
        //for (int i = 0; i < descCount; i++)
        //{
        //  int i1 = rng(descCount), i2 = rng(descCount);
        //  char tmp = usedMask[i1]; usedMask[i1] = usedMask[i2]; usedMask[i2] = tmp;
        //}

        for (int i = 0; i < descCount; i++)
        {
//          if (usedMask[i] && bowTrainer.descriptorsCount() < maxDescCount)
            bowTrainer.add(imageDescriptors.row(i));
        }
      }

//#ifdef DEBUG_DESC_PROGRESS
//      cout << images.size() << " images left, " << images[randImgIdx].id << " processed - "
//        <</* descs_extracted << "/" << image_descriptors.rows << " extracted - " << */
//        cvRound((static_cast<double>(bowTrainer.descriptorsCount()) / static_cast<double>(maxDescCount))*100.0)
//        << " % memory used" << (imageDescriptors.empty() ? " -> no descriptors extracted, skipping" : "") << endl;
//#endif

      // Delete the current element from images so it is not added again
      images.erase(images.begin() + randImgIdx);
    }

    cout << "Maximum allowed descriptor count: " << maxDescCount << ", Actual descriptor count: " << bowTrainer.descriptorsCount() << endl;

    cout << "Training vocabulary..." << endl;
    vocabulary = bowTrainer.cluster();

    if (!writeVocabulary(filename, vocabulary))
    {
      cout << "Error: file " << filename << " can not be opened to write" << endl;
      exit(-1);
    }
  }
  return vocabulary;
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
  {
    Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create("BruteForce");
    if (!featureDetector || !descExtractor || !descMatcher)
    {
      cout << "descMatcher was not created" << endl;
      exit(1);
    }
    bowExtractor = makePtr<BOWImgDescriptorExtractor>(descExtractor, descMatcher);
  }

  // 1. Train visual word vocabulary if a pre-calculated vocabulary file doesn't already exist from previous run
  string train_f = "data/train.xml";
  Mat vocabulary = trainVocabulary(train_f, vocData, vocabTrainParams,
    featureDetector, descExtractor);
}