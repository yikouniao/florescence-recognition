#include "SIFT.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void InitImages(vector<Image>& images) {
  for (int i = FULLY_BLOOMED; i < CLASS_CNT; ++i) {
    const string dir = data_dir + pic_dir[i];
    vector<String> filenames;
    glob(dir, filenames); // read a sequence of files within a folder
    for (size_t i = 0; i < filenames.size(); ++i) {
      images.push_back({filenames[i], static_cast<Florescence>(i), TEST});
    }
  }
}

static void SaveImages(const string& filename, const vector<Image>& images)
{
  cout << "Saving images...";
  FileStorage fs(filename, FileStorage::WRITE);
  fs << "images" << "["; // text - string sequence
  for (const auto& e : images) {
    if (e.datatype == TRAIN) {
      fs << e.f_name << "train";
    } else {
      fs << e.f_name << "test";
    }
  }
  fs << "]"; // close sequence
}

static bool readVocabulary(const string& filename, Mat& vocabulary)
{
  cout << "Reading vocabulary...";
  FileStorage fs(filename, FileStorage::READ);
  // comment out following lines to re-generate vocabularies each time
  // uncomment them to read existing vocabularies
  //if (fs.isOpened())
  //{
  //  fs["vocabulary"] >> vocabulary;
  //  cout << "done" << endl;
  //  return true;
  //}
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

static Mat trainVocabulary(const string& filename,
                           const Ptr<FeatureDetector>& fdetector,
                           const Ptr<DescriptorExtractor>& dextractor)
{
  Mat vocabulary;
  if (!readVocabulary(filename, vocabulary))
  {
    CV_Assert(dextractor->descriptorType() == CV_32FC1);

    cout << "Extracting data..." << endl;
    vector<Image> images;
    InitImages(images);

    cout << "Computing descriptors..." << endl;
    RNG& rng = theRNG();
    TermCriteria terminate_criterion;
    terminate_criterion.epsilon = FLT_EPSILON;
    BOWKMeansTrainer bowTrainer(vocab_size, terminate_criterion, 3, KMEANS_PP_CENTERS);

    size_t i = train_pic_num;
    while (i > 0)
    {
      // Randomly pick an image from the dataset which hasn't yet been seen
      // and compute the descriptors from that image.
      int randImgIdx = rng((unsigned)images.size());
      if (images[randImgIdx].datatype == TRAIN)
        continue;
      --i;
      images[randImgIdx].datatype = TRAIN;
      Mat colorImage = imread(images[randImgIdx].f_name);
      if (!colorImage.data) {
        cerr << images[randImgIdx].f_name << "can not be read.\n";
        exit(1);
      }
      vector<KeyPoint> imageKeypoints;
      fdetector->detect(colorImage, imageKeypoints);
      Mat imageDescriptors;
      dextractor->compute(colorImage, imageKeypoints, imageDescriptors);

      //check that there were descriptors calculated for the current image
      if (!imageDescriptors.empty())
      {
        for (int i = 0; i < imageDescriptors.rows; i++)
        {
            bowTrainer.add(imageDescriptors.row(i));
        }
      }
    }
    SaveImages(images_path, images);
    cout << "Actual descriptor count: " << bowTrainer.descriptorsCount() << endl;

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
  Mat vocabulary = trainVocabulary(train_vocabulary_path, featureDetector, descExtractor);
  bowExtractor->setVocabulary(vocabulary);

  // 2. Train a classifier and run a sample query for each object class
  // objClasses = {"bird","bicycle"...}
  //const vector<string>& objClasses = vocData.getObjectClasses(); // object class list
  const vector<string>& objClasses = pic_dir;
  for (size_t classIdx = 0; classIdx < objClasses.size(); ++classIdx)
  {
    // Train a classifier on train dataset
    Ptr<SVM> svm = trainSVMClassifier(svmTrainParamsExt, objClasses[classIdx], vocData,
      bowExtractor, featureDetector, resPath);

    // Now use the classifier over all images on the test dataset and rank according to score order
    // also calculating precision-recall etc.
    computeConfidences(svm, objClasses[classIdx], vocData,
      bowExtractor, featureDetector, resPath);
    // Calculate precision/recall/ap and use GNUPlot to output to a pdf file
    computeGnuPlotOutput(resPath, objClasses[classIdx], vocData);
  }
}