#include "SIFT.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;
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

static bool readBowImageDescriptor(const string& file, Mat& bowImageDescriptor)
{
  FileStorage fs(file, FileStorage::READ);
  if (fs.isOpened())
  {
    fs["imageDescriptor"] >> bowImageDescriptor;
    return true;
  }
  return false;
}

static bool writeBowImageDescriptor(const string& file, const Mat& bowImageDescriptor)
{
  FileStorage fs(file, FileStorage::WRITE);
  if (fs.isOpened())
  {
    fs << "imageDescriptor" << bowImageDescriptor;
    return true;
  }
  return false;
}

// Load in the bag of words vectors for a set of images, from file if possible
static void calculateImageDescriptors(const vector<Image>& images, vector<Mat>& imageDescriptors,
  Ptr<BOWImgDescriptorExtractor>& bowExtractor, const Ptr<FeatureDetector>& fdetector)
{
  CV_Assert(!bowExtractor->getVocabulary().empty());
  imageDescriptors.resize(images.size());

  for (size_t i = 0; i < images.size(); i++)
  {
    string filename = bowImageDescriptorsDir + "/" + images[i].f_name + ".xml.gz";
    // uncomment the following line if want to read existing BowImageDescriptor
    //if (readBowImageDescriptor(filename, imageDescriptors[i])) {}

    Mat colorImage = imread(images[i].f_name);
    vector<KeyPoint> keypoints;
    fdetector->detect(colorImage, keypoints);
    bowExtractor->compute(colorImage, keypoints, imageDescriptors[i]);
    if (!imageDescriptors[i].empty())
    {
      if (!writeBowImageDescriptor(filename, imageDescriptors[i]))
      {
        cout << "Error: file " << filename << "can not be opened to write bow image descriptor" << endl;
        exit(-1);
      }
    }
  }
}

static Ptr<SVM> trainSVMClassifier(const SVMTrainParamsExt& svmParamsExt, const string& objClassName, VocData& vocData,
  Ptr<BOWImgDescriptorExtractor>& bowExtractor, const Ptr<FeatureDetector>& fdetector)
{
  /* first check if a previously trained svm for the current class has been saved to file */
  string svmFilename = svms_dir + "/" + objClassName + ".xml.gz";
  Ptr<SVM> svm;

  FileStorage fs(svmFilename, FileStorage::READ);

  // uncomment the following lines to read existing svm data if available
  //if (fs.isOpened())
  //{
  //  cout << "LOADING SVM CLASSIFIER FOR CLASS " << objClassName << "\n";
  //  svm = StatModel::load<SVM>(svmFilename);
  //  return svm;
  //}

  cout << "TRAINING CLASSIFIER FOR CLASS " << objClassName << "\n";
  cout << "CALCULATING BOW VECTORS FOR TRAINING SET OF " << objClassName << "\n";

  // Get classification ground truth for images in the training set
  vector<Image> images;
  vector<Mat> bowImageDescriptors;

  // make images. each pic belongs to some class. maybe.
  //vocData.getClassImages(objClassName, CV_OBD_TRAIN, images, objectPresent);

  // Compute the bag of words vector for each image in the training set.
  calculateImageDescriptors(images, bowImageDescriptors, bowExtractor, fdetector);

  // Remove any images for which descriptors could not be calculated
  removeEmptyBowImageDescriptors(images, bowImageDescriptors);

  // Prepare the input matrices for SVM training.
  Mat trainData((int)images.size(), bowExtractor->getVocabulary().rows, CV_32FC1);
  Mat responses((int)images.size(), 1, CV_32SC1);

  // Transfer bag of words vectors and responses across to the training data matrices
  for (size_t imageIdx = 0; imageIdx < images.size(); imageIdx++)
  {
    // Transfer image descriptor (bag of words vector) to training data matrix
    Mat submat = trainData.row((int)imageIdx);
    if (bowImageDescriptors[imageIdx].cols != bowExtractor->descriptorSize())
    {
      cout << "Error: computed bow image descriptor size " << bowImageDescriptors[imageIdx].cols
        << " differs from vocabulary size" << bowExtractor->getVocabulary().cols << endl;
      exit(-1);
    }
    bowImageDescriptors[imageIdx].copyTo(submat);

    // Set response value
    responses.at<int>((int)imageIdx) = objectPresent[imageIdx] ? 1 : -1;
  }

  cout << "TRAINING SVM FOR CLASS ..." << objClassName << "..." << endl;
  svm = SVM::create();
  setSVMParams(svm, responses, svmParamsExt.balanceClasses);
  ParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
  setSVMTrainAutoParams(c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);

  svm->trainAuto(TrainData::create(trainData, ROW_SAMPLE, responses), 10,
    c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);
  cout << "SVM TRAINING FOR CLASS " << objClassName << " COMPLETED" << endl;

  svm->save(svmFilename);
  cout << "SAVED CLASSIFIER TO FILE" << endl;

  return svm;
}

void SVMTrainParamsExt::SVMTrainParamsExtFile() {
  string svmFilename = svms_dir + "/svm.xml";

  FileStorage paramsFS(svmFilename, FileStorage::READ);
  if (paramsFS.isOpened())
  {
    FileNode fn = paramsFS.root();
    FileNode currFn = fn;
    currFn = fn["svmTrainParamsExt"];
    read(currFn);
  }
  else
  {
    paramsFS.open(svmFilename, FileStorage::WRITE);
    if (paramsFS.isOpened())
    {
      paramsFS << "svmTrainParamsExt" << "{";
      write(paramsFS);
      paramsFS << "}";
      paramsFS.release();
    }
    else
    {
      cout << "File " << svmFilename << "can not be opened to write" << endl;
      exit(1);
    }
  }
}

static void removeEmptyBowImageDescriptors(vector<Image>& images, vector<Mat>& bowImageDescriptors)
{
  CV_Assert(!images.empty());
  for (int i = (int)images.size() - 1; i >= 0; i--)
  {
    //bool res = bowImageDescriptors[i].empty();
    //if (res)
    //?????????????????
    if (bowImageDescriptors[i].empty())
    {
      cout << "Removing image " << images[i].f_name << " due to no descriptors..." << endl;
      images.erase(images.begin() + i);
      bowImageDescriptors.erase(bowImageDescriptors.begin() + i);
    }
  }
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
  SVMTrainParamsExt svmTrainParamsExt;
  svmTrainParamsExt.SVMTrainParamsExtFile();
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