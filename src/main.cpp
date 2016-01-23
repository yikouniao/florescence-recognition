#include "train-test.h"

using namespace std;

static void Help() {
  cout << "florescence-recognition\n\n"
       << "Recognize the florescence of corn in digital images.\n"
       << "There're three stages of florescence for corn: "
       << "non-bloomed, partially-bloomed and fully-bloomed. "
       << "The features of flowers in different florescences, such as "
       << "shape and color, can be useful in automatic recognition.\n"
       << "The code is based on OpenCV3.10.\n";
}

int main(int argc, char** argv) {
  Help();
  vector<Image> images_train, images_test;
  InitImages(images_train, images_test);
  SaveImages(images_path, images_train, images_test);

  TrainTest(images_train, images_test);
  
  return 0;
}