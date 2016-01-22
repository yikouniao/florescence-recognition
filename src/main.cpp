#include "train-test.h"

using namespace std;

int main(int argc, char** argv) {
  vector<Image> images_train, images_test;
  InitImages(images_train, images_test);
  SaveImages(images_path, images_train, images_test);

  TrainTest(images_train, images_test);
  
  return 0;
}