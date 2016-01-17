#include "directory.h"
#include "florescence.h"

using namespace std;

static void MakeDir(const string& dir) {
#if defined WIN32 || defined _WIN32
  CreateDirectoryA(dir.c_str(), 0);
#else
  mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
}

void MakeUsedDirs() {
  MakeDir(svms_dir);
  MakeDir(bow_img_descriptors_dir);
  for (size_t i = 0; i < CLASS_CNT; ++i) {
    MakeDir(bow_img_descriptors_dir + "/" + obj_classes[i]);
  }
}