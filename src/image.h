#pragma once
#include <vector>
#include "florescence.h"

const size_t train_pic_num_each_class = 20;

struct Image {
  Image() : f_name(""), florescence(CLASS_UNKNOWN) {}
  Image(std::string p_f, Florescence p_florescence)
      : f_name(p_f), florescence(p_florescence) {}
  std::string f_name; // file name
  Florescence florescence; // a flag for different classes
};

void InitImages(std::vector<Image>& images_train,
                std::vector<Image>& images_test);
void SaveImages(const std::string& filename,
                const std::vector<Image>& images_train,
                const std::vector<Image>& images_test);