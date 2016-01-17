#pragma once
#include <string>

#if defined WIN32 || defined _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max
#include "sys/types.h"
#endif
#include <sys/stat.h>

const std::string data_dir = "data/";
const std::string images_path = data_dir + "images.xml";
const std::string train_vocabulary_path = data_dir + "vocabulary.xml.gz";
const std::string svms_dir = data_dir + "svms";
const std::string bow_img_descriptors_dir = data_dir + "bow_img_descrs";
const std::string results_dir = data_dir + "results.txt";

static void MakeDir(const std::string& dir);
void MakeUsedDirs();