#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <vector>
#include <array>
#include "image.h"
#include "florescence.h"
#include "directory.h"
#include "vocabulary.h"
#include "svm.h"
#include "bow.h"

static void Help();

void WriteClassifierResultsFile(
    const std::vector<Image>& images,
    const std::vector<std::array<float, CLASS_CNT>>& confidences,
    const std::vector<Florescence>& florescences);
void CalculateResult(
    const std::vector<std::array<float, CLASS_CNT>>& confidences,
    std::vector<Florescence>& florescences);
void TrainTest();