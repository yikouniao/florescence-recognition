#pragma once
#include <vector>

const std::vector<std::string> obj_classes {
  "Fully-bloomed",
  "Non-bloomed",
  "Partially-bloomed"
};

enum Florescence {
  FULLY_BLOOMED, NON_BLOOMED, PARTIALLY_BLOOMED,
  CLASS_CNT, CLASS_UNKNOWN = CLASS_CNT
};