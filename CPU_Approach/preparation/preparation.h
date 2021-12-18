#pragma once

#include <vector>
#include "../dataset/dataset.h"

void generateCandidates(const std::vector<ItemSet> & curFrequents, uint8_t numThreads, std::vector<ItemSet> & candidates);
