#pragma once

#include <vector>
#include "../dataset/dataset.h"

std::vector<ItemID> itemsByFrequency(const std::vector<ItemSet> & nextTests, uint8_t numThreads);