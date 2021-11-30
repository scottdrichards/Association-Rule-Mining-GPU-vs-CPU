#pragma once
#include <string>
#include <set>
#include <vector>

#include "./dataset.h"

namespace FrequencyAnalysis{
    double confidence(const TransactionMap& transactions, const ItemSet& antecedent, const ItemSet& consequent);
    double support(const TransactionMap& transactions, const ItemSet & itemSet);
    ItemMap transform(const TransactionMap& transactions);
}