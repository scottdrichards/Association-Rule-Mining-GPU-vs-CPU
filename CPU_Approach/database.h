#pragma once
#include <string>
#include <set>
#include <vector>

#include "./dataset.h"

namespace FrequencyAnalysis{
    double confidence(const TransactionList& transactions, const ItemSet& antecedent, const ItemSet& consequent);
    double support(const TransactionList& transactions, const ItemSet & itemSet);
    ItemMap transform(const TransactionList& transactions);
}