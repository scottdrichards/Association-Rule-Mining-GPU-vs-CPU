#pragma once
#include <string>
#include <set>
#include <vector>

#include "./dataset.h"

namespace FrequencyAnalysis{
    double confidence(const TransactionList& transactions, const ItemSet& antecedent, const ItemSet& consequent);
    double support(const ItemMap& itemMap, size_t numTransactions, const ItemSet & items);
    double support(const TransactionList& transactions, const ItemSet & items);

    ItemMap transform(const TransactionList& transactions);
}