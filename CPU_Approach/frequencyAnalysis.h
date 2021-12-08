#pragma once
#include <string>
#include <set>
#include <vector>

#include "./dataset.h"
#include "ItemIndex.h"

namespace FrequencyAnalysis{
    double confidence(const TransactionList& transactions, const ItemSet& antecedent, const ItemSet& consequent);
    double support(ItemIndex& itemMap, size_t numTransactions, const ItemSet & items);
    double support(const TransactionList& transactions, const ItemSet & items);

    ItemIndex transform(const TransactionList& transactions);
}