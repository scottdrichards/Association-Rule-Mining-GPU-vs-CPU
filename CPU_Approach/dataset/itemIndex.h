#pragma once

#include <vector>
#include <map>
#include "dataset.h"

class ItemIndex{
    private:
        typedef struct IndexInfo{
            size_t begin;
            size_t end;
        }IndexInfo;
        std::vector<IndexInfo> index;
        std::vector<TransactionID> database;
    public:
        ItemIndex(const std::vector<Transaction> & transactions, const size_t & numItems);
        std::pair<std::vector<TransactionID>::iterator,std::vector<TransactionID>::iterator> getTransactionIterators(const ItemID & itemID);
};