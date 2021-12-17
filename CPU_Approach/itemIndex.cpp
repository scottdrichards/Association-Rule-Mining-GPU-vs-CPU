#include <iterator> // advance
#include <algorithm> //sort
#include "itemIndex.h"
#include "bitsetUtils.h"

ItemIndex::ItemIndex(const std::vector<Transaction> & transactionsIn, const size_t & numItems){
    // First create a map 
    std::map<ItemID, std::vector<TransactionID>> map;
    for (const auto&transaction:transactionsIn){
        for (const auto&item: BitSetUtils::toIndices(transaction.items)){
            auto itemInMapIt = map.find(item);
            if (itemInMapIt == map.end()){
                map.insert({item,{transaction.id}});
            }else{
                itemInMapIt->second.push_back(transaction.id);
            }
        }
    }

    // Create the index
    this->index.resize(numItems);
    size_t databasePos = 0;
    for (ItemID itemID = 0; itemID< numItems; itemID++){
        IndexInfo index;
        index.begin = databasePos;

        const auto transactionIt = map.find(itemID);
        if (transactionIt != map.end()){
            const auto & transactions = (*transactionIt).second;
            this->database.insert(this->database.end(), transactions.begin(), transactions.end());
            databasePos += transactions.size();
        }

        index.end = databasePos;
        
        this->index[itemID] = index;
    }
}

std::pair<std::vector<TransactionID>::iterator,std::vector<TransactionID>::iterator> ItemIndex::getTransactionIterators(const ItemID & itemID){
    const auto & index = this->index[itemID];
    const auto beginIt = this->database.begin();
    const auto transactionStart = beginIt+index.begin;
    const auto transactionEnd = beginIt+index.end;
    return std::make_pair(transactionStart, transactionEnd);
}
