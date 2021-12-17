#include <string>
#include <set>
#include <algorithm> //copy-if

#include "./setUtils.h"
#include "./frequencyAnalysis.h"
#include "./bitsetUtils.h"


std::vector<TransactionID> mutualTransactions(TransactionList& transactions, const ItemSet & items){
    std::vector<TransactionID> result;
    for (const auto & transaction:transactions){
        if ((transaction.items & items) == items){
            result.push_back(transaction.id);
        }
    }
    return result;
}

size_t mutualTransactionCount(const TransactionList& transactions, const ItemSet & items){
    size_t count = 0;
    for (const auto & transaction:transactions){
        if ((transaction.items & items) == items) count++;
    }
    return count;
}


double FrequencyAnalysis::support(const TransactionList& transactions, const ItemSet & itemSet){
    auto sharedCount = mutualTransactionCount(transactions, itemSet);
    return (double) sharedCount/ transactions.size();
}

std::set<TransactionID> mutualTransactions(ItemIndex& itemIndex, const ItemSet & items){
    std::set<TransactionID> sharedTransactions;
    for (const auto& item:BitSetUtils::toIndices(items)){
        auto iterators = itemIndex.getTransactionIterators(item);
        auto beginTxns = iterators.first;
        auto endTxns = iterators.second;
        
        
        if (sharedTransactions.size() == 0){
            sharedTransactions.insert(beginTxns, endTxns);
        }else{
            sharedTransactions = intersection(sharedTransactions, beginTxns, endTxns);
            if (sharedTransactions.size()==0) return sharedTransactions;
        }
    }
    return sharedTransactions;
}

double FrequencyAnalysis::support(ItemIndex& itemMap, size_t numTransactions, const ItemSet & items){
    return (double)mutualTransactions(itemMap, items).size()/numTransactions;
}
