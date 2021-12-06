#include <string>
#include <set>
#include <algorithm> //copy-if

#include "./setUtils.h"
#include "./frequencyAnalysis.h"
#include "./bitsetUtils.h"


std::vector<TransactionID> mutualTransactions(const TransactionList& transactions, const ItemSet & items){
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

std::set<TransactionID> mutualTransactions(const ItemIndex& itemMap, const ItemSet & items){
    std::set<TransactionID> sharedTransactions;
    for (const auto& item:itemSetToIDs(items)){
        auto itemMapIt = itemMap.find(item);

        if (itemMapIt == itemMap.end()) continue;
        
        std::set<TransactionID> transactions(itemMapIt->second.begin(), itemMapIt->second.end());
        
        if (sharedTransactions.size() == 0){
            sharedTransactions.insert(transactions.begin(),transactions.end());
        }else{
            sharedTransactions = intersection(sharedTransactions, transactions);
            if (sharedTransactions.size()==0) return sharedTransactions;
        }
    }
    return sharedTransactions;
}


double FrequencyAnalysis::support(const ItemIndex& itemMap, size_t numTransactions, const ItemSet & items){
    return (double)mutualTransactions(itemMap, items).size()/numTransactions;
}

ItemIndex FrequencyAnalysis::transform(const TransactionList& transactions){
    ItemIndex items;
    for (const auto&transaction:transactions){
        for (const auto&item: itemSetToIDs(transaction.items)){
            auto itemInMapIt = items.find(item);
            if (itemInMapIt == items.end()){
                items.insert({item,{transaction.id}});
            }else{
                itemInMapIt->second.push_back(transaction.id);
            }
        }
    }
    return items;
}
