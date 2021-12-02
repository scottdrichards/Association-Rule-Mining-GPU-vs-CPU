#include <string>
#include <set>
#include <algorithm> //copy-if

#include "./setUtils.h"
#include "./frequencyAnalysis.h"


std::list<TransactionID> mutualTransactions(const TransactionList& transactions, const ItemSet & items){
    std::list<TransactionID> result;
    for (const auto & transaction:transactions){
        if (isSubset(items,transaction.items)){
            result.push_back(transaction.id);
        }
    }
    return result;
}

size_t mutualTransactionCount(const TransactionList& transactions, const ItemSet & items){
    size_t count = 0;
    for (const auto & transaction:transactions){
        if (isSubset(items,transaction.items)){
            count++;
        }
    }
    return count;
}


double FrequencyAnalysis::support(const TransactionList& transactions, const ItemSet & itemSet){
    auto sharedCount = mutualTransactionCount(transactions, itemSet);
    return (double) sharedCount/ transactions.size();
}

std::set<TransactionID> mutualTransactions(const ItemMap& itemMap, const ItemSet & items){
    std::set<TransactionID> sharedTransactions;
    for (const auto& item:items){
        auto itemMapIt = itemMap.find(item);

        std::set<TransactionID> transactions;
        if (itemMapIt != itemMap.end()) transactions = itemMapIt->second;
        
        if (sharedTransactions.size() == 0){
            sharedTransactions.insert(transactions.begin(),transactions.end());
        }else{
            sharedTransactions = intersection(sharedTransactions, transactions);
            if (sharedTransactions.size()==0) return sharedTransactions;
        }
    }
    return sharedTransactions;
}


double FrequencyAnalysis::support(const ItemMap& itemMap, size_t numTransactions, const ItemSet & items){
    return (double)mutualTransactions(itemMap, items).size()/numTransactions;
}

double FrequencyAnalysis::confidence(const TransactionList& transactions, const ItemSet& antecedent, const ItemSet& consequent){
    auto ruleTrue = 0;
    auto antecedentAppearances = 0;

    for (const auto& transaction:transactions){
        if (isSubset(antecedent, transaction.items)){
            antecedentAppearances++;
            if (isSubset(consequent, transaction.items)){
                ruleTrue++;
            }
        }
    }

    return (double) ruleTrue/antecedentAppearances;
}

ItemMap FrequencyAnalysis::transform(const TransactionList& transactions){
    ItemMap items;
    for (const auto&transaction:transactions){
        for (const auto&item:transaction.items){
            auto itemInMapIt = items.find(item);
            if (itemInMapIt == items.end()){
                items.insert(ItemMapPair(item,{transaction.id}));
            }else{
                itemInMapIt->second.insert(transaction.id);
            }
        }
    }
    return items;
}
