#include <string>
#include <set>

#include "./setUtils.h"
#include "./database.h"

double FrequencyAnalysis::support(const TransactionList& transactions, const ItemSet & itemSet){
    auto count = 0;
    for (const auto& transaction:transactions){
        if (isSubset(itemSet,transaction.items)){
            count++;
        }
    }
    return (double) count/transactions.size();
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

ItemMap transform(const TransactionList& transactions){
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
