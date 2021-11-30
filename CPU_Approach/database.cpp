#include <string>
#include <set>

#include "./setUtils.h"
#include "./database.h"

double FrequencyAnalysis::support(const TransactionMap& transactions, const ItemSet & itemSet){
    auto count = 0;
    for (const auto& entry:transactions){
        if (isSubset(itemSet,entry.second.items)){
            count++;
        }
    }
    return (double) count/transactions.size();
}

double FrequencyAnalysis::confidence(const TransactionMap& transactionMap, const ItemSet& antecedent, const ItemSet& consequent){
    auto ruleTrue = 0;
    auto antecedentAppearances = 0;

    for (const auto& transactionEntry:transactionMap){
        if (isSubset(antecedent, transactionEntry.second.items)){
            antecedentAppearances++;
            if (isSubset(consequent, transactionEntry.second.items)){
                ruleTrue++;
            }
        }
    }

    return (double) ruleTrue/antecedentAppearances;
}

ItemMap transform(const TransactionMap& transactionMap){
    ItemMap items;
    for (const auto&entry:transactionMap){
        for (const auto&item:entry.second.items){
            auto itemInMapIt = items.find(item);
            if (itemInMapIt == items.end()){
                items.insert(ItemMapPair(item,{entry.first}));
            }else{
                itemInMapIt->second.insert(entry.first);
            }
        }
    }
    return items;
}
