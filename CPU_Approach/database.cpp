#include <string>
#include <set>

#include "./database.h"

void Database::add(const ItemSet& transaction){
    this->transactions.push_back(transaction);
    this->allItems.insert(transaction.begin(), transaction.end());
}

bool isSubset(const ItemSet & subSet, const ItemSet & superSet){
    for (const auto&item:subSet){
        if (superSet.find(item)==superSet.end()){
            return false;
        }
    }
    return true;
}

double Database::support(const ItemSet & itemSet){
    auto count = 0;
    for (const auto& transaction:this->transactions){
        if (isSubset(itemSet,transaction)){
            count++;
        }
    }
    return (double) count/this->transactions.size();
}

double Database::confidence(const ItemSet& antecedent, const ItemSet& consequent){
    auto ruleTrue = 0;
    auto antecedentAppearances = 0;

    for (const auto& txn:this->transactions){
        if (isSubset(antecedent, txn)){
            antecedentAppearances++;
            if (isSubset(consequent, txn)){
                ruleTrue++;
            }
        }
    }

    return (double) ruleTrue/antecedentAppearances;
}