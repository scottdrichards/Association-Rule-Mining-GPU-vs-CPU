#include <string>
#include <set>

#include "./database.h"

void Database::add(TransactionList& transactions, const ItemSet& transaction){
    transactions.push_back(transaction);
}

bool isSubset(const ItemSet & subSet, const ItemSet & superSet){
    for (const auto&item:subSet){
        if (superSet.find(item)==superSet.end()){
            return false;
        }
    }
    return true;
}

double Database::support(const TransactionList& transactions, const ItemSet & itemSet){
    auto count = 0;
    for (const auto& transaction:transactions){
        if (isSubset(itemSet,transaction)){
            count++;
        }
    }
    return (double) count/transactions.size();
}

double Database::confidence(const TransactionList& transactions, const ItemSet& antecedent, const ItemSet& consequent){
    auto ruleTrue = 0;
    auto antecedentAppearances = 0;

    for (const auto& txn:transactions){
        if (isSubset(antecedent, txn)){
            antecedentAppearances++;
            if (isSubset(consequent, txn)){
                ruleTrue++;
            }
        }
    }

    return (double) ruleTrue/antecedentAppearances;
}