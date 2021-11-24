#include <string>
#include <set>

#include "./database.h"

std::size_t Database::size(){
    return this->db.size();
}

void Database::add(const ItemSet& transaction){
    this->db.insert(transaction);
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
    for (const auto& transaction:this->db){
        if (isSubset(itemSet,transaction)){
            count++;
        }
    }
    return (double) count/this->db.size();
}

double Database::confidence(const ItemSet& antecedent, const ItemSet& consequent){
    auto ruleTrue = 0;
    auto antecedentAppearances = 0;

    for (const auto& txn:this->db){
        if (isSubset(antecedent, txn)){
            antecedentAppearances++;
            if (isSubset(consequent, txn)){
                ruleTrue++;
            }
        }
    }

    return (double) ruleTrue/antecedentAppearances;
}