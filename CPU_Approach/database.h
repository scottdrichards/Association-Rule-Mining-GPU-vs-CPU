#pragma once
#include <string>
#include <set>
#include <list>

typedef char Item;
typedef std::set<Item> ItemSet;
typedef std::list<ItemSet> TransactionList;

namespace Database{
    void add(TransactionList& transactions, const ItemSet& transaction);
    double confidence(const TransactionList& transactions, const ItemSet& antecedent, const ItemSet& consequent);
    double support(const TransactionList& transactions, const ItemSet & itemSet);
}