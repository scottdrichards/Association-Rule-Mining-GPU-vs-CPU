#pragma once
#include <string>
#include <set>
#include <vector>

typedef char Item;
typedef std::set<Item> ItemSet;

class Database{
    public:
        std::vector<ItemSet> transactions;
        ItemSet allItems;
        void add(const ItemSet& transaction);
        double confidence(const ItemSet& antecedent, const ItemSet& consequent);
        double support(const ItemSet & itemSet);
};