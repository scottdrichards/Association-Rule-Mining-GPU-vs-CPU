#pragma once
#include <string>
#include <set>

typedef std::string Item;
typedef std::set<Item> ItemSet;

class Database{
    private:
        std::set<ItemSet> db;
        ItemSet allItems;
    
    public:
        void add(const ItemSet& transaction);
        double confidence(const ItemSet& antecedent, const ItemSet& consequent);
        double support(const ItemSet & itemSet);

        std::size_t size();
};