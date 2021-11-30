#pragma once
#include <set>
#include <string>
#include <vector>
#include <map>
#include <list>
#include <tuple>

typedef std::string Item;
typedef std::set<Item> ItemSet;
typedef std::uint32_t TransactionID;
typedef struct Transaction{
    TransactionID id;
    ItemSet items;
} Transaction;
typedef std::list<Transaction> TransactionList;
typedef std::map<Item, std::set<TransactionID>> ItemMap;
typedef std::pair<Item, std::set<TransactionID>> ItemMapPair;

namespace Dataset{
    std::tuple<TransactionList,std::vector<Item>> generate(
        const int & numClasses,
        const int & numTransactions,
        const double & skew,
        const int & maxTransactionSize,
        const int & minTransactionSize);
};