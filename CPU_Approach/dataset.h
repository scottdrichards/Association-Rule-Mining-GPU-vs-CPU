#pragma once
#include <set>
#include <string>
#include <vector>
#include <map>
#include <tuple>

typedef std::string Item;
typedef std::set<Item> ItemSet;
typedef std::uint32_t TransactionID;
typedef struct Transaction{
    TransactionID id;
    ItemSet items;
} Transaction;
typedef std::map<TransactionID, Transaction> TransactionMap;
typedef std::map<Item, std::set<TransactionID>> ItemMap;
typedef std::pair<Item, std::set<TransactionID>> ItemMapPair;

namespace Dataset{
    std::tuple<TransactionMap,std::vector<Item>> generate(
        const int & numClasses,
        const int & numTransactions,
        const double & skew,
        const int & maxTransactionSize,
        const int & minTransactionSize);
};