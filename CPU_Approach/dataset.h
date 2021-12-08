#pragma once
#include <set>
#include <string>
#include <vector>
#include <map>
#include <list>
#include <bitset>
#include <tuple>

#define MAX_NUM_ITEMS 128

typedef uint32_t ItemID;
// A bitmap of which item is active - up to 64 different items available
typedef std::bitset<MAX_NUM_ITEMS> ItemSet;
typedef uint32_t TransactionID;
typedef struct Transaction{
    TransactionID id;
    ItemSet items;
} Transaction;
typedef std::vector<Transaction> TransactionList;

namespace Dataset{
    TransactionList generate(
        const int & numClasses,
        const int & numTransactions,
        const double & skew,
        const int & maxTransactionSize,
        const int & minTransactionSize);
};