#pragma once
#include <list>
#include "./dataset.h"
namespace Frequents{
    std::set<ItemSet> getFrequents( const ItemMap &itemMap, const size_t & transactionCount, float minFrequent, float minSupport, std::set<ItemSet> prevFrequents);
    std::set<ItemSet> getFrequents( const TransactionList &transactions, float minFrequent, float minSupport, std::set<ItemSet> prevFrequents);
    
}