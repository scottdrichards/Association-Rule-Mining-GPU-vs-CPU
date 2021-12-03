#pragma once
#include <list>
#include <functional>
#include "./dataset.h"



namespace Frequents{

    typedef std::function<void(std::set<ItemSet>)> CallBack;

    typedef struct Job{
        float minFrequent;
        float minSupport;
        std::set<ItemSet> prevFrequents;
        ItemSet candidateItems;
        CallBack callback;
        bool testPrevFreq;
    }Job;

    std::set<ItemSet> getFrequents( const ItemMap &itemMap, const size_t & transactionCount, Job job);
    std::set<ItemSet> getFrequents( const TransactionList &transactions, Job job);
}