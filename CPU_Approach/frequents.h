#pragma once
#include <list>
#include <functional>
#include "./dataset.h"



namespace Frequents{

    typedef std::function<void(std::vector<ItemSet>)> CallBack;

    typedef struct Job{
        float minFrequent;
        float minSupport;
        std::vector<ItemSet> candidates;
        CallBack callback;
    }Job;

    std::vector<ItemSet> getFrequents( const ItemIndex &itemMap, const size_t & transactionCount, Job & job);
    std::vector<ItemSet> getFrequents( const TransactionList &transactions, Job & job);
}