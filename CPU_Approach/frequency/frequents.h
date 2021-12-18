#pragma once
#include <list>
#include <functional>
#include "../dataset/dataset.h"

namespace Frequents{

    typedef std::function<void(const std::vector<ItemSet> &, const size_t &)> CallBack;

    typedef struct Job{
        float minFrequent;
        float minSupport;
        std::vector<ItemSet> candidates;
        CallBack callback;
    }Job;

    std::vector<ItemSet> getFrequents( ItemIndex &itemMap, const size_t & transactionCount, Job & job);
    std::vector<ItemSet> getFrequents( const TransactionList &transactions, Job & job);

    
    void identifyFrequents(
                size_t candidateCount,
                std::vector<std::vector<ItemSet>> & testGroups,
                const float freqThreshold,bool indexTransactions,
                const TransactionList & transactions,
                ItemIndex &itemTransactions,
                std::vector<ItemSet> & newFrequents);
}