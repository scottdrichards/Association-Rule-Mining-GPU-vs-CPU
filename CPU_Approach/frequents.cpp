#include <iostream>
#include <algorithm>
#include <functional>
#include "./frequencyAnalysis.h"
#include "./frequents.h"
#include "./progressBar.h"
#include "./exclusiveRun.h"
#include "./bitsetUtils.h"    

std::vector<ItemSet> getFrequentsGeneric(std::function<double(ItemSet)> supportFn, Frequents::Job & job){
    auto & candidates = job.candidates;
    size_t iteration = 0;
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),[&](ItemSet candidate)->bool{
        job.callback(candidates, iteration++);
        double support = supportFn(candidate);
        return support<job.minSupport;        
    }), candidates.end());
    return candidates;
}


std::vector<ItemSet> Frequents::getFrequents(ItemIndex &itemMap, const size_t & transactionCount, Frequents::Job & job){
    auto supportFn = [&itemMap, &transactionCount](ItemSet items){
        return FrequencyAnalysis::support(itemMap, transactionCount, items);
    };
    return getFrequentsGeneric(supportFn, job);
}

std::vector<ItemSet> Frequents::getFrequents( const TransactionList &transactions, Frequents::Job & job){
    auto supportFn = [&transactions](ItemSet items){
        return FrequencyAnalysis::support(transactions,items);
    };
    return getFrequentsGeneric(supportFn, job);
}
