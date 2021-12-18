#include <iostream>
#include <algorithm>
#include <functional>
#include "./frequencyAnalysis.h"
#include "./frequents.h"
#include "../utils/progressBar.h"
#include "../utils/exclusiveRun.h"
#include "../utils/bitsetUtils.h"    

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

void Frequents::identifyFrequents(
            size_t candidateCount,
            std::vector<std::vector<ItemSet>> & testGroups,
            const float freqThreshold,bool indexTransactions,
            const TransactionList & transactions,
            ItemIndex &itemTransactions,
            std::vector<ItemSet> & newFrequents){


    std::cout<<"Identifying frequents within "<<int(candidateCount)<<" candidates"<<std::endl;
    ProgressBar progressBar(candidateCount, "IdentifyFrequents", true);


    typedef struct {
        std::thread thread;
        std::vector<ItemSet> frequents;
    } Job;
    std::vector<Job> jobs(testGroups.size());
    for (size_t i = 0; i<testGroups.size(); i++){
        const auto & testGroup = testGroups.at(i);
        auto & job = jobs.at(i);

        job.thread = std::thread(
            [&](const std::vector<ItemSet> & candidates, std::vector<ItemSet>& result){
                Frequents::Job job = {};
                job.callback = [&progressBar](const std::vector<ItemSet>& newFrequents, const size_t & iteration){
                        progressBar.increment();
                    };
                job.candidates = std::move(candidates);
                job.minFrequent = 0;
                job.minSupport = freqThreshold;
                
                // This is the core part!!!!!!
                if (indexTransactions){
                    result = Frequents::getFrequents(itemTransactions, transactions.size(), job);
                }else{
                    result = Frequents::getFrequents(transactions, job);            
                }
            },
        std::ref(testGroup), std::ref(job.frequents));
    }

    // Wait for each job to finish and accumulate. could orchestrate better and have threads pull work when free
    // but that might be too difficult for the minimal payoff
    for (auto &threadResult:jobs){
        threadResult.thread.join();
        newFrequents.insert(newFrequents.end(),threadResult.frequents.begin(), threadResult.frequents.end());
    }
}