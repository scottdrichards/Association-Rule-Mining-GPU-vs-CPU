#include "./preparation.h"

#include <iostream>
#include <vector>
#include <thread>
#include <string>
#include <algorithm>

#include "../utils/bitsetUtils.h"
#include "../utils/progressBar.h"
#include "../dataset/dataset.h"



ItemSet identifyActiveItems(const std::vector<ItemSet> & curFrequents){
    std::cout<<"Identifying Active Items"<<std::endl;
    ItemSet activeItems = 0;   
    ProgressBar progressBar(curFrequents.size());

    auto iteration = 0;
    for (const auto & curFrequent: curFrequents){
        activeItems |= curFrequent;

        if (activeItems.all()) return activeItems;

        progressBar.update(iteration);
    }

    std::cout<<std::endl<<"There are currently "<<activeItems.size()<<" active items"<<std::endl;
    return activeItems;
}

void generateCandidates(
        const std::vector<ItemSet> & curFrequents,
        const std::vector<std::size_t> & activeItems,
        uint8_t numThreads,
        std::vector<ItemSet> &candidates){

    std::cout<<"Generating Candidates, each having "<<BitSetUtils::toIndices(curFrequents.front()).size()<<" items"<<std::endl;

    typedef struct{
        std::thread thread;
        std::vector<ItemSet> itemSets;
    } Job;

    std::vector<Job> jobs(numThreads);
    auto itemsPerThread = curFrequents.size()/numThreads;
    ProgressBar progressBar(curFrequents.size());
    for (auto i = 0; i<numThreads; i++){
        auto & job = jobs[i];

        auto beginIndex = i*itemsPerThread;
        auto endIndex = (i+1)*itemsPerThread;

        const auto lastIteration = i == numThreads-1;
        if (lastIteration) endIndex = curFrequents.size();

        std::thread thread([&activeItems, &curFrequents, &progressBar](size_t beginIndex, size_t endIndex, std::vector<ItemSet> & candidates){
            for (auto cur = beginIndex; cur != endIndex; cur++){

                const auto & frequent = curFrequents.at(cur);
                // Add another item ID to the itemset
                for (const auto itemID:activeItems){
                    // If we already have it, move along, we can't add anything
                    if (frequent.test(itemID)) continue;
                    
                    // Make a copy of it
                    auto candidate = frequent;

                    candidate.set(itemID);

                    candidates.push_back(candidate);
                }         
                progressBar.increment();
            }
        }, beginIndex, endIndex, std::ref(job.itemSets));
        job.thread = std::move(thread);
    }

    for (auto & job: jobs){
        job.thread.join();
        candidates.insert(candidates.end(), job.itemSets.begin(), job.itemSets.end());
    }
    progressBar.complete("generated " + std::to_string(candidates.size()) +" candidates");
}


// Takes a set of current frequents and generates candidate itemsets that have one extra item
void generateCandidates(const std::vector<ItemSet> & curFrequents, uint8_t numThreads, std::vector<ItemSet> & candidates){

    const auto activeItems = BitSetUtils::toIndices(identifyActiveItems(curFrequents));
    
    generateCandidates(curFrequents, activeItems, numThreads, candidates);
}