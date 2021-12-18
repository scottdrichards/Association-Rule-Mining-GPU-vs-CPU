#include "./itemsByFrequency.h"

#include <vector>
#include <thread>
#include <iostream>
#include <algorithm>

#include "../utils/progressBar.h"
#include "../dataset/dataset.h"

// Determines how frequent each item is and returns them in decreasing order
std::vector<ItemID> itemsByFrequency(const std::vector<ItemSet> & nextTests, uint8_t numThreads){
    std::cout<<"Counting item frequencies"<<std::endl;

    // First we count the items to figure out which have the highest frequency
    std::vector<uint32_t> classCounts(MAX_NUM_ITEMS);
    ProgressBar progressBar(nextTests.size(), "CountItemByFreq");

    typedef struct{
        std::thread thread;
        std::vector<uint32_t> counts;
    } Job;
    std::vector<Job> jobs(numThreads);
    const auto numItemsPerJob = nextTests.size()/jobs.size();
    const auto numItemsPerPercent = nextTests.size()/100;
    for (size_t i = 0; i<jobs.size(); i++){
        auto & job = jobs.at(i);
        job.counts.resize(ItemSet().size());
        const auto beginIndex = numItemsPerJob*i;
        const auto endIndex = (i==jobs.size())?nextTests.size():beginIndex+numItemsPerJob;
        job.thread = std::thread(
            [&nextTests, &progressBar, &numItemsPerPercent](size_t begin, size_t endIndex, std::vector<uint32_t> &classCounts){
                size_t lastUpdateIndex = begin;
                for (auto testIndex = begin; testIndex<endIndex; testIndex++){
                    const auto & test = nextTests.at(testIndex);
                    for (auto classIndex = 0; classIndex<MAX_NUM_ITEMS; classIndex++){
                        if (test.test(classIndex)) classCounts[classIndex]++;
                    }
                    if (numItemsPerPercent &&(testIndex%numItemsPerPercent==0 || testIndex == endIndex-1 )){
                        progressBar.increment(testIndex-lastUpdateIndex);
                        lastUpdateIndex = testIndex;
                    }
                }
            },
            beginIndex, endIndex, std::ref(job.counts));
    }
    std::vector<uint32_t> counts(ItemSet().size());
    for(auto & job:jobs){
        job.thread.join();
        for(size_t i =0; i<counts.size();i++){
            counts[i] += job.counts[i];
        }
    }
    
    std::vector<ItemID> itemIDs;
    for (size_t i = 0; i<counts.size(); i++) itemIDs.push_back(i);

    std::sort(itemIDs.begin(), itemIDs.end(), [&counts](const ItemID & a, const ItemID & b){
        return std::greater<ItemID>{}(counts[a],counts[b]);
    });

    while(counts[itemIDs.back()] == 0) itemIDs.pop_back();
    return itemIDs;
}