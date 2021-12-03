#include <iostream>
#include <functional>
#include "./frequencyAnalysis.h"
#include "./frequents.h"
#include "./progressBar.h"
#include "./exclusiveRun.h"
    
std::set<ItemSet> generateTests(const ItemSet &buildFrom, const ItemSet & candidateItems){
    std::set<ItemSet> tests;

    for (const auto candidateItem:candidateItems){
        auto test = buildFrom;
        test.insert(candidateItem);
        const auto testIsNew = buildFrom.size()!=test.size();
        if (testIsNew) tests.insert(test);         
    }
    return tests;
}


std::set<ItemSet> getFrequentsGeneric(std::function<double(ItemSet)> supportFn, Frequents::Job job){

    std::set<ItemSet> frequents;
    std::set<ItemSet> tested;
    for (auto prevFrequent:job.prevFrequents){

        std::set<ItemSet> tests;
        if (job.testPrevFreq){
            tests = {prevFrequent};
        }else{
            tests = generateTests(prevFrequent, job.candidateItems);
        }

        // Now that we have tests, go through them
        
        for (auto iterator = tests.begin(); iterator != tests.end(); ){
            auto curIt = iterator;
            iterator++;
            auto curTest = *curIt;
            
            if (tested.find(curTest)!=tested.end()){
                tests.erase(curIt);
                continue;
            }

            tested.insert(curTest);

            double support = supportFn(curTest);
            if (support<job.minSupport){
                tests.erase(curIt);
            }
        }
        job.callback(tests);
        frequents.insert(tests.begin(), tests.end());
    }
    return frequents;
}


std::set<ItemSet> Frequents::getFrequents( const ItemMap &itemMap, const size_t & transactionCount, Frequents::Job job){
    auto supportFn = [&itemMap, &transactionCount](ItemSet items){
        return FrequencyAnalysis::support(itemMap, transactionCount, items);
    };
    return getFrequentsGeneric(supportFn, job);
}

std::set<ItemSet> Frequents::getFrequents( const TransactionList &transactions, Frequents::Job job){
    auto supportFn = [&transactions](ItemSet items){
        return FrequencyAnalysis::support(transactions,items);
    };
    return getFrequentsGeneric(supportFn, job);
}
