#include <iostream>
#include <algorithm> // find
#include <list>
#include <set>
#include <tuple>
#include <thread>
#include <future>
#include "./dataset.h"
#include "./frequencyAnalysis.h"
#include "./progressBar.h"
#include "./frequents.h"
#include "./exclusiveRun.h"


int main(int argc, char const *argv[])
{
    std::cout<<"Frequency Analysis - CPU"<<std::endl;
    if (argc!=8){
        std::cout<<"Expected 7 arguments"<<std::endl;
        return -1;
    }
    auto argvIndex = 1;
    const auto freqThreshold = std::stof(argv[argvIndex++]);
    const auto numClasses = std::stoi(argv[argvIndex++]);
    const auto numTransactions = std::stoi(argv[argvIndex++]);
    const auto skew = std::stof(argv[argvIndex++]);
    const auto maxTransactionSize = std::stoi(argv[argvIndex++]);
    const auto minTransactionSize = std::stoi(argv[argvIndex++]);
    const auto numThreads = std::stoi(argv[argvIndex++]);

    TransactionList transactions;
    std::vector<Item> classVector;
    std::tie (transactions, classVector) = Dataset::generate(numClasses, numTransactions,skew, maxTransactionSize, minTransactionSize);

    std::set<ItemSet> curFrequents;
    std::for_each( classVector.begin(), classVector.end(), [&curFrequents](Item item){
        curFrequents.insert( ItemSet({item}));
    });

    // for(const auto & txn:transactions){
    //     std::cout<<txn.id<<": {";
    //     for (auto it = txn.items.begin(); it != txn.items.end(); it++){
    //         std::cout<<*it;
    //         if (std::next(it)!=txn.items.end()) std::cout<<",";
    //     }
    //     std::cout<<"}, ";
    // }
    auto itemTransactions = FrequencyAnalysis::transform(transactions);



    std::set<ItemSet> allFrequents;
    auto firstRun = true;
    auto curSize = 1;
    while (curFrequents.size()){
        std::vector<std::set<ItemSet>> workSets;
        ItemSet candidateItmes;
        for (auto i = 0; i<numThreads;i++) workSets.push_back({});
        auto i = 0;
        for (const auto & freq:curFrequents){
            workSets[(i++)%workSets.size()].insert(freq);
            candidateItmes.insert(freq.begin(), freq.end());
        }

        ProgressBar progressBar(curFrequents.size());
        auto callback = [&progressBar](std::set<ItemSet> newItems){
            exclusiveRun([&progressBar](){
                progressBar.update(1);
            });
        };

        auto workProcess = [&](const std::set<ItemSet> & myFrequents){
            Frequents::Job job = {};
            job.callback = callback;
            job.prevFrequents = myFrequents;
            job.candidateItems = candidateItmes;
            job.minFrequent = 0;
            job.minSupport = freqThreshold;
            job.testPrevFreq = firstRun;
            
            bool useTranspose = false;
            if (useTranspose){
                return Frequents::getFrequents(itemTransactions, transactions.size(), job);
            }
            return Frequents::getFrequents(transactions, job);            
        };

        std::list<std::future<std::set<ItemSet>>> promises;
        for (auto workSet:workSets){
            promises.push_back(std::async(workProcess, workSet));
        }
        curFrequents.clear();
        for (auto &promise:promises){
            auto result = promise.get();
            curFrequents.insert(result.begin(), result.end());
        }
        allFrequents.insert(curFrequents.begin(), curFrequents.end());
        std::cout<<"\nCompleted group of size "<<curSize++<<", there are currently "<<allFrequents.size()<<" frequents"<<std::endl;

        firstRun = false;
    }


    std::cout<<"Counted "<<allFrequents.size()<<" frequents:"<<std::endl;
    for (auto&frequent: allFrequents){
        std::cout<<"{";
        for (auto it = frequent.begin(); it != frequent.end(); it++){
            std::cout<<*it;
            if (std::next(it)!=frequent.end()) std::cout<<",";
        }
        std::cout<<"}, ";
    }
    std::cout<<std::endl;

    return 0;
}
