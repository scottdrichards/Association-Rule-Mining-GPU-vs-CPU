#include <iostream>
#include <chrono> // keep track of how long each iteration takes
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
    unsigned int numThreads = std::stoi(argv[argvIndex++]);

    if (numThreads > std::thread::hardware_concurrency()){
        std::cout<<"!!! numThreads is higher than number of concurrent threads the system supports, setting to system max: ";
        std::cout<<std::thread::hardware_concurrency()<<std::endl;
        numThreads = std::thread::hardware_concurrency();
    }

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

    auto beginAll = std::chrono::high_resolution_clock::now();

    std::set<ItemSet> allFrequents;
    auto firstRun = true;
    auto curSize = 1;
    while (curFrequents.size()){
        auto beginIteration = std::chrono::high_resolution_clock::now();

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

        auto endIteration = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> milliseconds = endIteration-beginIteration;


        std::cout<<"\nCompleted group of size "<<curSize++<<", there are currently "<<allFrequents.size()<<" frequents"<<std::endl;
        std::cout<<"Duration: "<<milliseconds.count()<<"ms"<<std::endl;

        firstRun = false;
    }

    auto endAll = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> milliseconds = endAll-beginAll;

    std::cout<< std::string(60,'-') <<std::endl;
    std::cout<< std::string(25,'-') <<" COMPLETE "<<std::string(25,'-') <<std::endl;
    std::cout<<"Configuration:"<<std::endl;
    std::cout<<"\tFrequency threshold:   "<<freqThreshold<<std::endl;
    std::cout<<"\tNumber of classes:     "<<numClasses<<std::endl;
    std::cout<<"\tNumber of Transactions:"<<numTransactions<<std::endl;
    std::cout<<"\tSkew (variance):       "<<skew<<std::endl;
    std::cout<<"\tTransactionSize (min): "<<minTransactionSize<<std::endl;
    std::cout<<"\tTransactionSize (max): "<<maxTransactionSize<<std::endl;
    std::cout<<"\tNumber of threads:     "<<numThreads<<" (system supports up to "<<std::thread::hardware_concurrency()<<")"<<std::endl;
    std::cout<<"Duration"<<std::endl;
    std::cout<<"\tTotal:                 "<<int(milliseconds.count())<<" ms"<<std::endl;
    std::cout<<"\tUtilization:           "<<int(milliseconds.count())*numThreads<<" ms*thread"<<std::endl;

    std::cout<<"Counted "<<allFrequents.size()<<" frequents."<<std::endl;
    return 0;
}
