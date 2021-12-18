#include <iostream>
#include <algorithm> // find
#include <list>
#include <set>
#include <tuple>
#include <vector>
#include <thread>
#include <future>
#include <string>
#include "dataset/dataset.h"
#include "frequency/frequencyAnalysis.h"
#include "frequency/frequents.h"
#include "dataset/itemIndex.h"
#include "preparation/preparation.h"
#include "preparation/itemsByFrequency.h"
#include "utils/progressBar.h"
#include "utils/bitsetUtils.h"
#include "utils/bitsetUtils.h"
#include "utils/log.h"


int main(int argc, char const *argv[])
{
    ProgressBar totalProgress(1, "total", false, true);
    std::cout<<"Frequency Analysis - CPU"<<std::endl;
    if (argc!=9){
        std::cout<<"Expected 8 arguments"<<std::endl;
        return -1;
    }
    auto argvIndex = 1;
    const auto freqThreshold = std::stof(argv[argvIndex++]);
    auto numClasses = std::stoi(argv[argvIndex++]);
    const auto numTransactions = std::stoi(argv[argvIndex++]);
    const auto skew = std::stof(argv[argvIndex++]);
    const auto maxTransactionSize = std::stoi(argv[argvIndex++]);
    const auto minTransactionSize = std::stoi(argv[argvIndex++]);
    unsigned int numThreads = std::stoi(argv[argvIndex++]);
    const bool indexTransactions = std::string(argv[argvIndex++])=="index";



    if (numThreads > std::thread::hardware_concurrency()){
        std::cout<<"!!! numThreads is higher than number of concurrent threads the system supports, setting to system max: ";
        std::cout<<std::thread::hardware_concurrency()<<std::endl;
        numThreads = std::thread::hardware_concurrency();
    }
    if (numClasses > MAX_NUM_ITEMS){
        std::cout<<"!!! num classes is higher than defined MAX_NUM_ITEMS, setting to max:"<<MAX_NUM_ITEMS<<std::endl;
        numClasses = MAX_NUM_ITEMS;
    }

    psuedoLogger.insert(PsuedoLogEntry("FrequencyThreshold", std::to_string(freqThreshold)));
    psuedoLogger.insert(PsuedoLogEntry("NumClasses", std::to_string(numClasses)));
    psuedoLogger.insert(PsuedoLogEntry("NumTransactions", std::to_string(numTransactions)));
    psuedoLogger.insert(PsuedoLogEntry("Skew", std::to_string(skew)));
    psuedoLogger.insert(PsuedoLogEntry("MaxTransactionSize", std::to_string(maxTransactionSize)));
    psuedoLogger.insert(PsuedoLogEntry("MinTransactionSize", std::to_string(minTransactionSize)));
    psuedoLogger.insert(PsuedoLogEntry("NumThreads", std::to_string(numThreads)));
    psuedoLogger.insert(PsuedoLogEntry("UseIndex", std::to_string(indexTransactions)));


    TransactionList transactions = Dataset::generate(numClasses, numTransactions,skew, maxTransactionSize, minTransactionSize);

    std::vector<std::vector<ItemSet>> equivalenceClasses;
    for (auto i = 0; i<numClasses; i++){
        equivalenceClasses.push_back({(ItemSet{}).set(i)});
    }

    ItemIndex itemTransactions(transactions, numClasses);

    auto beginAll = std::chrono::high_resolution_clock::now();

    std::vector<ItemSet> allFrequents;
    while (1){
        ///////////////////////////////////////////////////////////////////////////
        // Header
        size_t candidateCount = 0;
        for (const auto & equivalenceClass:equivalenceClasses) candidateCount += equivalenceClass.size();
        std::cout<<"\n"<<std::string(70,'/')+std::string(50,'/')<<std::endl;

        ///////////////////////////////////////////////////////////////////////////
        // Assign equivalence classes to groups
        auto idealTestCount = candidateCount/numThreads;
        std::vector<std::vector<ItemSet>> testGroups{{}};
        while(equivalenceClasses.size()){
            // Extract the equivalence class from the candidates
            const auto equivalenceClass = equivalenceClasses.front();
            equivalenceClasses.erase(equivalenceClasses.begin());

            // If the current job is too large, start a new work job
            if (testGroups.back().size()>=idealTestCount)testGroups.push_back({});

            // Add the equivalance class to most recent job
            auto jobIt = testGroups.rbegin();
            (*jobIt).insert((*jobIt).end(),equivalenceClass.begin(),equivalenceClass.end());
        }

        std::vector<ItemSet> newFrequents;

        Frequents::identifyFrequents(candidateCount, testGroups, freqThreshold, indexTransactions,transactions, itemTransactions, newFrequents);
        
        if (newFrequents.size()==0) break;

        allFrequents.insert(allFrequents.end(), newFrequents.begin(), newFrequents.end());

        ///////////////////////////////////////////////////////////////////////////
        // Create next tests (if possible)

        // Generate candidates based on the recently received frequents

        std::vector<ItemSet> candidates;
        generateCandidates(newFrequents, numThreads, candidates);

        if (candidates.size() == 0) break;

        auto orderedItems = itemsByFrequency(candidates, numThreads);

        equivalenceClasses.clear();
        // No go through and create equivalence classes
        for (const auto &item:orderedItems){
            std::vector<ItemSet> equivalenceClass;
            // Go through ALL the remaining frequents and find the correct ones, removing them as you go
            candidates.erase(std::remove_if(candidates.begin(),candidates.end(),[&](ItemSet candidate){
                
                // We only want the ones that include the item for the equivalence class
                if (!candidate.test(item)) return false;
                
                equivalenceClass.push_back(candidate);
                return true;
            }),candidates.end());
            if (equivalenceClass.size()>0){
                equivalenceClasses.push_back(equivalenceClass);
            }
        }
    }

    auto endAll = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> milliseconds = endAll-beginAll;

    std::cout<< std::string(60,'-') <<std::endl;
    std::cout<< std::string(25,'-') <<" COMPLETE "<<std::string(25,'-') <<std::endl;
    std::cout<<"Counted "<<allFrequents.size()<<" frequents."<<std::endl;

    std::cout<<"Configuration:"<<std::endl;
    std::cout<<"\tFrequency threshold:   "<<freqThreshold<<std::endl;
    std::cout<<"\tNumber of classes:     "<<numClasses<<std::endl;
    std::cout<<"\tNumber of Transactions:"<<numTransactions<<std::endl;
    std::cout<<"\tSkew (variance):       "<<skew<<std::endl;
    std::cout<<"\tTransactionSize (min): "<<minTransactionSize<<std::endl;
    std::cout<<"\tTransactionSize (max): "<<maxTransactionSize<<std::endl;
    std::cout<<"\tNumber of threads:     "<<numThreads<<" (system supports up to "<<std::thread::hardware_concurrency()<<")"<<std::endl;
    std::cout<<"\tIndex transactions?:   "<<(indexTransactions?"YES":"NO")<<std::endl;
    
    std::cout<<"Duration"<<std::endl;
    std::cout<<"\tTotal:                 "<<int(milliseconds.count())<<" ms"<<std::endl;
    std::cout<<"\tUtilization:           "<<int(milliseconds.count())*numThreads<<" ms*thread"<<std::endl;

    totalProgress.complete();

    printLog();
    return 0;
}
