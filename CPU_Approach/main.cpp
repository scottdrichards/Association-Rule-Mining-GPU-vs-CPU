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
#include "./bitsetUtils.h"

int main(int argc, char const *argv[])
{
    std::cout<<"Frequency Analysis - CPU"<<std::endl;
    if (argc!=8){
        std::cout<<"Expected 7 arguments"<<std::endl;
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

    if (numThreads > std::thread::hardware_concurrency()){
        std::cout<<"!!! numThreads is higher than number of concurrent threads the system supports, setting to system max: ";
        std::cout<<std::thread::hardware_concurrency()<<std::endl;
        numThreads = std::thread::hardware_concurrency();
    }
    if (numClasses > MAX_NUM_ITEMS){
        std::cout<<"!!! num classes is higher than defined MAX_NUM_ITEMS, setting to max:"<<MAX_NUM_ITEMS<<std::endl;
        numClasses = MAX_NUM_ITEMS;
    }

    TransactionList transactions = Dataset::generate(numClasses, numTransactions,skew, maxTransactionSize, minTransactionSize);

    std::vector<std::vector<ItemSet>> initialEquivalenceClass;
    for (auto i = 0; i<numClasses; i++){
        initialEquivalenceClass.push_back({(ItemSet{}).set(i)});
    }

    auto itemTransactions = FrequencyAnalysis::transform(transactions);

    auto beginAll = std::chrono::high_resolution_clock::now();


    auto firstRun = true;
    auto curSize = 1;
    std::vector<ItemSet> allFrequents;
    std::vector<ItemSet> curFrequents;
    while (firstRun || curFrequents.size()){
        auto totalWork = firstRun? numClasses:curFrequents.size();
        auto beginIteration = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<ItemSet>> equivalenceClasses;
        if (firstRun){
            equivalenceClasses = initialEquivalenceClass;
        }else{
            std::map<ItemID,uint32_t> classCounts;
            for (const auto & frequent: curFrequents){
                for (const auto & classID:itemSetToIDs(frequent)){
                    auto classNode = classCounts.find(classID);
                    if (classNode == classCounts.end()){
                        classCounts.insert({classID,1});
                    }else{
                        classNode->second++;
                    }
                }
            }

            // items will be sorted such that the lowest number will have the greater number of occurrences
            std::vector<ItemID> items;
            for (const auto & classCount:classCounts){
                items.push_back(classCount.first);
            }
            sort(items.begin(), items.end(), [&classCounts](const ItemID &a, const ItemID &b){
                return std::greater<uint32_t>{}(classCounts[a],classCounts[b]);
            });

            for (auto itemIt = items.begin(); itemIt != items.end(); itemIt++){
                auto itemID = *itemIt;
                std::vector<ItemSet> equivalenceClass;
                // Go through ALL the remaining frequents and find the correct ones, removing them as you go
                curFrequents.erase(std::remove_if(curFrequents.begin(),curFrequents.end(),[&](ItemSet curFrequent){
                    // Does not apply to this itemID
                    if (!curFrequent.test(itemID)) return false;
                    
                    // ... go through possible candidates adding them as necessary
                    for (auto otherIt = itemIt+1; otherIt != items.end(); otherIt++){
                        auto candidateSet = curFrequent;
                        const auto & otherID = *otherIt;
                        // If the set already includes the ID, just keep moving
                        if (candidateSet[otherID]) continue;
                        // Otherwise add it to the equivalence class
                        candidateSet.set(otherID);
                        equivalenceClass.push_back(candidateSet);
                    }
                    return true;
                }),curFrequents.end());
                
                equivalenceClasses.push_back(equivalenceClass);
            }
        }
        
        size_t totalTests = 0;
        for (const auto & equivalenceClass:equivalenceClasses){
            totalTests += equivalenceClass.size();
        }

        auto targetTestCount = totalTests/numThreads;
        std::vector<std::vector<ItemSet>> workSets{{}};
        while(equivalenceClasses.size()){
            // If the current job is too large, start a new work set
            if (workSets.back().size()>=targetTestCount)workSets.push_back({});

            // Copy the equivalence class to the job
            const auto & equivalenceClass = equivalenceClasses.front();
            auto & job = workSets.back();
            job.insert(job.end(),equivalenceClass.begin(),equivalenceClass.end());

            // Remove the equivalence class from the candidates
            equivalenceClasses.erase(equivalenceClasses.begin());
        }

        ProgressBar progressBar(totalWork);
        auto callback = [&progressBar](std::vector<ItemSet> newItems){
            exclusiveRun([&progressBar](){
                progressBar.update(1);
            });
        };

        auto workProcess = [&](const std::vector<ItemSet> & candidates, bool useTranspose = true){
            Frequents::Job job = {};
            job.callback = callback;
            job.candidates = candidates;
            job.minFrequent = 0;
            job.minSupport = freqThreshold;
            
            if (useTranspose){
                return Frequents::getFrequents(itemTransactions, transactions.size(), job);
            }
            return Frequents::getFrequents(transactions, job);            
        };

        std::list<std::future<std::vector<ItemSet>>> promises;
        for (auto workSet:workSets){
            promises.push_back(std::async(workProcess, workSet));
        }
        curFrequents.clear();
        for (auto &promise:promises){
            auto result = promise.get();
            curFrequents.insert(curFrequents.end(),result.begin(), result.end());
        }
        allFrequents.insert(allFrequents.end(), curFrequents.begin(), curFrequents.end());

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
