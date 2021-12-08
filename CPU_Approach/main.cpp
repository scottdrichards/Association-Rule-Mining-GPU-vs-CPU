#include <iostream>
#include <chrono> // keep track of how long each iteration takes
#include <algorithm> // find
#include <list>
#include <set>
#include <tuple>
#include <vector>
#include <thread>
#include <future>
#include "./dataset.h"
#include "./frequencyAnalysis.h"
#include "./progressBar.h"
#include "./frequents.h"
#include "./exclusiveRun.h"
#include "./bitsetUtils.h"
#include "./itemIndex.h"

// Takes a set of current frequents and generates candidate itemsets that have one extra
// Item
std::vector<ItemSet> generateCandidates(const std::vector<ItemSet> & curFrequents){
    // Find out which items were found in the frequents
    ItemSet activeItems = 0;    
    for (const auto & curFrequent: curFrequents){
        activeItems |= curFrequent;
    }

    std::vector<ItemSet> candidates;
    for (const auto & frequent: curFrequents){
        // Add another item ID to the itemset
        for (const auto itemID:itemSetToIDs(activeItems)){
            // If we already have it, move along, we can't add anything
            if (frequent.test(itemID)) continue;
            
            // Make a copy of it
            auto candidate = frequent;

            candidate.set(itemID);

            candidates.push_back(candidate);
        }
    }
    return candidates;
}

// Determines how frequent each item is and returns them in decreasing order
std::vector<ItemID> itemsByFrequency(const std::vector<ItemSet> & nextTests){
    // First we count the items to figure out which have the highest frequency
    std::map<ItemID,uint32_t> classCounts;
    for (const auto & test: nextTests){
        for (const auto & classID:itemSetToIDs(test)){
            auto classNode = classCounts.find(classID);
            if (classNode == classCounts.end()){
                classCounts.insert({classID,1});
            }else{
                classNode->second++;
            }
        }
    }
    
    // Next step we provide a list of items that is sorted by frequency
    std::vector<ItemID> items;

    // Extract just the items
    for (const auto & classCount:classCounts){
        items.push_back(classCount.first);
    }
    // Sort them by their counts
    sort(items.begin(), items.end(), [&classCounts](const ItemID &a, const ItemID &b){
        return std::greater<uint32_t>{}(classCounts[a],classCounts[b]);
    });
    return items;
}


int main(int argc, char const *argv[])
{
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

    TransactionList transactions = Dataset::generate(numClasses, numTransactions,skew, maxTransactionSize, minTransactionSize);

    std::vector<std::vector<ItemSet>> equivalenceClasses;
    for (auto i = 0; i<numClasses; i++){
        equivalenceClasses.push_back({(ItemSet{}).set(i)});
    }

    ItemIndex itemTransactions(transactions, numClasses);

    auto beginAll = std::chrono::high_resolution_clock::now();

    std::vector<ItemSet> allFrequents;
    std::chrono::duration<double, std::milli> processingMilliseconds;
    std::chrono::duration<double, std::milli> orchestratingMilliseconds;
    while (equivalenceClasses.size()){
        auto beginProcessing = std::chrono::high_resolution_clock::now();

        ///////////////////////////////////////////////////////////////////////////
        // Decomposition (divide up work into jobs)
        size_t candidateCount = 0;
        for (const auto & equivalenceClass:equivalenceClasses){
            candidateCount += equivalenceClass.size();
        }
        std::cout<<"Beginning to process "<<int(candidateCount)<<" candidates"<<std::endl;

        auto targetTestCount = candidateCount/numThreads;
        std::vector<std::vector<ItemSet>> jobs{{}};
        while(equivalenceClasses.size()){
            // Extract the equivalence class from the candidates
            const auto equivalenceClass = equivalenceClasses.front();
            equivalenceClasses.erase(equivalenceClasses.begin());

            // If the current job is too large, start a new work job
            if (jobs.back().size()>=targetTestCount)jobs.push_back({});

            // Add the equivalance class to most recent job
            auto jobIt = jobs.rbegin();
            (*jobIt).insert((*jobIt).end(),equivalenceClass.begin(),equivalenceClass.end());
        }


        ///////////////////////////////////////////////////////////////////////////
        // Orchestration (assign jobs to threads and get results)

        ProgressBar progressBar(candidateCount);
        auto callback = [&progressBar](const std::vector<ItemSet>& newFrequents){
            exclusiveRun([&progressBar](){
                progressBar.update(1);
            });
        };

        // Define the function we will have each thread run
        auto workProcess = [&](const std::vector<ItemSet> & candidates, std::vector<ItemSet>& result){
            Frequents::Job job = {};
            job.callback = callback;
            job.candidates = std::move(candidates);
            job.minFrequent = 0;
            job.minSupport = freqThreshold;
            
            if (indexTransactions){
                result = Frequents::getFrequents(itemTransactions, transactions.size(), job);
            }else{
                result = Frequents::getFrequents(transactions, job);            
            }
        };


        // Create the threadWatchers object right now, so that we can reference into it.
        // If we dynamically create it (push_back), the references might changeto the memory locations
        std::vector<std::pair<std::thread,std::vector<ItemSet>>> threadWatchers(jobs.size());   
        // Go through and get the jobs
        for (size_t i = 0; i<jobs.size(); i++){
            const auto & job = jobs.at(i);
            std::thread thread(workProcess,std::ref(job), std::ref(threadWatchers.at(i).second));
            threadWatchers.at(i).first = std::move(thread);
        }

        // Wait for each job to finish and accumulate. could orchestrate better and have threads pull work when free
        // but that might be too difficult for the minimal payoff
        std::vector<ItemSet> curFrequents;
        for (auto &threadResult:threadWatchers){
            threadResult.first.join();
            curFrequents.insert(curFrequents.end(),threadResult.second.begin(), threadResult.second.end());
        }
        allFrequents.insert(allFrequents.end(), curFrequents.begin(), curFrequents.end());

        auto endProcessing = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> processingTime = endProcessing-beginProcessing;
        processingMilliseconds += processingTime;
        auto beginOrchestrating = std::chrono::high_resolution_clock::now();


        ///////////////////////////////////////////////////////////////////////////
        // Create next tests (if possible)

        // Generate candidates based on the recently received frequents
        auto candidates = generateCandidates(curFrequents);

        auto orderedItems = itemsByFrequency(candidates);

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
        auto endOrchestrating = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> orchestrationTime = endOrchestrating-beginOrchestrating;
        orchestratingMilliseconds += orchestrationTime;
        std::cout<<"\nCompleted group, processing time: "<<int(processingMilliseconds.count())<<", orchestration: "<<int(orchestrationTime.count())<<std::endl;
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
    std::cout<<"\tProcess:               "<<int(processingMilliseconds.count())<<" ms"<<std::endl;
    std::cout<<"\tOrchestration:         "<<int(orchestratingMilliseconds.count())<<" ms"<<std::endl;

    return 0;
}
