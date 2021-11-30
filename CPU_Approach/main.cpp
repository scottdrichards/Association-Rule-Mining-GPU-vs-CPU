#include <iostream>
#include <algorithm> // find
#include <list>
#include <set>
#include <tuple>
#include "./dataset.h"
#include "./database.h"
#include "./progressBar.h"

int main(int argc, char const *argv[])
{
    auto argvIndex = 1;
    const auto freqThreshold = std::stof(argv[argvIndex++]);
    const auto numClasses = std::stoi(argv[argvIndex++]);
    const auto numTransactions = std::stoi(argv[argvIndex++]);
    const auto skew = std::stof(argv[argvIndex++]);
    const auto maxTransactionSize = std::stoi(argv[argvIndex++]);
    const auto minTransactionSize = std::stoi(argv[argvIndex++]);

    TransactionMap transactions;
    std::vector<Item> classes;
    std::tie (transactions, classes) = Dataset::generate(numClasses, numTransactions,skew, maxTransactionSize, minTransactionSize);

    // This list item is a set of itemsets. Each list item has increasing size
    std::list<std::set<ItemSet>> frequentTree;
    std::set<ItemSet> toProcess;
    std::cout<<"Processing frequents of size "<<1<<std::endl;
    uint32_t index = 0;
    for (auto item:classes){
        progressBar((double)index++/classes.size());
        ItemSet itemSet{item};
        auto support  = FrequencyAnalysis::support(transactions, itemSet);
        if (support > freqThreshold) toProcess.insert(itemSet);
    }
    progressBar(1);
    std::cout<<std::endl;


    while(toProcess.size()){
        auto currentSetSize = (*toProcess.begin()).size()+1;
        frequentTree.push_back(toProcess);
        toProcess.clear();
        auto currentSets = frequentTree.back();
        ItemSet currentMembers;
        for (const auto &currentSet:currentSets){
            currentMembers.insert(currentSet.begin(),currentSet.end());
        }
        std::cout<<"Processing frequents of size "<<currentSetSize<<" having "<< currentMembers.size() <<" members"<<std::endl;


        uint32_t setIndex = 0;
        auto setLength = 1.0/currentSets.size();
        for (const auto & curSet:currentSets){
            auto setProgress = setIndex*setLength;
            setIndex++;            
            uint32_t tryIndex = 0;
            for (auto item:currentMembers){
                auto tryProgress = double(tryIndex)/currentMembers.size();
                tryIndex++;
                progressBar(setProgress+tryProgress*setLength);

                // Check to see if item was already in curset
                if (curSet.find(item) != curSet.end()) continue;

                ItemSet newSet;
                newSet.insert(curSet.begin(),curSet.end());
                newSet.insert(item);

                // Have we already done this combined set?
                if (toProcess.find(newSet) != toProcess.end()) continue;

                bool supportFromCurrentSets = true;
                for (auto removeIt: newSet){
                    if (removeIt == item) continue;
                    auto combinedCopy = newSet;
                    combinedCopy.erase(removeIt);
                    if (currentSets.find(combinedCopy)==currentSets.end()){
                        supportFromCurrentSets = false;
                        break;
                    }
                }

                if (!supportFromCurrentSets) continue;

                auto support  = FrequencyAnalysis::support(transactions, newSet);
                if (support > freqThreshold){
                    std::cout<<" ";
                    for (auto item:newSet) std::cout<<"\""<<item<<"\"";
                    std::cout<<"           ";
                    std::cout.flush();
                    toProcess.insert(newSet);
                };
            }

        }
        progressBar(1);
        // Erase current found
        if (toProcess.size()){
            std::cout<<std::string(currentSetSize+2,' ');
            std::cout.flush();
        }
        std::cout<< std::endl;
    }

    std::set<ItemSet> frequents;
    for (auto &frequentSet:frequentTree){
        frequents.insert(frequentSet.begin(),frequentSet.end());
    }

    std::cout<<"Counted "<<frequents.size()<<" frequents:"<<std::endl;
    for (auto&frequent: frequents){
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
