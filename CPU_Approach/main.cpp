#include <iostream>
#include <algorithm> // find
#include <list>
#include <set>
#include "./database.h"
#include "./progressBar.h"

#define NUM_TRANSACTIONS 100
#define MAX_TRANSACTION_SIZE 100
#define FREQ_THRESHOLD .4
static const char letters[]="abcdefghijklmnopqrstuvwxyz";

int main(int argc, char const *argv[])
{
    TransactionList transactions;
    ItemSet allItems;
    for (auto i = 0; i<NUM_TRANSACTIONS; i++){
        ItemSet transaction;
        for (auto j = 0; j<20+std::rand()%MAX_TRANSACTION_SIZE; j++){
            auto c = letters[std::rand()%26];
            transaction.insert(c);            
        }
        Database::add(transactions, transaction);
        allItems.insert(transaction.begin(),transaction.end());
    }
    std::cout<<"Added "<<transactions.size()<<" transactions including "<<allItems.size()<<" items"<<std::endl;

    // This list item is a set of itemsets. Each list item has increasing size
    std::list<std::set<ItemSet>> frequentTree;
    std::set<ItemSet> toProcess;
    std::cout<<"Processing frequents of size "<<1<<std::endl;
    uint32_t index = 0;
    for (auto item:allItems){
        progressBar((double)index++/allItems.size());
        ItemSet itemSet{item};
        auto support  = Database::support(transactions, itemSet);
        if (support > FREQ_THRESHOLD) toProcess.insert(itemSet);
    }
    progressBar(1);
    std::cout<<std::endl;


    while(toProcess.size()){
        auto currentSetSize = (*toProcess.begin()).size()+1;
        std::cout<<"Processing frequents of size "<<currentSetSize<<std::endl;
        frequentTree.push_back(toProcess);
        toProcess.clear();
        auto currentSets = frequentTree.back();
        ItemSet currentMembers;
        for (const auto &currentSet:currentSets){
            currentMembers.insert(currentSet.begin(),currentSet.end());
        }


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

                auto support  = Database::support(transactions, newSet);
                if (support > FREQ_THRESHOLD){
                    std::cout<<" "<<std::string(newSet.begin(),newSet.end());
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
        std::cout<<std::string(frequent.begin(),frequent.end());
        std::cout<<"}, ";
    }
    std::cout<<std::endl;

    return 0;
}
