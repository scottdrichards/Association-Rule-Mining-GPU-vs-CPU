#include <iostream>
#include <algorithm> // find
#include <list>
#include <set>
#include "./database.h"
#define NUM_TRANSACTIONS 100000
#define MAX_TRANSACTION_SIZE 10
#define FREQ_THRESHOLD .4
static const char letters[]="abcdefghijklmnopqrstuvwxyz";

int main(int argc, char const *argv[])
{
    Database db;
    for (auto i = 0; i<NUM_TRANSACTIONS; i++){
        ItemSet transaction;
        for (auto j = 0; j<20+std::rand()%MAX_TRANSACTION_SIZE; j++){
            auto c = letters[std::rand()%26];
            transaction.insert(c);
            
        }
        db.add(transaction);
    }
    std::cout<<"Added "<<db.transactions.size()<<" transactions including "<<db.allItems.size()<<" items"<<std::endl;

    // This list item is a set of itemsets. Each list item has increasing size
    std::list<std::set<ItemSet>> frequentTree;
    std::set<ItemSet> toProcess;
    for (auto item:db.allItems){
        ItemSet itemSet{item};
        auto support  = db.support(itemSet);
        if (support > FREQ_THRESHOLD) toProcess.insert(itemSet);
    }

    while(toProcess.size()){
        frequentTree.push_back(toProcess);
        toProcess.clear();
        auto currentSets = frequentTree.back();
        ItemSet currentMembers;
        for (const auto &currentSet:currentSets){
            currentMembers.insert(currentSet.begin(),currentSet.end());
        }
        for (const auto & curSet:currentSets){
            for (auto item:currentMembers){
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

                auto support  = db.support(newSet);
                if (support > FREQ_THRESHOLD){
                    std::cout<<"Adding "<<std::string(newSet.begin(),newSet.end())<<std::endl;
                    toProcess.insert(newSet);
                };
            }
        }
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