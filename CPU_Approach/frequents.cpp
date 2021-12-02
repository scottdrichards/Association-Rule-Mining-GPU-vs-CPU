#include <functional>
#include "./frequencyAnalysis.h"
#include "./frequents.h"
#include "./progressBar.h"

std::set<ItemSet> getFrequentsGeneric(std::function<double(ItemSet)> supportFn, float minFrequent, float minSupport, std::set<ItemSet> prevFrequents){
    ItemSet currentItems;
    for (const auto &prevFrequent:prevFrequents){
        currentItems.insert(prevFrequent.begin(),prevFrequent.end());
    }

    std::set<ItemSet> frequents;
    std::set<ItemSet> tested;
    size_t index=0;
    for (auto frequentTest:prevFrequents){
        progressBar((double)index++/prevFrequents.size());

        for (const auto freqItem:currentItems){
            const auto originalSize = frequentTest.size();
            frequentTest.insert(freqItem);

            const auto itemAlreadyInSet = frequentTest.size() == originalSize;
            if (itemAlreadyInSet) continue;

            if (tested.find(frequentTest)!=tested.end()){
                frequentTest.erase(freqItem);
                continue;
            }

            double support = supportFn(frequentTest);
            if (support>minSupport) frequents.insert(frequentTest);
        }
    }
    progressBar();

    return frequents;
}


std::set<ItemSet> Frequents::getFrequents( const ItemMap &itemMap, const size_t & transactionCount, float minFrequent, float minSupport, std::set<ItemSet> prevFrequents){
    auto supportFn = [&itemMap, &transactionCount](ItemSet items){
        return FrequencyAnalysis::support(itemMap, transactionCount, items);
    };
    return getFrequentsGeneric(supportFn, minFrequent, minSupport, prevFrequents);
}

std::set<ItemSet> Frequents::getFrequents( const TransactionList &transactions, float minFrequent, float minSupport, std::set<ItemSet> prevFrequents){
    auto supportFn = [&transactions](ItemSet items){
        return FrequencyAnalysis::support(transactions,items);
    };
    return getFrequentsGeneric(supportFn, minFrequent, minSupport, prevFrequents);
}
