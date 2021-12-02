#include <iostream>
#include <algorithm> // find
#include <list>
#include <set>
#include <tuple>
#include "./dataset.h"
#include "./frequencyAnalysis.h"
#include "./progressBar.h"
#include "./frequents.h"

int main(int argc, char const *argv[])
{
    std::cout<<"Frequency Analysis - CPU"<<std::endl;
    if (argc!=7){
        std::cout<<"Expected 6 arguments"<<std::endl;
        return -1;
    }
    auto argvIndex = 1;
    const auto freqThreshold = std::stof(argv[argvIndex++]);
    const auto numClasses = std::stoi(argv[argvIndex++]);
    const auto numTransactions = std::stoi(argv[argvIndex++]);
    const auto skew = std::stof(argv[argvIndex++]);
    const auto maxTransactionSize = std::stoi(argv[argvIndex++]);
    const auto minTransactionSize = std::stoi(argv[argvIndex++]);

    TransactionList transactions;
    std::vector<Item> classVector;
    std::tie (transactions, classVector) = Dataset::generate(numClasses, numTransactions,skew, maxTransactionSize, minTransactionSize);

    std::set<ItemSet> curFrequents;
    std::for_each( classVector.begin(), classVector.end(), [&curFrequents](Item item){
        curFrequents.insert( ItemSet({item}));
    });

    auto itemTransactions = FrequencyAnalysis::transform(transactions);

    std::set<ItemSet> allFrequents;
    while (curFrequents.size()){
        bool useTranspose = false;
        if (useTranspose){
            curFrequents = Frequents::getFrequents(itemTransactions,transactions.size(),0,freqThreshold,curFrequents);
        }else{
            curFrequents = Frequents::getFrequents(transactions,0,freqThreshold,curFrequents);
        }
        allFrequents.insert(curFrequents.begin(), curFrequents.end());
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
