#include <vector>
#include <random>
#include <tuple>
#include "dataset.h"

#define RNG_SEED 1

TransactionList Dataset::generate(
  const int & numClasses,
  const int & numTransactions,
  const double & skew,
  const int & maxTransactionSize,
  const int & minTransactionSize){

  std::mt19937 gen(RNG_SEED);
  auto genSample = [&](double mean = 0){
    std::normal_distribution<> dist{mean,skew};
    int sample = round(dist(gen));
    sample %= numClasses;
    if (sample<0)sample = abs(sample+1);
    return sample;
  };

  TransactionList transactions;
  for (auto i = 0; i<numTransactions; i++){
    auto itemCount = minTransactionSize+std::rand()%(maxTransactionSize-minTransactionSize);
    Transaction transaction;
    transaction.id = i;
    transaction.items = 0;
    
    // Have them all clustered around the first item
    auto firstIndex = genSample();
    transaction.items.set(firstIndex);
    for (auto j = 0; j<itemCount; j++){
      auto classIndex = genSample(firstIndex);
      transaction.items.set(classIndex);
    }
    transactions.push_back(transaction);
  }
  return transactions;
}
