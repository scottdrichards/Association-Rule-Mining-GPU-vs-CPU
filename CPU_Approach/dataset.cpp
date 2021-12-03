#include <vector>
#include <random>
#include <tuple>
#include "dataset.h"


std::tuple<TransactionList,std::vector<Item>> Dataset::generate(
  const int & numClasses,
  const int & numTransactions,
  const double & skew,
  const int & maxTransactionSize,
  const int & minTransactionSize){
  std::string letters ="0123456789abcdef";

  std::random_device rd;
  std::mt19937 gen(rd());
  auto genSample = [&](double mean = 0){
    std::normal_distribution<> dist{mean,skew};
    int sample = round(dist(gen));
    sample %= numClasses;
    if (sample<0)sample = abs(sample+1);
    return sample;
  };


  std::vector<Item> classes;
  for (auto i = 0; i<numClasses; i++){
    Item cur;
    auto val = i;
    if (val == 0) cur = letters[0];
    while (val){
      auto mod = val%letters.size();
      cur = letters[mod] + cur;
      val -= mod;
      val /= letters.size();
    }
    classes.push_back(cur);
  };

  TransactionList transactions;
  for (auto i = 0; i<numTransactions; i++){
    auto itemCount = minTransactionSize+std::rand()%(maxTransactionSize-minTransactionSize);
    Transaction transaction;
    transaction.id = i;
    auto prevIndex = 0;
    for (auto j = 0; j<itemCount; j++){
      auto classIndex = genSample(prevIndex);
      prevIndex = classIndex;
      transaction.items.insert(classes[classIndex]);
    }
    transactions.push_back(transaction);
  }
  return std::tie(transactions, classes);
}
