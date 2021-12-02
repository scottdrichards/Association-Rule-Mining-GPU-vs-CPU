#pragma once
#include <set>
#include <cstdint>

template<typename T>
std::set<T> intersection(const std::set<T> & a,const std::set<T> &b){
    std::set<T> result;
    if (b.size() == 0) return result;
    
    for (auto aIt = a.lower_bound(*b.begin()); aIt != a.upper_bound(*b.rbegin()); aIt++){
        if (b.find(*aIt)!=b.end()) result.insert(*aIt); 
    }
    return result;
}

template<typename T>
bool isSubset(const std::set<T> & subSet, const std::set<T> & superSet){
    for (const auto&item:subSet){
        if (superSet.find(item)==superSet.end()){
            return false;
        }
    }
    return true;
};