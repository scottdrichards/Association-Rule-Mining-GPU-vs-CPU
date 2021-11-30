#pragma once
#include <set>
#include <cstdint>

typedef struct IntersectCounts{
    uint32_t left;
    uint32_t overlap;
    uint32_t right;
} IntersectCounts;

template<typename T>
IntersectCounts intersection(const std::set<T> & a,const std::set<T> &b){
    IntersectCounts results{0,0,0};
    for (auto aIt = a.lower_bound(*b.begin()); aIt != a.upper_bound(*b.rbegin()); aIt++){
        if (b.find(*aIt)!=b.end()) results.overlap++; 
    }
    results.left = a.size()-results.overlap;
    results.right = b.size()-results.overlap;
    return results;
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