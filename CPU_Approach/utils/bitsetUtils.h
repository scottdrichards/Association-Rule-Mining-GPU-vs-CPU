#pragma once

#include <bitset>
#include <vector>
#include <thread>
#include <iterator>

namespace BitSetUtils{
    template<std::size_t BitSetSize>
    std::vector<std::size_t> toIndices(std::bitset<BitSetSize> bitSet){        
        std::vector<std::size_t> indices(bitSet.count());
        std::size_t bitIndex = bitSet._Find_first();
        size_t i = 0;
        while (bitIndex != bitSet.size()){
            indices[i++] = bitIndex;
            bitSet.set(bitIndex, false);
            bitIndex = bitSet._Find_first();
        }
        return indices;
    }

    template<std::size_t BitSetSize>
    std::bitset<BitSetSize> vectorOR(const std::vector<std::bitset<BitSetSize>> & bitsetVector, size_t threadCount = 0){
        typedef std::vector<std::bitset<BitSetSize>> BitSetVector;
        if (threadCount == 0) threadCount = std::thread::hardware_concurrency();
        // Just in case hardware_concurrency is 0
        if (threadCount == 0) threadCount = 1;

        typedef struct {
            std::thread thread;
            std::bitset<BitSetSize> bitset;
         } Job;

        auto workFn = [](typename BitSetVector::iterator begin, typename BitSetVector::iterator end,  std::bitset<BitSetSize> & result ){
            for (auto it = begin; it != end; it++){
                result |= *it;
            }
        };
        
        std::vector<Job> jobs(threadCount);
        auto itemsPerThread = bitsetVector.size()/threadCount;
        for (auto threadNumber = 0; threadNumber<threadCount; threadNumber++){
            const Job & job = jobs.at(threadNumber);
            const auto start = bitsetVector.begin()+(itemsPerThread*threadNumber);

            const auto isLast = (threadNumber==threadCount-1);
            const auto end = isLast?bitsetVector.end():start + itemsPerThread;
            job.thread = std::thread(workFn, start, end, std::ref(job.bitset));
        }

        std::bitset<BitSetSize> result;
        for (const Job & job:jobs){
            job.thread.join();
            result &= job.bitset;
        }

        return result;
    };

    template<size_t BitSetSize>
    std::bitset<BitSetSize> vectorAND(const std::vector<std::bitset<BitSetSize>> & bitsetVector){
        std::bitset<BitSetSize> result = 0;    
        for (const auto & bitset: bitsetVector){
            result &= bitset;
        }
        return result;
    }
}