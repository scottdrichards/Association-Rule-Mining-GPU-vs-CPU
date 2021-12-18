#pragma once

#include <chrono>
#include <string>

#include <cstddef>
#include <atomic>
class ProgressBar{
    private:
        size_t workToDo;
        std::chrono::_V2::system_clock::time_point startTime;
        bool hasCleared;
        std::atomic<uint32_t> workDone;
        std::atomic<uint32_t> completedLastPrinted;
        void printProgress(double progress);
    public:
        ProgressBar(size_t totalWork);
        void update(size_t amount);
        void increment(size_t amount = 1);
        void complete(std::string str = "");
        ~ProgressBar();
};