#pragma once

#include <chrono>

#include <cstddef>
#include <atomic>
class ProgressBar{
    private:
        size_t workToDo;
        std::atomic<uint32_t> workDone;
        std::chrono::_V2::system_clock::time_point startTime;
    public:
        ProgressBar(size_t totalWork);
        ~ProgressBar();
        void update(size_t amount = 1);
};