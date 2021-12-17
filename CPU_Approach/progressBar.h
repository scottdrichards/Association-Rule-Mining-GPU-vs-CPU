#pragma once

#include <cstddef>
#include <atomic>
class ProgressBar{
    private:
        size_t workToDo;
        std::atomic<uint32_t> workDone;
    public:
        inline ProgressBar(size_t totalWork):workToDo(totalWork),workDone(0){
            this->workDone = 0;
            this->workToDo = totalWork;
        };
        ~ProgressBar();
        void update(size_t amount = 1);
};