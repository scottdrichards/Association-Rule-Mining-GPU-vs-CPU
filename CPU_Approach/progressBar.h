#pragma once
#include <cstddef>
class ProgressBar{
    private:
        size_t workToDo;
        size_t workDone;
    public:
        inline ProgressBar(size_t totalWork):workToDo(totalWork),workDone(0){
            this->workDone = 0;
            this->workToDo = totalWork;
        };
        void update(size_t amount = 1);
};