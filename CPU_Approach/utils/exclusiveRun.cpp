#include <mutex>
#include "./exclusiveRun.h"

std::mutex runMutex{};

void exclusiveRun(std::function<void()> fn){
    runMutex.lock();
    fn();
    runMutex.unlock();
}
