#include "progressBar.h"

#include <chrono>
#include <iostream>
#include <string>
#include <stdint.h> 
#include "./exclusiveRun.h"
#include "log.h"

#define BAR_WIDTH 70


ProgressBar::ProgressBar(size_t totalWork, std::string label, bool isParallel, bool silent){
    this->workDone = 0;
    this->workToDo = totalWork;
    this->completedLastPrinted = 0;
    this->hasCleared = false;
    this->startTime = std::chrono::high_resolution_clock::now();
    this->label = label;
    this->isParallel = isParallel;
    this->silent = silent;
}

void ProgressBar::complete(std::string str){
    if (this->hasCleared) return;
    this->hasCleared = true;


    const auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = endTime - this->startTime;

    if (!silent){
        // Clear line;
        std::cout<<"\r\e[K";

        std::cout<<"Complete: "<<int(duration.count())<<"ms";
        if (str != ""){
            std::cout<<" "<<str;
        }
        std::cout<<std::endl;
    }

    // Save results to log
    std::string key = (this->isParallel?"[P]":"[S]_")+this->label;
    const auto & found = timeLogger.find(key);
    if (found == timeLogger.end()){
        timeLogger.insert(TimeLogEntry(key, duration));
    }else{
        (*found).second += duration;
    }
}

ProgressBar::~ProgressBar(){
    this->complete();
};


void ProgressBar::printProgress(double progress){
    auto completed = int(progress*BAR_WIDTH);

    if (progress < 0.9999999 && completed == int(this->completedLastPrinted)){
        return;
    }

    this->completedLastPrinted = completed;

    std::string str = "\r[";
    str += std::string(completed,'=');
    str += ">" + std::string(BAR_WIDTH-completed,' ');
    str += "]";
    str += std::to_string(int(progress*100))+"%";
    exclusiveRun([str](){
        std::cout<<str<<std::flush;
    });
}

void ProgressBar::update(size_t workDone){
    this->workDone = workDone;
    if (silent) return;
    printProgress((double)this->workDone/this->workToDo);
}

void ProgressBar::increment(size_t amount){
    this->workDone+=amount;
    if (silent) return;
    printProgress((double)this->workDone/this->workToDo);
}