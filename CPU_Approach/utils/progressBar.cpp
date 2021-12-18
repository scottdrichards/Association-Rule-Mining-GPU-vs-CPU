#include "progressBar.h"

#include <chrono>
#include <iostream>
#include <string>
#include <stdint.h> 
#include "./exclusiveRun.h"

#define BAR_WIDTH 70


ProgressBar::ProgressBar(size_t totalWork){
    this->workDone = 0;
    this->workToDo = totalWork;
    this->completedLastPrinted = 0;
    this->hasCleared = false;
    this->startTime = std::chrono::high_resolution_clock::now();
}

void ProgressBar::complete(std::string str){
    if (this->hasCleared) return;
    this->hasCleared = true;


    const auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = endTime - this->startTime;

    // Clear line;
    std::cout<<"\r\e[K";

    std::cout<<"Complete: "<<int(duration.count())<<"ms";
    if (str != ""){
        std::cout<<" "<<str;
    }
    std::cout<<std::endl;
}

ProgressBar::~ProgressBar(){
    this->complete();
};


void ProgressBar::printProgress(double progress){
    auto completed = int(progress*BAR_WIDTH);

    if (progress < 0.9999999 && completed == this->completedLastPrinted){
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
    printProgress((double)this->workDone/this->workToDo);
}

void ProgressBar::increment(size_t amount){
    this->workDone+=amount;
    printProgress((double)this->workDone/this->workToDo);
}