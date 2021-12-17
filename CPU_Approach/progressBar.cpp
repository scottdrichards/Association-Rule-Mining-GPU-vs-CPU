#include "progressBar.h"

#include <chrono>
#include <iostream>
#include <string>
#include <stdint.h> 
#include "./exclusiveRun.h"

#define BAR_WIDTH 70

void printProgress(double progress){
    auto completed = int(progress*BAR_WIDTH);
    std::string str = "\r[";
    str += std::string(completed,'=');
    str += ">" + std::string(BAR_WIDTH-completed,' ');
    str += "]";
    str += std::to_string(int(progress*100))+"%";
    exclusiveRun([str](){
        std::cout<<str<<std::flush;
    });
}

ProgressBar::ProgressBar(size_t totalWork){
    this->workDone = 0;
    this->workToDo = totalWork;
    this->startTime = std::chrono::high_resolution_clock::now();
}

ProgressBar::~ProgressBar(){
    const auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = endTime - this->startTime;
    std::cout<<"\r\e[KComplete: "<<int(duration.count())<<"ms\n"<<std::endl;
};


void ProgressBar::update(size_t amount){
    this->workDone+=amount;
    printProgress((double)this->workDone/this->workToDo);
}