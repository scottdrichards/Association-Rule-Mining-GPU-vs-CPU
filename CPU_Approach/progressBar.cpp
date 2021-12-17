#include "progressBar.h"

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

ProgressBar::~ProgressBar(){
    std::cout<<std::endl;
};


void ProgressBar::update(size_t amount){
    this->workDone+=amount;
    printProgress((double)this->workDone/this->workToDo);
}