#include "progressBar.h"

#include <iostream>
#include <stdint.h> 

#define BAR_WIDTH 70

void printProgress(double progress){
    auto completed = int(progress*BAR_WIDTH);
    std::cout<<'\r';
    std::cout<<"[";
    std::cout<<std::string(completed,'=');
    std::cout<<">";
    std::cout<<std::string(BAR_WIDTH-completed,' ');
    std::cout<<"]";
    printf("%*d",3,int(progress*100));
    std::cout<<'%';
    std::cout.flush();
}

void ProgressBar::update(size_t amount){
    this->workDone+=amount;
    printProgress((double)this->workDone/this->workToDo);
}