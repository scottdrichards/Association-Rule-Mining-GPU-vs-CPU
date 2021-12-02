#include "progressBar.h"

#include <iostream>
#include <stdint.h> 

#define BAR_WIDTH 70

void progressBar(double progress, bool carriageReturn){
    auto completed = int(progress*BAR_WIDTH);
    if (carriageReturn) std::cout<<'\r';
    std::cout<<"[";
    std::cout<<std::string(completed,'=');
    std::cout<<">";
    std::cout<<std::string(BAR_WIDTH-completed,' ');
    std::cout<<"]";
    printf("%*d",3,int(progress*100));
    std::cout<<'%';
    std::cout.flush();
}

void progressBar(bool complete){
    progressBar(1.0);
    std::cout<<std::endl;
}
