#include "log.h"

#include <vector>
#include <algorithm>
#include <fstream>
#include <sys/stat.h> // Check if file exists

TimeLog timeLogger;
PsuedoLog psuedoLogger;

const std::string FILE_PATH = "results.csv";

void printLog(){
    std::vector<TimeLogEntry> logEntries;
    logEntries.assign(timeLogger.begin(), timeLogger.end());
    std::sort(logEntries.begin(), logEntries.end(), [](const TimeLogEntry & a, const TimeLogEntry & b){
        return a.first<b.first;
    });

    // Create file if it doesn't exist and add headers
    struct stat buf;
    const auto fileExists = stat(FILE_PATH.c_str(), &buf) != -1;
    if (!fileExists){
        std::ofstream file(FILE_PATH);
        for (const auto entry:psuedoLogger){
            file << entry.first;
            file << ", ";
        }
        for (const auto entry:logEntries){
            file << entry.first;
            if (entry != *logEntries.rbegin()) file << ", ";
        }
        file<<std::endl;
    }

    std::ofstream file;
    const auto accessMode = std::ios_base::app;
    file.open(FILE_PATH, accessMode);
    
    for (const auto entry:psuedoLogger){
        file << entry.second;
        file << ", ";
    }
    for (const auto entry:logEntries){
        file << entry.second.count();
        if (entry != *logEntries.rbegin()) file << ", ";
    }
    file << std::endl;
    file.close();
}
