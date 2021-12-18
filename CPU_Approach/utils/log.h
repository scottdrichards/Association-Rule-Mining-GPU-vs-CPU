#pragma once
#include <map>
#include <string>
#include <chrono>

typedef std::pair<std::string, std::chrono::duration<double, std::milli>> TimeLogEntry;
typedef std::map<std::string, std::chrono::duration<double, std::milli>> TimeLog;

extern TimeLog timeLogger;


typedef std::pair<std::string, std::string> PsuedoLogEntry;
typedef std::map<std::string, std::string> PsuedoLog;

extern PsuedoLog psuedoLogger;

void printLog();