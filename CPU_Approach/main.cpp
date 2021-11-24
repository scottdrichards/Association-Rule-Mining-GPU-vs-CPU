#include <iostream>
#include "./database.h"
#define NUM_TRANSACTIONS 10000
#define MAX_TRANSACTION_SIZE 10
static const char letters[]="abcdefghijklmnopqrstuvwxyz";

int main(int argc, char const *argv[])
{
    auto db = Database();
    for (auto i = 0; i<NUM_TRANSACTIONS; i++){
        ItemSet transaction;
        for (auto j = 0; j<MAX_TRANSACTION_SIZE; j++){
            std::string c = ""+ letters[std::rand()%26];
            if (transaction.find(c)==transaction.end()){
                break;
            }
            transaction.insert(c);
        }
        db.add(transaction);
    }

    std::cout<<"Added "<<db.size()<<" transactions"<<std::endl;
    return 0;
}
