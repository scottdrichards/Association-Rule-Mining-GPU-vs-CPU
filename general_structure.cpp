#include <string>
#include <set>

typedef std::string Item;
typedef std::set<Item> Transaction;

class Database{
    public:
        std::set<Transaction> db;
        auto add(Transaction t){
            this.db.add(t);
            return this;
        }
        auto support(Item x){
            auto count = 0;
            for (auto transaction:this.db){
                if (transaction.contains(x)) count++;
            }
            return (double) count/this.db.size();
        }
        auto support(Item x, Item y){
            auto count = 0;
            for (auto transaction:this.db){
                if (transaction.contains(x) && transaction.contains(y)) count++;
            }
            return (double) count/this.db.size();
        }
        auto confidence(Item x, Item y){
            auto countXY = 0;
            auto countX = 0;
            for (auto transaction:this.db){
                if (transaction.contains(x)){
                    countX++;
                    if (transaction.contains(y)){
                        countXY++;
                    }
                }
            }
            return (double) countX/countXY;
        }
}

void decomposition(){}
void assignment(){}
void orchestration(){

}