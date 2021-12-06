#include "bitsetUtils.h"

std::vector<ItemID> itemSetToIDs(ItemSet itemSet){
    std::vector<ItemID> itemIDs(itemSet.count());
    std::size_t bitIndex = itemSet._Find_first();
    size_t i = 0;
    while (bitIndex != itemSet.size()){
        itemIDs[i++] = bitIndex;
        itemSet.set(bitIndex, false);
        bitIndex = itemSet._Find_first();
    }
    return itemIDs;
}
