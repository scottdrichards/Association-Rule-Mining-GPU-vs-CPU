#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__

void cudaTranspose(int num_of_transactions, int num_of_itemsets, char* db_cu[], char* db_tran_cu[])  {
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int column = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ((column < num_of_transactions) && (row < num_of_itemsets))  {
	db_tran_cu[row*num_of_transactions + column] =  "Hello";   // Checking this memory
	printf("row: %d  column: %d\n %s \n",row, column, db_tran_cu[row*num_of_transactions + column]);

        db_tran_cu[row*num_of_transactions + column] =  db_cu[column*num_of_itemsets + row];
        printf("row: %d  column: %d\n %s %s \n",row, column, db_tran_cu[row*num_of_transactions + column], db_cu[column*num_of_itemsets + row]);
    }
}
