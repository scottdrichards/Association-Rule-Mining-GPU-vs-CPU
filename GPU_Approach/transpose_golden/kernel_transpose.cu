#include "kernel.h"

__global__ void cudaTranspose(cudaPitchedPtr db_gpuPitchPtr, dim3 db_gpuDim, cudaPitchedPtr db_tran_gpuPitchPtr)  {
    int i,j,k;
    bool matched = false;

    int num_of_transactions = db_gpuDim.y;
    int num_of_itemsets = db_gpuDim.z;

    char *db_ptr = (char *)db_gpuPitchPtr.ptr;
    size_t db_pitch = db_gpuPitchPtr.pitch;
    size_t db_slicePitch = db_pitch * num_of_transactions;
    char *db_slice;
    char *db_current_slice;
    char *db_element;
    char *db_current_element;

    char *db_tran_ptr = (char *)db_tran_gpuPitchPtr.ptr;
    size_t db_tran_pitch = db_tran_gpuPitchPtr.pitch;
    size_t db_tran_slicePitch = db_pitch * num_of_itemsets * num_of_transactions;
    char *db_tran_current_element;

    int row    = (blockIdx.x  * gridDim.x ) + blockIdx.y;
    int column = (threadIdx.x * blockDim.x) + threadIdx.y;
//    printf("blockIdx.x: %d, blockIdx.y: %d, blockDim.x: %d, blockDim.y: %d, threadIdx.x: %d, threadIdx.y: %d\n",blockIdx.x,blockIdx.y,blockDim.x,blockDim.y,threadIdx.x,threadIdx.y);
//    printf("row: %d  column: %d\n",row, column);

    if ((row < num_of_transactions) && (column < num_of_itemsets))  {

        db_current_slice = db_ptr + column * db_slicePitch;
        db_current_element = (char *)(db_current_slice + row * db_pitch);
//        printf("row: %d  column: %d  db: %s\n",row, column, db_current_element);

	if (db_current_element[0] != 0)  {
            db_tran_current_element = (char *)(db_tran_ptr + (db_tran_pitch * ((row * num_of_itemsets) + column)));

	    for (i = 0; i < num_of_transactions; i++)  {
		if (i >= row)  break;

	        for (j = 0; j < num_of_itemsets; j++)  {
	            db_slice = db_ptr + j * db_slicePitch;
		    db_element = (char *)(db_slice + i * db_pitch);
//	              printf("row: %d  column: %d  db: %s\n",i, j, db_element);

		    if (db_element[0] == 0)  break;

		    k = 0;
		    matched = true;
		    while (db_element[k] != 0  ||  db_current_element[k] != 0)  {
			if (db_element[k] != db_current_element[k])  {
			    matched = false;
			    break;
			}
			k++;
		    }
		    if (matched)  {
			db_tran_current_element = (char *)(db_tran_ptr + (db_tran_pitch * ((i * num_of_itemsets) + j)));
			break;
		    }
		}
		if (matched)  break;
	    }
	    if (!matched)  {
		memcpy(db_tran_current_element,db_current_element,db_gpuDim.x);
		db_tran_current_element = (char *)(db_tran_current_element + ((row + 1) * db_tran_slicePitch));
		memcpy(db_tran_current_element,"p",db_gpuDim.x);
//	          printf("row: %d  column: %d  db: %s\n",row, column, db_current_element);
	    }
	    else  {
		db_tran_current_element = (char *)(db_tran_current_element + ((row + 1) * db_tran_slicePitch));
		memcpy(db_tran_current_element,"r",db_gpuDim.x);
//                printf("row: %d  column: %d  db: %s\n",row, column, db_current_element);
	    }
	}
    }
}

