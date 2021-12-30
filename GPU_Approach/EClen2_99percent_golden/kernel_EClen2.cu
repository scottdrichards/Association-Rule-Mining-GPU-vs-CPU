#include "kernel_EClen2.h"

__global__ void cudaEClen2Part1(cudaPitchedPtr db_gpuPitchPtr, dim3 db_gpuDim, cudaPitchedPtr db_tran_gpuPitchPtr, cudaPitchedPtr EC_len2_gpuPitchPtr, int EC_size)  {
    int i,j,k,l;
    int r,c1,c2;

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
    size_t db_tran_slicePitch = db_tran_pitch * num_of_itemsets * num_of_transactions;
    char *db_tran_current_slice;
    char *db_tran_current_element;
    char *db_tran_element;

    char *EC_len2_ptr = (char *)EC_len2_gpuPitchPtr.ptr;
    size_t EC_len2_pitch = EC_len2_gpuPitchPtr.pitch;
    size_t EC_len2_slicePitch = EC_len2_pitch * num_of_transactions * EC_size;
    char *EC_len2_slice;
    char *EC_len2_element;

    char *bufferHigh, *bufferLow;

    int row    = (blockIdx.x  * gridDim.x ) + blockIdx.y;
    int column = (threadIdx.x * blockDim.x) + threadIdx.y;
//    printf("blockIdx.x: %d, blockIdx.y: %d, blockDim.x: %d, blockDim.y: %d, threadIdx.x: %d, threadIdx.y: %d\n",blockIdx.x,blockIdx.y,blockDim.x,blockDim.y,threadIdx.x,threadIdx.y);
//    printf("row: %d  column: %d\n",row, column);

    if ((row < num_of_transactions) && (column < num_of_itemsets))  {

        db_current_slice = db_ptr + column * db_slicePitch;
        db_current_element = (char *)(db_current_slice + row * db_pitch);
//        printf("row: %d  column: %d  db: %s\n",row, column, db_current_element);

	if (db_current_element[0] != 0)  {
	    k = 0;
	    for (j = 1; j <= column; j++)  k = k + (num_of_itemsets - j);  // k = 0 for j = column = 0

            for (r = 0; r < num_of_transactions*num_of_itemsets; r++)  {
                db_tran_current_element = (char *)(db_tran_ptr + db_tran_pitch * r);
                c1 = r;
                if (compare(db_tran_current_element,db_current_element) == 0)  break;
            }

	    for (j = column+1; j < num_of_itemsets; j++)  {
		db_slice = db_ptr + j * db_slicePitch;
		db_element = (char *)(db_slice + row * db_pitch);

		if (db_element[0] == 0)  {
		    if (column == 0 && j == 1)  {
			EC_len2_slice = EC_len2_ptr + ((row * EC_size) + k) * EC_len2_pitch;
			l = sortIndex(db_current_element[0]);
			EC_len2_element = (char *)(EC_len2_slice + ((l*(num_of_transactions+2)) + 0) * EC_len2_slicePitch);
			memcpy(EC_len2_element,db_current_element,db_gpuDim.x);

                        for (r = 0; r < num_of_transactions+1; r++)  {
                            db_tran_current_slice = db_tran_ptr + r * db_tran_slicePitch;
                            db_tran_current_element = (char *)(db_tran_current_slice + c1 * db_tran_pitch);

//                            printf("db_tran_current_element %s\n",db_tran_current_element);

                            if (compare(db_tran_current_element,"p") == 0)  {
                                EC_len2_element = (char *)(EC_len2_slice + ((l*(num_of_transactions+2)) + r+1) * EC_len2_slicePitch);
                                memcpy(EC_len2_element,"p",db_gpuDim.x);
                            }
			}
                    }
		    break;
		}

                for (r = 0; r < num_of_transactions*num_of_itemsets; r++)  {
                    db_tran_element = (char *)(db_tran_ptr + db_tran_pitch * r);
		    c2 = r;
                    if (compare(db_tran_element,db_element) == 0)  break;
                }

		i = compare(db_current_element,db_element);
		if (i == 1) {
		    bufferLow  = db_element;
		    bufferHigh = db_current_element;
		}
		else  {
		    bufferLow  = db_current_element;
		    bufferHigh = db_element;
		}

		EC_len2_slice = EC_len2_ptr + ((row * EC_size) + k) * EC_len2_pitch;
		k++;

		l = sortIndex(bufferLow[0]);
		EC_len2_element = (char *)(EC_len2_slice + ((l*(num_of_transactions+2)) + 0) * EC_len2_slicePitch);
		memcpy(EC_len2_element,bufferLow,db_gpuDim.x);
		EC_len2_element = (char *)(EC_len2_slice + ((l*(num_of_transactions+2)) + 1) * EC_len2_slicePitch);
		memcpy(EC_len2_element,bufferHigh,db_gpuDim.x);

//		  printf("row: %d  column: %d  EC_len2: %s%s\n",row, column, db_current_element,db_element);
                
		for (r = 0; r < num_of_transactions+1; r++)  {
                    db_tran_current_slice = db_tran_ptr + r * db_tran_slicePitch;
                    db_tran_current_element = (char *)(db_tran_current_slice + c1 * db_tran_pitch);
                    db_tran_element = (char *)(db_tran_current_slice + c2 * db_tran_pitch);

//                      printf("db_tran_current_element %s, db_tran_element %s\n",db_tran_current_element,db_tran_element);

		    if (compare(db_tran_current_element,"p") == 0 && compare(db_tran_element,"p") == 0)  {
			EC_len2_element = (char *)(EC_len2_slice + ((l*(num_of_transactions+2)) + r+1) * EC_len2_slicePitch);
			memcpy(EC_len2_element,"p",db_gpuDim.x);
		    }
                }
	    }
	}
    }
}

__device__ int compare(char *Low, char *High)  {
    int i = 0;
    while(Low[i] != 0 || High[i] != 0)  {
        if (Low[i] > High[i])  return 1;
        else if (Low[i] < High[i]) return 2;
        else  i++;
    }
    if (Low[i] != 0 && High[i] == 0)  return 1;
    else if (Low[i] == 0 && High[i] != 0)  return 2;
    else if (Low[i] == 0 && High[i] == 0)  return 0;

    return 0;
}

__device__ void swap(char *Low, char *High, char *temp, int size)  {
    memcpy(temp,"",size);
    memcpy(temp,High,size);
    memcpy(High,Low,size);
    memcpy(Low,temp,size);
}

__device__ int sortIndex(char element)  {
    int i;
    if (element >= 65 && element <= 90)  i = element - 65;
    else if (element >= 97 && element <= 122)  i = element - 97;
    else i = 26;

    return i;
}

