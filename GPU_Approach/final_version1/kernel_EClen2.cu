#include "kernel_EClen2.h"

__global__ void cudaEC_len2(cudaPitchedPtr db_gpuPitchPtr, dim3 db_gpuDim, cudaPitchedPtr db_tran_gpuPitchPtr, cudaPitchedPtr EC_len2_gpuPitchPtr, cudaPitchedPtr output_element_len2_gpuPitchPtr, cudaPitchedPtr output_support_confidence_len2_gpuPitchPtr, int EC_size)  {

/**********************  SOME INITIALIZATION STUFFS  ********************************************/
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

    char *op_len2_ptr = (char *)output_element_len2_gpuPitchPtr.ptr;
    size_t op_len2_pitch = output_element_len2_gpuPitchPtr.pitch;
    size_t op_len2_slicePitch = op_len2_pitch * num_of_transactions * EC_size * 2;
    char *op_len2_slice;
    char *op_len2_element;

    float *op_sc_len2_ptr = (float *)output_support_confidence_len2_gpuPitchPtr.ptr;
    size_t op_sc_len2_pitch = output_support_confidence_len2_gpuPitchPtr.pitch;
    size_t op_sc_len2_slicePitch = op_sc_len2_pitch * num_of_transactions * EC_size * 2;
    float *op_sc_len2_slice;
    float *op_sc_len2_element;

    float countIntersection, countLow, countHigh;
    float support, conf1, conf2;

    char *bufferHigh, *bufferLow;
/*************************  END  ***************************************/
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
	    for (j = 1; j <= column; j++)  k = k + (num_of_itemsets - j);
		/** k = 0 for j = column = 0
		    k is the fine tuning of the EC_len2 element location, row * EC_size is the coarse tuning of the same.
		    (row * EC_size) + k is the actual position of the EC_len2 element  */

            for (r = 0; r < num_of_transactions*num_of_itemsets; r++)  {
                db_tran_current_element = (char *)(db_tran_ptr + db_tran_pitch * r);
                c1 = r;  // save the row number of the current db transpose element; remember, row in db transpose is the column in db
                if (compare(db_tran_current_element,db_current_element) == 0)  break;
            }

	    EC_len2_slice = EC_len2_ptr + ((row * EC_size) + k) * EC_len2_pitch;

	    for (j = column+1; j < num_of_itemsets; j++)  {
		// Form EC length=2 elements by combining the current element and the elements after this current element within the same transaction

		db_slice = db_ptr + j * db_slicePitch;
		db_element = (char *)(db_slice + row * db_pitch);

		if (db_element[0] == 0)  {
		    if (column == 0 && j == 1)  {  // Special case where there is only 1 item in the transaction!
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
		    c2 = r;  // save the row number of the combining db transpose element; remember, row in db transpose is the column in db
                    if (compare(db_tran_element,db_element) == 0)  break;
                }

		i = compare(db_current_element,db_element);  // Compare the 2 elements to combine them in alphabetical order
		if (i == 1) {
		    bufferLow  = db_element;
		    bufferHigh = db_current_element;
		}
		else  {
		    bufferLow  = db_current_element;
		    bufferHigh = db_element;
		}

		l = sortIndex(bufferLow[0]);
		EC_len2_element = (char *)(EC_len2_slice + ((l*(num_of_transactions+2)) + 0) * EC_len2_slicePitch);
                op_len2_slice   =  op_len2_ptr + ((row * EC_size * 2) + (2*k)) * op_len2_pitch;
                op_len2_element = (char *) op_len2_slice;
		memcpy(EC_len2_element,bufferLow,db_gpuDim.x);
		memcpy(op_len2_element,bufferLow,db_gpuDim.x);

		EC_len2_element = (char *)(EC_len2_slice + ((l*(num_of_transactions+2)) + 1) * EC_len2_slicePitch);
		op_len2_element = (char *)(op_len2_slice + op_len2_slicePitch);
		memcpy(EC_len2_element,bufferHigh,db_gpuDim.x);
		memcpy(op_len2_element,bufferHigh,db_gpuDim.x);

                op_len2_slice   =  op_len2_ptr + ((row * EC_size * 2) + (2*k) + 1) * op_len2_pitch;
		op_len2_element = (char *) op_len2_slice;
		memcpy(op_len2_element,bufferHigh,db_gpuDim.x);

		op_len2_element = (char *)(op_len2_slice + op_len2_slicePitch);
		memcpy(op_len2_element,bufferLow,db_gpuDim.x);

//		  printf("row: %d  column: %d  EC_len2: %s%s\n",row, column, db_current_element,db_element);
                countIntersection = 0.0000;
		countLow = 0.0000;
		countHigh = 0.0000;

		// To identify which one is the lower element
		db_tran_current_element = (char *)(db_tran_ptr + c1 * db_tran_pitch);
		if (compare(db_tran_current_element,bufferLow) == 0) i = 1;
		else i = 2;

		for (r = 0; r < num_of_transactions+1; r++)  {
                    db_tran_current_slice = db_tran_ptr + r * db_tran_slicePitch;
                    db_tran_current_element = (char *)(db_tran_current_slice + c1 * db_tran_pitch);
                    db_tran_element = (char *)(db_tran_current_slice + c2 * db_tran_pitch);

//                      printf("db_tran_current_element %s, db_tran_element %s\n",db_tran_current_element,db_tran_element);

		    if (compare(db_tran_current_element,"p") == 0)  {
			if (i == 1) countLow++;
			else countHigh++;
		    }
		    if (compare(db_tran_element,"p") == 0)  {
		        if (i == 1) countHigh++;
			else countLow++;
		    }
		    if (compare(db_tran_current_element,"p") == 0 && compare(db_tran_element,"p") == 0)  {
			    countIntersection++;
			    EC_len2_element = (char *)(EC_len2_slice + ((l*(num_of_transactions+2)) + r+1) * EC_len2_slicePitch);
			    memcpy(EC_len2_element,"p",db_gpuDim.x);
		    }
                }

		support = (countIntersection * 100) / num_of_transactions;
                op_sc_len2_slice = op_sc_len2_ptr + ((row * EC_size * 2) + (2*k)) * op_sc_len2_pitch;
                op_sc_len2_element = (float *) op_sc_len2_slice;
		memcpy(&op_sc_len2_element[0],&support,sizeof(float));

		conf1 = (countIntersection * 100) / countLow;
		op_sc_len2_element = (float *)(op_sc_len2_slice + op_sc_len2_slicePitch);
		memcpy(&op_sc_len2_element[0],&conf1,sizeof(float));

                op_sc_len2_slice = op_sc_len2_ptr + ((row * EC_size * 2) + (2*k) + 1) * op_sc_len2_pitch;
                op_sc_len2_element = (float *) op_sc_len2_slice;
		memcpy(&op_sc_len2_element[0],&support,sizeof(float));

                conf2 = (countIntersection * 100) / countHigh;
		op_sc_len2_element = (float *)(op_sc_len2_slice + op_sc_len2_slicePitch);
		memcpy(&op_sc_len2_element[0],&conf2,sizeof(float));

		k++;
	    }
	}
/**
		if (row == 1 && column == 5)  {
		    for (i = 0; i < num_of_transactions*EC_size*2; i++)  {
        		for (j = 0; j < 2; j++)  {
            		    op_sc_len2_slice = op_sc_len2_ptr + i * op_sc_len2_pitch;
            		    op_sc_len2_element = (float *)(op_sc_len2_slice + j * op_sc_len2_slicePitch);
			    printf("i: %d, j: %d, Result: %f \n",i,j,op_sc_len2_element[0]);
			}
		    }
		}  */
    }
}

__device__ int compare(char *Low, char *High)  {
    int i = 0;
    while(Low[i] != 0 || High[i] != 0)  {
        if (Low[i] > High[i])  	    return 1;
        else if (Low[i] < High[i])  return 2;
        else  i++;
    }
    if (Low[i] != 0 && High[i] == 0)  	   return 1;	// Elements may have different lengths
    else if (Low[i] == 0 && High[i] != 0)  return 2;	// Elements may have different lengths
    else if (Low[i] == 0 && High[i] == 0)  return 0;	// Elements have same length and a match is found

    return 0;
}

__device__ int sortIndex(char element)  {
    int i;
    if (element >= 65 && element <= 90)  	i = element - 65;
    else if (element >= 97 && element <= 122)   i = element - 97;
    else i = 26;

    return i;
}

