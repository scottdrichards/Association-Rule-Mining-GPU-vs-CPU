#include "kernel_EChigherlen.h"

__global__ void cudaEC_higherlen(dim3 db_gpuDim, cudaPitchedPtr db_tran_gpuPitchPtr, cudaPitchedPtr EC_set1_gpuPitchPtr, cudaPitchedPtr EC_set2_gpuPitchPtr, cudaPitchedPtr op_element_gpuPitchPtr, cudaPitchedPtr op_sc_gpuPitchPtr, int EC_size, int iteration)  {
    int i,j,k,l,m,n,p,r;

    int num_of_transactions = db_gpuDim.y;
    int num_of_itemsets = db_gpuDim.z;

    char *EC_set1_ptr = (char *)EC_set1_gpuPitchPtr.ptr;
    size_t EC_set1_pitch = EC_set1_gpuPitchPtr.pitch;
    size_t EC_set1_slicePitch = EC_set1_pitch * num_of_transactions * EC_size;
    char *EC_set1_slice1;
    char *EC_set1_slice2;
    char *EC_set1_element1;
    char *EC_set1_element2;

    char *EC_set2_ptr = (char *)EC_set2_gpuPitchPtr.ptr;
    size_t EC_set2_pitch = EC_set2_gpuPitchPtr.pitch;
    size_t EC_set2_slicePitch = EC_set2_pitch * num_of_transactions * EC_size;
    char *EC_set2_slice;
    char *EC_set2_element;

    char *op_ptr = (char *)op_element_gpuPitchPtr.ptr;
    size_t op_pitch = op_element_gpuPitchPtr.pitch;
    size_t op_slicePitch = op_pitch * num_of_transactions * EC_size * num_of_itemsets;
    char *op_slice;
    char *op_element;
    char *op_element_buffer;

    float *op_sc_ptr = (float *)op_sc_gpuPitchPtr.ptr;
    size_t op_sc_pitch = op_sc_gpuPitchPtr.pitch;
    size_t op_sc_slicePitch = op_sc_pitch * num_of_transactions * EC_size * num_of_itemsets;
    float *op_sc_slice;
    float *op_sc_element;

    int countIter;
    float countIntersection,count;
    float support,conf;

    char *bufferLow, *bufferHigh;

    l = threadIdx.x;
    n = 0;
    p = l * (num_of_transactions * EC_size * num_of_itemsets / 27);

    if (l < 27)  {
	for (i = 0; i < num_of_transactions * EC_size; i++)  {
	    EC_set1_slice1 = EC_set1_ptr + ((l*(num_of_transactions+num_of_itemsets)) + 0) * EC_set1_slicePitch;
	    EC_set1_element1 = (char *)(EC_set1_slice1 + i * EC_set1_pitch);
	    if (EC_set1_element1[0] == 0)  continue;

	    for (j = i+1; j < num_of_transactions * EC_size; j++)  {
		EC_set1_slice2 = EC_set1_ptr + ((l*(num_of_transactions+num_of_itemsets)) + 0) * EC_set1_slicePitch;
            	EC_set1_element2 = (char *)(EC_set1_slice2 + j * EC_set1_pitch);

            	if (EC_set1_element2[0] == 0)  continue;
		if (compare1(EC_set1_element1,EC_set1_element2) != 0) continue;

		countIter = 0;
		for (k = 1; k < iteration-1; k++)  {
		    EC_set1_slice1 = EC_set1_ptr + ((l*(num_of_transactions+num_of_itemsets)) + k) * EC_set1_slicePitch;
		    EC_set1_element1 = (char *)(EC_set1_slice1 + i * EC_set1_pitch);

		    EC_set1_slice2 = EC_set1_ptr + ((l*(num_of_transactions+num_of_itemsets)) + k) * EC_set1_slicePitch;
                    EC_set1_element2 = (char *)(EC_set1_slice2 + j * EC_set1_pitch);

		    if (compare1(EC_set1_element1,"") == 0 || compare1(EC_set1_element2,"") == 0)  break;
		    if (compare1(EC_set1_element1,EC_set1_element2) == 0)  countIter++;
		    else break;
		}

		if(countIter == (iteration - 2))  {
		    EC_set1_slice1 = EC_set1_ptr + ((l*(num_of_transactions+num_of_itemsets)) + iteration-1) * EC_set1_slicePitch;
		    EC_set1_element1 = (char *)(EC_set1_slice1 + i * EC_set1_pitch);

		    EC_set1_slice2 = EC_set1_ptr + ((l*(num_of_transactions+num_of_itemsets)) + iteration-1) * EC_set1_slicePitch;
		    EC_set1_element2 = (char *)(EC_set1_slice2 + j * EC_set1_pitch);

		    if(EC_set1_element1[0] != 0 && EC_set1_element2[0] != 0)  {

                        countIntersection = 0.0000;
                        count = 0.0000;

                        for(r = 0; r < num_of_transactions; r++)  {
                            EC_set1_slice1 = EC_set1_ptr + ((l*(num_of_transactions+num_of_itemsets)) + num_of_itemsets+r) * EC_set1_slicePitch;
                            EC_set1_element1 = (char *)(EC_set1_slice1 + i * EC_set1_pitch);

                            EC_set1_slice2 = EC_set1_ptr + ((l*(num_of_transactions+num_of_itemsets)) + num_of_itemsets+r) * EC_set1_slicePitch;
                            EC_set1_element2 = (char *)(EC_set1_slice2 + j * EC_set1_pitch);

                            if (compare1(EC_set1_element1,"p") == 0 && compare1(EC_set1_element2,"p") == 0)  {
                                countIntersection++;
                                EC_set2_slice = EC_set2_ptr + ((l*(num_of_transactions+num_of_itemsets)) + num_of_itemsets+r) * EC_set2_slicePitch;
                                EC_set2_element = (char *)(EC_set2_slice + n * EC_set2_pitch);
                                memcpy(EC_set2_element,"p",db_gpuDim.x);
                            }
                        }

			if (countIntersection > 0)  {
                    	    EC_set1_slice1 = EC_set1_ptr + ((l*(num_of_transactions+num_of_itemsets)) + iteration-1) * EC_set1_slicePitch;
                    	    EC_set1_element1 = (char *)(EC_set1_slice1 + i * EC_set1_pitch);

                    	    EC_set1_slice2 = EC_set1_ptr + ((l*(num_of_transactions+num_of_itemsets)) + iteration-1) * EC_set1_slicePitch;
                    	    EC_set1_element2 = (char *)(EC_set1_slice2 + j * EC_set1_pitch);

		            m = compare1(EC_set1_element1,EC_set1_element2);
		            if (m == 1) {
                                bufferLow  = EC_set1_element2;
                                bufferHigh = EC_set1_element1;
                            }
                            else  {
                                bufferLow  = EC_set1_element1;
                                bufferHigh = EC_set1_element2;
                            }

			    op_slice = op_ptr + (p + (n * (iteration + 1))) * op_pitch;
		            for (k = 0; k < iteration-1; k++)  {
			        EC_set1_slice1 = EC_set1_ptr + ((l*(num_of_transactions+num_of_itemsets)) + k) * EC_set1_slicePitch;
                                EC_set1_element1 = (char *)(EC_set1_slice1 + i * EC_set1_pitch);

			        EC_set2_slice = EC_set2_ptr + ((l*(num_of_transactions+num_of_itemsets)) + k) * EC_set2_slicePitch;
			        EC_set2_element = (char *)(EC_set2_slice + n * EC_set2_pitch);

			        op_element = (char *)(op_slice + k * op_slicePitch);
			        memcpy(op_element,EC_set1_element1,db_gpuDim.x);

			        memcpy(EC_set2_element,EC_set1_element1,db_gpuDim.x);
		            }

		            EC_set2_slice = EC_set2_ptr + ((l*(num_of_transactions+num_of_itemsets)) + iteration-1) * EC_set2_slicePitch;
		            EC_set2_element = (char *)(EC_set2_slice + n * EC_set2_pitch);
		            memcpy(EC_set2_element,bufferLow,db_gpuDim.x);

		            EC_set2_slice = EC_set2_ptr + ((l*(num_of_transactions+num_of_itemsets)) + iteration) * EC_set2_slicePitch;
 		            EC_set2_element = (char *)(EC_set2_slice + n * EC_set2_pitch);
		            memcpy(EC_set2_element,bufferHigh,db_gpuDim.x);

                            op_element = (char *)(op_slice + (iteration-1) * op_slicePitch);
                            memcpy(op_element,bufferLow,db_gpuDim.x);

			    op_element = (char *)(op_slice + iteration * op_slicePitch);
			    memcpy(op_element,bufferHigh,db_gpuDim.x);

			    support = (countIntersection * 100) / num_of_transactions;
			    op_slice = op_ptr + (p + (n * (iteration + 1))) * op_pitch;
			    op_element = (char *)(op_slice);

			    count = countTransactions (db_gpuDim, db_tran_gpuPitchPtr, op_element);
			    conf = (countIntersection * 100) / count;

                            op_sc_slice = op_sc_ptr + (p + (n * (iteration + 1))) * op_sc_pitch;
                            op_sc_element = (float *) op_sc_slice;
                            memcpy(&op_sc_element[0],&support,sizeof(float));

                            op_sc_element = (float *)(op_sc_slice + op_sc_slicePitch);
                            memcpy(&op_sc_element[0],&conf,sizeof(float));

			    for(r = 1; r <= iteration; r++)  {
			        op_slice = op_ptr + (p + (n * (iteration + 1) + r)) * op_pitch;
			        op_element = (char *)(op_slice);
			        op_element_buffer = (char *)(op_ptr + (p + (n * (iteration + 1))) * op_pitch + r * op_slicePitch);
			        memcpy(op_element,op_element_buffer,db_gpuDim.x);

			        count = countTransactions (db_gpuDim, db_tran_gpuPitchPtr, op_element_buffer);
			        conf = (countIntersection * 100) / count;

                	        op_sc_slice = op_sc_ptr + (p + (n * (iteration + 1) + r)) * op_sc_pitch;
                	        op_sc_element = (float *) op_sc_slice;
                	        memcpy(&op_sc_element[0],&support,sizeof(float));

                	        op_sc_element = (float *)(op_sc_slice + op_sc_slicePitch);
                	        memcpy(&op_sc_element[0],&conf,sizeof(float));

			        for(k = 1; k <= iteration; k++)  {
				    op_element = (char *)(op_slice + k * op_slicePitch);
				    if(r >= k)  op_element_buffer = (char *)(op_ptr + (p + (n*(iteration + 1))) * op_pitch + (k-1) * op_slicePitch);
				    else op_element_buffer = (char *)(op_ptr + (p + (n*(iteration + 1))) * op_pitch + k * op_slicePitch);
                                    memcpy(op_element,op_element_buffer,db_gpuDim.x);
			        }
			    }
			    n++;
			}
		    }
		}
	    }
	}
    }
}

__device__ int compare1(char *Low, char *High)  {
    int i = 0;
    while(Low[i] != 0 || High[i] != 0)  {
        if (Low[i] > High[i])       return 1;
        else if (Low[i] < High[i])  return 2;
        else  i++;
    }
    if (Low[i] != 0 && High[i] == 0)       return 1;    // Elements may have different lengths
    else if (Low[i] == 0 && High[i] != 0)  return 2;    // Elements may have different lengths
    else if (Low[i] == 0 && High[i] == 0)  return 0;    // Elements have same length and a match is found

    return 0;
}

__device__ int countTransactions(dim3 db_gpuDim, cudaPitchedPtr db_tran_gpuPitchPtr, char* op_element_buffer)  {
    int i,r;
    int count = 0;

    int num_of_transactions = db_gpuDim.y;
    int num_of_itemsets = db_gpuDim.z;

    char *db_tran_ptr = (char *)db_tran_gpuPitchPtr.ptr;
    size_t db_tran_pitch = db_tran_gpuPitchPtr.pitch;
    size_t db_tran_slicePitch = db_tran_pitch * num_of_itemsets * num_of_transactions;
    char *db_tran_slice;
    char *db_tran_element;

    for (i = 0; i < num_of_transactions*num_of_itemsets; i++)  {
        db_tran_element = (char *)(db_tran_ptr + db_tran_pitch * i);
        r = i;
        if (compare1(db_tran_element,op_element_buffer) == 0)  break;
    }

    for (i = 1; i < num_of_transactions+1; i++)  {
        db_tran_slice = db_tran_ptr + i * db_tran_slicePitch;
        db_tran_element = (char *)(db_tran_slice + r * db_tran_pitch);

        if (compare1(db_tran_element,"p") == 0)  count++;
    }

    return count;
}

