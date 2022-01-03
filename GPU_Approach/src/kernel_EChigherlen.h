#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

__device__ int compare1(char *Low, char *High);
__device__ int countTransactions(dim3 db_gpuDim, cudaPitchedPtr db_tran_gpuPitchPtr, char* op_element_buffer);

__global__ void cudaEC_higherlen(dim3 db_gpuDim, cudaPitchedPtr db_tran_gpuPitchPtr, cudaPitchedPtr EC_set1_gpuPitchPtr, cudaPitchedPtr EC_set2_gpuPitchPtr, cudaPitchedPtr op_element_gpuPitchPtr, cudaPitchedPtr op_sc_gpuPitchPtr, int EC_size, int iteration);

