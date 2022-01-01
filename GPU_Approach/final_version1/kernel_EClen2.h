#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

__device__ int  compare(char *Low, char *High);
__device__ int  sortIndex(char element);

__global__ void cudaEC_len2(cudaPitchedPtr db_gpuPitchPtr, dim3 db_gpuDim, cudaPitchedPtr db_tran_gpuPitchPtr, cudaPitchedPtr EC_len2_gpuPitchPtr, cudaPitchedPtr output_element_len2_gpuPitchPtr, cudaPitchedPtr output_support_confidence_len2_gpuPitchPtr, int EC_size);

