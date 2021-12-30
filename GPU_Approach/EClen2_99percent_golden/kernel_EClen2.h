#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

__device__ int  compare(char *Low, char *High);
__device__ void swap(char *Low, char *High, char *temp, int size);
__device__ int  sortIndex(char element);

__global__ void cudaEClen2Part1(cudaPitchedPtr db_gpuPitchPtr, dim3 db_gpuDim, cudaPitchedPtr db_tran_gpuPitchPtr, cudaPitchedPtr EC_len2_gpuPitchPtr, int EC_size);

