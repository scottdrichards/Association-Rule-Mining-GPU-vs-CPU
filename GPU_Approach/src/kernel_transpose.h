#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

__global__ void cudaTranspose(cudaPitchedPtr db_gpuPitchPtr, dim3 db_gpuDim, cudaPitchedPtr db_tran_gpuPitchPtr);

