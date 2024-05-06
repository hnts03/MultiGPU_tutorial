/*
   This is tutorial file of Multi-GPU and Stream feature.
   Author: Geonwoo Choi
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

// intput A and B array, output C array
__global__ void GPU_kernel(int* matA, int* matB, int* matC,
												int matA_H, int matA_W, int matB_W){
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	for (int i = tid; i < matA_W; i += blockDim.x){
		for(int j = 0; j < matB_W; j++){
			// matC[bid*matA_W + j] += matA[bid*matA_W + i] * matB[j * matB_W + bid];
			// matC[bid*matA_W + j] = atomicAdd(&(matC[bid*matA_W+j]), matA[bid*matA_W + i] * matB[j * matB_W + bid]);
			atomicAdd(&(matC[bid*matA_W+j]), matA[bid*matA_W + i] * matB[j * matB_W + bid]);
		}
		
	}
}
