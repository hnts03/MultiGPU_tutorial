/*
   This is aggregation header file for CUDA kernels.
*/
#include "gpuKernel.cuh"

#include <math.h>

void CPU_kernel(int* matA, int* matB, int* matC, int A_H, int A_W, int B_W){
	long long unsigned int input_matrix_size = A_H * A_W;
	long long unsigned int output_matrix_size = A_H * B_W;

	for(int i = 0; i < A_H; i++){			// Loop for Height of matrix A
		for(int j = 0; j < A_W; j++){		// Loop for Width of matrix A
			for(int k = 0; k < B_W; k++){	// Loop for Width of matrix B
				matC[i*A_W + k] += matA[i*A_W + j] * matB[j* B_W + k];
			}
		}
	}
}

void CPU_kernel(float* matA, float* matB, float* matC, int A_H, int A_W, int B_W){
	long long unsigned int input_matrix_size = A_H * A_W;
	long long unsigned int output_matrix_size = A_H * B_W;

	for(int i = 0; i < A_H; i++){			// Loop for Height of matrix A
		for(int j = 0; j < A_W; j++){		// Loop for Width of matrix A
			for(int k = 0; k < B_W; k++){	// Loop for Width of matrix B
				matC[i*A_W + k] += matA[i*A_W + j] * matB[j* B_W + k];
			}
		}
	}
}



// return 0 when wrong output detected
int is_valid(float* outCPU, float* outGPU, int output_matrix_size){
	for (int i = 0; i < output_matrix_size; i++){
		if (fabs(outCPU[i] - outGPU[i]) >= 0.00002) {
			printf("err on %d-th element\ncpu[%d] = %f, gpu[%d] = %f\n", i, i, outCPU[i], i, outGPU[i]);	
			return 0;
		}
	}
	return 1;

}

int is_valid(int* outCPU, int* outGPU, int output_matrix_size){
	for (int i = 0; i < output_matrix_size; i++){
		if(outCPU[i] != outGPU[i]) {
			printf("err on %d-th element\ncpu[%d] = %d, gpu[%d] = %d\n", i, i, outCPU[i], i, outGPU[i]);	
			return 0;
		}
	}
	return 1;
}
