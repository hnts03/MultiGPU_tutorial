/*
 * This is main function for tutorial.
 * We'll use 3 A6000 GPUs
 * So the matrix size will be the multiply of 3.
 *
 */

// -------------------------------------------------------------------------

// C standard libraries
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA libraries
// #include <cuda.h>
// #include <cuda_runtime.h>

// User define header file
#include "kernel.cuh"

// -------------------------------------------------------------------------

// CUDA programming level define
#define BLOCK_SIZE 1024
// Grid size will be dynamically calculated

// Matrix Feature define
// #define A_H 1024
#define A_H 2048
#define A_W A_H
#define B_H A_W
#define B_W A_H

// Datatype define
#define DTYPE int

// Stream size define
#define NUM_STREAMS 4

// -------------------------------------------------------------------------

#if (CPU)
int main(){

	time_t start, end;

	// Matrix Declaration
	DTYPE *matA, *matB, *matC;
	
	int num_gpus;
	cudaGetDeviceCount(&num_gpus);

	int height = A_H * num_gpus;
	int width = height;
	int input_matrix_size = height * width;
	int output_matrix_size = height * width;
	
	printf("The input matrices have same size.\nHeight: %d. Width: %d\n", height, width);

	// Initialize
	cudaMallocHost((void**)&matA, sizeof(DTYPE) * input_matrix_size);
	cudaMallocHost((void**)&matB, sizeof(DTYPE) * input_matrix_size);
	cudaMallocHost((void**)&matC, sizeof(DTYPE) * output_matrix_size);

	for (int i = 0; i < input_matrix_size; i++){
		matA[i] = 1;
		matB[i] = 2;
	}
	
	// Run kernel
	start = clock();
	CPU_kernel(matA, matB, matC, height, width, width);
	end = clock();
#ifdef DEBUG
	for (int i = 0; i < output_matrix_size; i++){
		printf("%d ", matC[i]);
		if(i % A_H == 0) printf("\n");
	}
#endif
	printf("CPU: Kernel runtime: %lfs\n", (double)(end-start)/CLOCKS_PER_SEC);

	return 0;
}



#endif // CPU end

#if (GPU)
#if (MULTI == 0 && STREAM_ENABLE == 0) // Single GPU, No Stream uses
int main(){
	// input matA_H, matB_H, output: matC_H
	// matrix is 1d-array but it is unfolded shape.
	DTYPE *matA_H, *matB_H, *matC_H;
	DTYPE *matA_D, *matB_D, *matC_D;

	time_t start, end;
	
	int num_gpus;
	cudaGetDeviceCount(&num_gpus);
	int height = A_H * num_gpus;
	int width = height;

	printf("The input matrices have same size.\nHeight: %d. Width: %d\n", height, width);

	long long unsigned int input_matrix_size = height * width;
	long long unsigned int output_matrix_size = height * width;

	// A_H * (ratio * A_W) X (B_H * ratio) * B_W = A_H * B*W
	cudaMallocHost((void**)&matA_H, sizeof(DTYPE) * input_matrix_size);
	cudaMallocHost((void**)&matB_H, sizeof(DTYPE) * input_matrix_size);
	cudaMallocHost((void**)&matC_H, sizeof(DTYPE) * output_matrix_size);

	
	// Initialize input matrix
	for (int i = 0; i < input_matrix_size; i++){
		matA_H[i] = 1;
		matB_H[i] = 2;
	}
	
	// Memory allocation on GPU's off-chip memory
	cudaMalloc((void**)&matA_D, sizeof(DTYPE) * input_matrix_size);
	cudaMalloc((void**)&matB_D, sizeof(DTYPE) * input_matrix_size);
	cudaMalloc((void**)&matC_D, sizeof(DTYPE) * output_matrix_size);

	// Memory Copy CPU -> GPU
	start = clock();
	cudaMemcpy(matA_D, matA_H, sizeof(DTYPE)*input_matrix_size, cudaMemcpyHostToDevice);
	cudaMemcpy(matB_D, matB_H, sizeof(DTYPE)*input_matrix_size, cudaMemcpyHostToDevice);
//	end = clock();
//	printf("Memcpy CPU -> GPU: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);


	// Define grid dimension and block dimension
	dim3 grid_dim(height);				// 1 CTA per matA's row
	dim3 block_dim(BLOCK_SIZE);			// 1 CTA has fixed number of threads(1024)
	
//	start = clock();
	GPU_kernel<<<grid_dim, block_dim>>>(matA_D, matB_D, matC_D, height, width, width);
	cudaDeviceSynchronize();
//	end = clock();
//	printf("Kernel runtime: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	// Memory Copy GPU -> CPU
//	start = clock();
	cudaMemcpy(matC_H, matC_D, sizeof(DTYPE)*output_matrix_size, cudaMemcpyDeviceToHost);
	end = clock();
//	printf("Memcpy GPU -> CPU: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
	printf("SingleGPU & NoStream: Total Runtime: %lfms\n\n", (double)(end-start)/CLOCKS_PER_SEC*1000);


#ifdef DEBUG
	int* outCPU;
	cudaMallocHost((void**)&outCPU, sizeof(DTYPE) * output_matrix_size);
	CPU_kernel(matA_H, matB_H, outCPU, height, width, width);
	if(!is_valid(outCPU, matC_H, output_matrix_size)) {
		printf("wrong!\n");
		exit(0);
	}
	printf("done!\n");
#endif


	return 0;
}

#elif (MULTI == 1 && STREAM_ENABLE == 0) // MultiGPU, No Stream uses
int main(){
	// input matA_H, matB_H, output: matC_H
	// matrix is 1d-array but it is unfolded shape.
	int num_gpus;
	cudaGetDeviceCount(&num_gpus);

	DTYPE *matA_H, *matB_H, *matC_H;
	DTYPE *matA_D[num_gpus], *matB_D[num_gpus], *matC_D[num_gpus];

	time_t start, end;
	
	int host_height = A_H * num_gpus;
	int host_width = host_height;

	int dev_height = A_H;
	int dev_width = dev_height * num_gpus;

	printf("The input matrices have same size.\nHeight: %d. Width: %d\n", host_height, host_width);
	printf("The input matrices to GPUs have same size.\nHeight: %d. Width: %d\n", dev_height, dev_width);

	long long unsigned int host_input_matrix_size = host_height * host_width;
	long long unsigned int host_output_matrix_size = host_height * host_width;

	long long unsigned int dev_input_matrix_size = dev_height * dev_width;
	long long unsigned int dev_output_matrix_size = dev_height * dev_width;

	cudaMallocHost((void**)&matA_H, sizeof(DTYPE) * host_input_matrix_size);
	cudaMallocHost((void**)&matB_H, sizeof(DTYPE) * host_input_matrix_size);
	cudaMallocHost((void**)&matC_H, sizeof(DTYPE) * host_output_matrix_size);

	
	// Initialize input matrix
	for (int i = 0; i < host_input_matrix_size; i++){
		matA_H[i] = 1;
		matB_H[i] = 2;
	}
	
	// Memory allocation on GPU's off-chip memory
	for (int gpuid = 0; gpuid < num_gpus; gpuid++){
		cudaSetDevice(gpuid); // Set specific GPU to use it
		
		// Devide matrixA into num_gpus tiles for row but matrixB still has origin shape
		cudaMalloc((void**)&matA_D[gpuid], sizeof(DTYPE) * dev_input_matrix_size);
		cudaMalloc((void**)&matB_D[gpuid], sizeof(DTYPE) * host_input_matrix_size);
		cudaMalloc((void**)&matC_D[gpuid], sizeof(DTYPE) * dev_output_matrix_size);
	}

	// Memory Copy CPU -> GPU
	start = clock();
	for (int gpuid = 0; gpuid < num_gpus; gpuid++){
		cudaSetDevice(gpuid);

		cudaMemcpy(matA_D[gpuid], matA_H + (gpuid*dev_input_matrix_size), sizeof(DTYPE)*dev_input_matrix_size, 
																				cudaMemcpyHostToDevice);
		cudaMemcpy(matB_D[gpuid], matB_H, sizeof(DTYPE)*host_input_matrix_size, cudaMemcpyHostToDevice);
	}


	// Define grid dimension and block dimension
	dim3 grid_dim(host_height/num_gpus);				// 1 CTA per matA's row
	dim3 block_dim(BLOCK_SIZE);			// 1 CTA has fixed number of threads(1024)
	
	for (int gpuid = 0; gpuid < num_gpus; gpuid++){
		cudaSetDevice(gpuid);
		GPU_kernel<<<grid_dim, block_dim>>>(matA_D[gpuid], matB_D[gpuid], matC_D[gpuid],
																		dev_height, dev_width, dev_width);
	}

	// Memory Copy GPU -> CPU
	for (int gpuid = 0; gpuid < num_gpus; gpuid++){
		cudaSetDevice(gpuid);
		cudaMemcpy(matC_H+(dev_output_matrix_size * gpuid), matC_D[gpuid], sizeof(DTYPE)*dev_output_matrix_size, 
																				cudaMemcpyDeviceToHost);
	}
	end = clock();
	printf("MultiGPU & NoStream: Total Runtime: %lfms\n\n", (double)(end-start)/CLOCKS_PER_SEC*1000);


#ifdef DEBUG
	int* outCPU;
	cudaMallocHost((void**)&outCPU, sizeof(DTYPE) * host_output_matrix_size);
	CPU_kernel(matA_H, matB_H, outCPU, host_height, host_width, host_width);
	if(!is_valid(outCPU, matC_H, host_output_matrix_size)) {
		printf("wrong!\n");
		exit(0);
	}
	printf("done!\n");
#endif


	return 0;
}

#elif (MULTI == 0 && STREAM_ENABLE == 1)		// Single GPU and using Stream features
int main(){
	// input matA_H, matB_H, output: matC_H
	// matrix is 1d-array but it is unfolded shape.
	int num_gpus;
	cudaGetDeviceCount(&num_gpus);

	DTYPE *matA_H, *matB_H, *matC_H;
	DTYPE *matA_D, *matB_D, *matC_D;

	time_t start, end;
	
	// Declare and Initialize cudaStream_t object for single GPU
	int num_streams = NUM_STREAMS;
	cudaStream_t streams[num_streams];
	for(uint64_t stream = 0; stream < num_streams; stream++){
		cudaStreamCreate(&streams[stream]);
	}

	// int host_height = A_H * num_streams; 
	int host_height = A_H * num_gpus; 
	int host_width = host_height;

	int dev_height = A_H * num_gpus / num_streams;
	//int dev_width = dev_height * num_streams;
	int dev_width = A_H * num_gpus;


	printf("The input matrices have same size.\nHeight: %d. Width: %d\n", host_height, host_width);
	printf("The input matrices splited into num_streams which have same size.\nHeight: %d. Width: %d\n", dev_height, dev_width);
	

	long long unsigned int host_input_matrix_size = host_height * host_width;
	long long unsigned int host_output_matrix_size = host_height * host_width;

	long long unsigned int dev_input_matrix_size = dev_height * dev_width;
	long long unsigned int dev_output_matrix_size = dev_height * dev_width;

	cudaMallocHost((void**)&matA_H, sizeof(DTYPE) * host_input_matrix_size);
	cudaMallocHost((void**)&matB_H, sizeof(DTYPE) * host_input_matrix_size);
	cudaMallocHost((void**)&matC_H, sizeof(DTYPE) * host_output_matrix_size);

	
	// Initialize input matrix
	for (int i = 0; i < host_input_matrix_size; i++){
		matA_H[i] = 1;
		matB_H[i] = 2;
	}

	
	// Memory allocation on GPU's off-chip memory
	cudaMalloc((void**)&matA_D, sizeof(DTYPE) * host_input_matrix_size);
	cudaMalloc((void**)&matB_D, sizeof(DTYPE) * host_input_matrix_size);
	cudaMalloc((void**)&matC_D, sizeof(DTYPE) * host_input_matrix_size);


	// Define grid and block dimension
	dim3 grid_dim(host_height/num_streams);
	dim3 block_dim(BLOCK_SIZE);


	// Asynchronously Memcpy and run kernel
	start = clock();
	cudaMemcpy(matB_D, matB_H, sizeof(DTYPE)*host_input_matrix_size, cudaMemcpyHostToDevice);
	for (uint64_t stream = 0; stream < num_streams; stream++){
		// memcpy cpu -> gpu
		cudaMemcpyAsync(matA_D+(stream*dev_input_matrix_size), matA_H+(stream*dev_input_matrix_size), 
							sizeof(DTYPE)*dev_input_matrix_size, cudaMemcpyHostToDevice, streams[stream]);

		// run kernel
		GPU_kernel<<<grid_dim, block_dim, 0, streams[stream]>>>(
				matA_D + (stream*dev_input_matrix_size), matB_D, 
				matC_D + (stream*dev_output_matrix_size), 
				dev_height, dev_width, dev_width
		);

		// memcpy gpu -> cpu
		cudaMemcpyAsync(matC_H+(stream*dev_output_matrix_size), matC_D+(stream*dev_output_matrix_size),
							sizeof(DTYPE)*dev_output_matrix_size, cudaMemcpyDeviceToHost, streams[stream]);
	}

	// Synchronize the each streams
	for (uint64_t stream = 0; stream < num_streams; stream++){
		cudaStreamSynchronize(streams[stream]);
	}
	end = clock();
	printf("SingleGPU & Stream: Total Runtime: %lfms\n\n", (double)(end-start)/CLOCKS_PER_SEC*1000);

#ifdef DEBUG
	int* outCPU;
	cudaMallocHost((void**)&outCPU, sizeof(DTYPE) * host_output_matrix_size);
	CPU_kernel(matA_H, matB_H, outCPU, host_height, host_width, host_width);
	if(!is_valid(outCPU, matC_H, host_output_matrix_size)) {
		printf("wrong!\n");
		exit(0);
	}
	printf("done!\n");
#endif


	return 0;
}

#else	// Multi-gpu & Stream enabled
int main(){
	time_t start, end;
	
	int num_gpus;
	cudaGetDeviceCount(&num_gpus);

	int num_streams = NUM_STREAMS;
	cudaStream_t streams[num_gpus][num_streams];

	for (int gpuid = 0; gpuid < num_gpus; gpuid++){
		cudaSetDevice(gpuid);
		for (uint64_t stream = 0; stream < num_streams; stream++)
			cudaStreamCreate(&streams[gpuid][stream]);
	}


	int height = A_H * num_gpus;
	int width = height;

	int dev_height = A_H;
	int dev_width = A_H * num_gpus;

	int strm_height = A_H / num_streams;
	int strm_width = A_H * num_gpus;


	printf("The input matrices have same size.\nHeight: %d. Width: %d\n", height, width);
	printf("The input matrices are splited into single gpu.\nHeight: %d. Width: %d\n", dev_height, dev_width);
	printf("The input matrix A is splited into each streams.\nHeight: %d. Width: %d\n", strm_height, strm_width);

	long long unsigned int host_input_matrix_size = height * width;
	long long unsigned int host_output_matrix_size = height * width;

	long long unsigned int dev_input_matrix_size = dev_height * dev_width;
	long long unsigned int dev_output_matrix_size = dev_height * dev_width;

	long long unsigned int strm_input_matrix_size = strm_height * strm_width;
	long long unsigned int strm_output_matrix_size = strm_height * strm_width;

	// input matA_H, matB_H, output: matC_H
	// matrix is 1d-array but it is unfolded shape.
	DTYPE *matA_H, *matB_H, *matC_H;
	DTYPE *matA_D[num_gpus][num_streams], *matB_D[num_gpus], *matC_D[num_gpus][num_streams]; // matB will be shared over streams within a single gpu

	// Memory allocation on host size
	// A_H * (ratio * A_W) X (B_H * ratio) * B_W = A_H * B*W
	cudaMallocHost((void**)&matA_H, sizeof(DTYPE) * host_input_matrix_size);
	cudaMallocHost((void**)&matB_H, sizeof(DTYPE) * host_input_matrix_size);
	cudaMallocHost((void**)&matC_H, sizeof(DTYPE) * host_output_matrix_size);

	
	// Initialize input matrix
	for (int i = 0; i < host_input_matrix_size; i++){
		matA_H[i] = 1;
		matB_H[i] = 2;
	}
	
	// Memory allocation on GPU's off-chip memory
	for (int gpuid = 0; gpuid < num_gpus; gpuid++){
		cudaSetDevice(gpuid);
		for (uint64_t stream = 0; stream < num_streams; stream++){
			cudaMalloc((void**)&matA_D[gpuid][stream], sizeof(DTYPE) * strm_input_matrix_size);
			cudaMalloc((void**)&matC_D[gpuid][stream], sizeof(DTYPE) * strm_output_matrix_size);
		}
		cudaMalloc((void**)&matB_D[gpuid], sizeof(DTYPE) * host_input_matrix_size);
	}


	// Set grid and block dimension
	dim3 grid_dim(height/num_gpus/num_streams);
	dim3 block_dim(BLOCK_SIZE);
	
	// Kernel launch asynchronously
	start = clock();
	for (int gpuid = 0; gpuid < num_gpus; gpuid++){
		cudaSetDevice(gpuid);
		cudaMemcpy(matB_D[gpuid], matB_H, sizeof(DTYPE) * host_input_matrix_size, cudaMemcpyHostToDevice);
		for (uint64_t stream = 0; stream < num_streams; stream++){
			// memcpy cpu -> gpu
			cudaMemcpyAsync(matA_D[gpuid][stream], matA_H+(gpuid*dev_input_matrix_size)+(stream+strm_input_matrix_size),
					sizeof(DTYPE)*strm_input_matrix_size, cudaMemcpyHostToDevice, streams[gpuid][stream]);

			// launch kernel
			GPU_kernel<<<grid_dim, block_dim, 0, streams[gpuid][stream]>>>(
					matA_D[gpuid][stream], matB_D[gpuid], matC_D[gpuid][stream],
					strm_height, strm_width, dev_width
			);

			// memcpu gpu -> cpu
			cudaMemcpyAsync(matC_H+(gpuid*dev_output_matrix_size)+(stream*strm_output_matrix_size), matC_D[gpuid][stream],
					sizeof(DTYPE)*strm_output_matrix_size, cudaMemcpyDeviceToHost, streams[gpuid][stream]);
		}
	}

	// Synchronize each streams within single gpu device
	for (int gpuid = 0; gpuid < num_gpus; gpuid++){
		cudaSetDevice(gpuid);
		for (uint64_t stream = 0; stream < num_streams; stream++)
			cudaStreamSynchronize(streams[gpuid][stream]);
	}
	end = clock();
	printf("MultiGPU & Stream: Total Runtime: %lfms\n\n", (double)(end-start)/CLOCKS_PER_SEC*1000);



#ifdef DEBUG
	int* outCPU;
	cudaMallocHost((void**)&outCPU, sizeof(DTYPE) * host_output_matrix_size);
	CPU_kernel(matA_H, matB_H, outCPU, height, width, width);
	if(!is_valid(outCPU, matC_H, host_output_matrix_size)) {
		printf("wrong!\n");
		exit(0);
	}
	printf("done!\n");
#endif


	return 0;
}




#endif // STREAM_ENABLE end
#endif // GPU end
