#include <bits/stdc++.h>
#define N 16
#define K_SIZE 3
#define BLOCK_DIM 16

using namespace std;

__global__ void convolve(float img[], float kernel[], float conv_img[])
{
	int global_x = threadIdx.x + blockIdx.x * blockDim.x, global_y = threadIdx.y + blockIdx.y * blockDim.y, global_ID = N * global_y + global_x;
	int conv_size = blockDim.x + K_SIZE - 1, pad = K_SIZE / 2;
	__shared__ float block_sub_matrix[BLOCK_DIM + K_SIZE - 1][BLOCK_DIM + K_SIZE - 1][3];

	for(int k = 0; k < 3; k++)
	{
		// Left-Top
		if(global_y - pad >= 0 && global_x - pad >= 0)
			block_sub_matrix[threadIdx.y][threadIdx.x][k] = img[((global_y - pad) * N + (global_x - pad)) * 3 + k];
		else
			block_sub_matrix[threadIdx.y][threadIdx.x][k] = 0;

		// Right-Top
		if(global_y - pad >= 0 && global_x + pad < N)
			block_sub_matrix[threadIdx.y][threadIdx.x + K_SIZE - 1][k] = img[((global_y - pad) * N + (global_x + pad)) * 3 + k];
		else
			block_sub_matrix[threadIdx.y][threadIdx.x + K_SIZE - 1][k] = 0;

		// Left-Bottom
		if(global_y + pad < N && global_x - pad >= 0)
			block_sub_matrix[threadIdx.y + K_SIZE - 1][threadIdx.x][k] = img[((global_y + pad) * N + (global_x - pad)) * 3 + k];
		else
			block_sub_matrix[threadIdx.y + K_SIZE - 1][threadIdx.x][k] = 0;

		// Right-Bottom
		if(global_y + pad < N && global_x + pad < N)
			block_sub_matrix[threadIdx.y + K_SIZE - 1][threadIdx.x + K_SIZE - 1][k] = img[((global_y + pad) * N + (global_x + pad)) * 3 + k];
		else
			block_sub_matrix[threadIdx.y + K_SIZE - 1][threadIdx.x + K_SIZE - 1][k] = 0;
	}

	__syncthreads();

	for(int k = 0; k < 3; k++)
	{
		conv_img[global_ID * 3 + k] = 0;
		for(int y = 0; y < K_SIZE; y++)
			for(int x = 0; x < K_SIZE; x++)
				conv_img[global_ID * 3 + k] += block_sub_matrix[threadIdx.y + K_SIZE - 1 - y][threadIdx.x + K_SIZE - 1 - x][k] * kernel[y * K_SIZE + x];
	}
}

void print_matrix(float mat[])
{
	for(int k = 0; k < 3; k++)
	{
		for(int i = 0; i < N; i++)
		{
			for(int j = 0; j < N; j++)
				cout << mat[(i * N + j) * 3 + k] << "  ";
			cout << endl;
		}
		cout << endl << endl;
	}
} 

void init_matrix(float mat[])
{
	for(int k = 0; k < 3; k++)
	{
		for(int i = 0; i < N; i++)
			for(int j = 0; j < N; j++)
				mat[(i * N + j) * 3 + k] = 1;
	} 
}

void init_kernel(float kernel[])
{
	for(int i = 0; i < K_SIZE; i++)
		for(int j = 0; j < K_SIZE; j++)
			kernel[i * K_SIZE + j] = 1.0 / 9;
  
}

int main()
{
	float *host_img = new float[N * N * 3], *host_kernel = new float[K_SIZE * K_SIZE], *host_conv_img = new float[N * N * 3], *cuda_img, *cuda_kernel, *cuda_conv_img;
	// Assuming N is a multiple of 16 
	dim3 grid_dim(N / BLOCK_DIM, N / BLOCK_DIM), block_dim(BLOCK_DIM, BLOCK_DIM);
	init_matrix(host_img);
	print_matrix(host_img);
	init_kernel(host_kernel); 
	cudaMalloc(&cuda_img, sizeof(float) * N * N * 3);
	cudaMalloc(&cuda_kernel, sizeof(float) * K_SIZE * K_SIZE);
	cudaMalloc(&cuda_conv_img, sizeof(float) * N * N * 3);
	cudaMemcpy(cuda_img, host_img, sizeof(float) * N * N * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_kernel, host_kernel, sizeof(float) * K_SIZE * K_SIZE, cudaMemcpyHostToDevice);
	convolve<<<grid_dim, block_dim>>>(cuda_img, cuda_kernel, cuda_conv_img);
	cudaMemcpy(host_conv_img, cuda_conv_img, sizeof(float) * N * N * 3, cudaMemcpyDeviceToHost);
    	print_matrix(host_conv_img);
	free(host_img); free(host_kernel); free(host_conv_img);
	cudaFree(cuda_img); cudaFree(cuda_kernel); cudaFree(cuda_conv_img);
	return 0;
}
