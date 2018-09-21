#include<bits/stdc++.h>
#define N 16
#define BLOCK_DIM 16

using namespace std;

__global__ void multiply(float A[], float B[], float C[])
{
        __shared__ float sub_A[BLOCK_DIM][BLOCK_DIM], sub_B[BLOCK_DIM][BLOCK_DIM];
        int global_x = threadIdx.x + blockIdx.x * blockDim.x, global_y = threadIdx.y + blockIdx.y * blockDim.y, global_ID = global_y * N + global_x;
        C[global_ID] = 0;
        for(int i = 0; i < N / BLOCK_DIM; i++)
        {
         	sub_A[threadIdx.y][threadIdx.x] = A[global_y * N + global_x + BLOCK_DIM * i];
         	sub_B[threadIdx.y][threadIdx.x] = B[(global_y + BLOCK_DIM * i) * N + global_x];	       
        	__syncthreads();
        	for(int j = 0; j < BLOCK_DIM; j++)
        		C[global_ID] += sub_A[threadIdx.y][j] * sub_B[j][threadIdx.x];
		__syncthreads();
        }
}

void init_matrix(float mat[])
{
	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++)
			mat[i * N + j] = 1;
}

void print_matrix(float mat[])
{

	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
			cout << mat[i * N + j] << "  ";
		cout << endl;
	}
	cout << endl;
}

int main()
{
	float *A = new float[N * N], *B = new float[N * N], *C = new float[N * N], *cuda_A, *cuda_B, * cuda_C;
	init_matrix(A);
	cout << "A : " << endl;
	print_matrix(A);
	init_matrix(B);
	cout << "B : " << endl;
	print_matrix(B);
	cudaMalloc(&cuda_A, sizeof(float) * N * N);
	cudaMalloc(&cuda_B, sizeof(float) * N * N);
	cudaMalloc(&cuda_C, sizeof(float) * N * N);
	cudaMemcpy(cuda_A, A, sizeof(float) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_B, B, sizeof(float) * N * N, cudaMemcpyHostToDevice);
	dim3 grid_dim(N / BLOCK_DIM, N / BLOCK_DIM), block_dim(BLOCK_DIM, BLOCK_DIM);
	multiply<<<grid_dim, block_dim>>>(cuda_A, cuda_B, cuda_C);
	cudaMemcpy(C, cuda_C, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
	cout << "C : " << endl;
	print_matrix(C);	
	return 0;
}
