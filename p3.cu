#include<iostream>
#define M 6
#define N 6
#define THREADS_PER_BLOCK 256

using namespace std;

__global__ void add_matrix(int mat_1[], int mat_2[], int mat_sum[])
{
	int global_thread_ID = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if(global_thread_ID < M * N)
		mat_sum[global_thread_ID] = mat_1[global_thread_ID] + mat_2[global_thread_ID];
}

void print_matrix(int mat[])
{
	for(int i = 0; i < M; i++)
	{
		for(int j = 0; j < N; j++)
			cout << mat[i * N + j] << "  ";
		cout << endl;
	}
	cout << endl;
} 

void init_matrix(int mat[])
{
	for(int i = 0; i < M; i++)
		for(int j = 0; j < N; j++)
			mat[i * N + j] = i * N + j; 
}

int main()
{
	int *host_A = new int[M * N], *host_B = new int[M * N], *host_sum = new int[M * N], *cuda_A, *cuda_B, *cuda_sum;
	int blocks = M * N / THREADS_PER_BLOCK + ((M * N % THREADS_PER_BLOCK == 0) ? 0 : 1); 
	init_matrix(host_A);
	cout << "A:\n"; 
	print_matrix(host_A);
	init_matrix(host_B);
	cout << "B:\n";
	print_matrix(host_B);
	cudaMalloc(&cuda_A, sizeof(int) * M * N);
	cudaMalloc(&cuda_B, sizeof(int) * M * N);
	cudaMalloc(&cuda_sum, sizeof(int) * M * N);
	cudaMemcpy(cuda_A, host_A, sizeof(int) * M * N, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_B, host_B, sizeof(int) * M * N, cudaMemcpyHostToDevice);
	add_matrix<<<blocks, THREADS_PER_BLOCK>>>(cuda_A, cuda_B, cuda_sum);
	cudaMemcpy(host_sum, cuda_sum, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
	cout << "A + B:\n";
	print_matrix(host_sum);
	free(host_A); free(host_B); free(host_sum);
	cudaFree(cuda_A); cudaFree(cuda_B); cudaFree(cuda_sum);
	return 0;
}
