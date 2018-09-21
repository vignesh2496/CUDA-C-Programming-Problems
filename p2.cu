#include<iostream>
#define THREADS_PER_BLOCK 256
#define BLOCKS 128
#define N (1 << 16)

using namespace std;

__global__ void add_array(float A[], float blocks[])
{
	__shared__ int array_per_block[THREADS_PER_BLOCK];
        int global_thread_ID = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x, step = gridDim.x * THREADS_PER_BLOCK, my_sum = 0, num_threads = THREADS_PER_BLOCK;
        for(int i = global_thread_ID; i < N; i += step)
                my_sum += A[i];
	array_per_block[threadIdx.x] = my_sum;
	__syncthreads();
	while(threadIdx.x < num_threads && num_threads > 1) 
        {
		if(threadIdx.x < num_threads / 2)
			array_per_block[threadIdx.x] +=  array_per_block[threadIdx.x + num_threads / 2];
		num_threads = num_threads >> 1;
		__syncthreads();
	}
	if(threadIdx.x == 0)
		blocks[blockIdx.x] = array_per_block[0];
}

void init_array(float A[])
{
        for(int i = 0; i < N; i++)
                A[i] = 1;
}

int main()
{
        float *host_A = new float[N], *host_blocks = new float[BLOCKS], *cuda_A, *cuda_blocks, final_sum = 0;
        init_array(host_A);
        cudaMalloc(&cuda_A, sizeof(float) * N);
	cudaMemcpy(cuda_A, host_A, sizeof(float) * N, cudaMemcpyHostToDevice);
        cudaMalloc(&cuda_blocks, sizeof(float) * BLOCKS);
	add_array<<<BLOCKS, THREADS_PER_BLOCK>>>(cuda_A, cuda_blocks);
	cudaMemcpy(host_blocks, cuda_blocks, sizeof(float) * BLOCKS, cudaMemcpyDeviceToHost); 
	for(int i = 0; i < BLOCKS; i++)
		final_sum += host_blocks[i];
	cout << "Final Sum : " << final_sum << endl;
	free(host_A); free(host_blocks);
	cudaFree(cuda_A); cudaFree(cuda_blocks);
        return 0;
}

