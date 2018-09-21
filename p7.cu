#include <bits/stdc++.h>

#define N (1<<20)
#define NUM_BINS 4096

using namespace std;


void init_array( unsigned int a[], int k, int size)
{
        for( int i = 0; i < size; i++)
       	{
		if(k == 0)
        	a[i] = rand() % (NUM_BINS);
		else if(k == 1)
		a[i] = 0;
      	}
}

__global__ void hist(unsigned int cuda_input[],unsigned int cuda_result[])
{
        __shared__ unsigned int histogram[NUM_BINS];
        int ID = threadIdx.x + blockIdx.x * blockDim.x;
	for( int i = threadIdx.x ; i < NUM_BINS ; i+= (blockDim.x))
	{
		histogram[i] = 0;
	}

	__syncthreads();
        for(int  i = ID ; i < N; i += (gridDim.x * blockDim.x))
        {
                atomicAdd(&(histogram[cuda_input[i]]),1);
        }
        __syncthreads();

        for( int i = threadIdx.x ; i < NUM_BINS ; i += blockDim.x)
        {
                atomicAdd(&(cuda_result[i]), histogram[i]);
        }
}

int main()
{
        unsigned int * input = (unsigned int *)malloc(sizeof(unsigned int) * N) ;
        unsigned int * result = (unsigned int *)malloc(sizeof(unsigned int) * NUM_BINS) ;
        init_array(input, 0, N);
        unsigned int *cuda_input, *cuda_result;
        cudaMalloc(&cuda_input,sizeof(unsigned int) * N);
        cudaMalloc(&cuda_result,sizeof(unsigned int) * NUM_BINS);
        init_array(result, 1, NUM_BINS);
	cudaMemcpy(cuda_input, input, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_result, result, sizeof(unsigned int) * NUM_BINS, cudaMemcpyHostToDevice);
        int block_dim = 256;
        int grid_dim = 256;
        hist <<< block_dim, grid_dim >>> (cuda_input,cuda_result); 
        cudaMemcpy(result, cuda_result, sizeof(unsigned int) * NUM_BINS , cudaMemcpyDeviceToHost);
        for(int i = 0 ; i < NUM_BINS ; i++)
        {	
		result[i] = min(result[i], 127);
                printf("%u ",result[i]);
        }
        free(input); 
        free(result);
        cudaFree(cuda_input); 
        cudaFree(cuda_result);
        return 0;
}

