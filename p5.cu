#include <bits/stdc++.h>
#define N 16

using namespace std;

__global__ void RGBtoGray(float img[], float gray_img[])
{
        int ID = threadIdx.x + blockIdx.x * blockDim.x;
        for(int  i = ID ; i < N * N; i += gridDim.x * blockIdx.x)
        {
                gray_img[i] = 0.21 * img[i * 3] + 0.71 * img[i * 3 + 1] + 0.07 * img[i * 3 + 2];
        }
        __syncthreads();

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

void print_gray_matrix(float mat[])
{
        for(int i = 0; i < N; i++)
        {
                for(int j = 0; j < N; j++)
                        cout << mat[(i * N + j)] << "  ";
                cout << endl;
        }
        cout << endl << endl;
}

void init_matrix(float mat[])
{
        for(int k = 0; k < 3; k++)
        {
                for(int i = 0; i < N; i++)
                        for(int j = 0; j < N; j++)
                                mat[(i * N + j) * 3 + k] = k + 1;
        }
}


int main()
{
        float *host_img = new float[N * N * 3], *host_gray_img = new float[N * N ], *cuda_img, *cuda_gray_img;
        // Assuming N is a multiple of 16 
        //dim3 grid_dim(N / 16, N / 16), block_dim(16, 16);
        int block_dim = 256, grid_dim;
        if( (N * N) % 256 == 0)
        {
                grid_dim = (N * N) / 256;
        }
        else
        {
               	grid_dim = (N * N) / 256 + 1;
        }
        init_matrix(host_img);
        print_matrix(host_img);
        cudaMalloc(&cuda_img, sizeof(float) * N * N * 3);
        cudaMalloc(&cuda_gray_img, sizeof(float) * N * N );
        cudaMemcpy(cuda_img, host_img, sizeof(float) * N * N * 3, cudaMemcpyHostToDevice);
        RGBtoGray<<<grid_dim, block_dim>>>(cuda_img, cuda_gray_img);
        cudaMemcpy(host_gray_img, cuda_gray_img, sizeof(float) * N * N , cudaMemcpyDeviceToHost);
        print_gray_matrix(host_gray_img);
        free(host_img); 
        free(host_gray_img);
        cudaFree(cuda_img); 
        cudaFree(cuda_gray_img);
        return 0;
}

