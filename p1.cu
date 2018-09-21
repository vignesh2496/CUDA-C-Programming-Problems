#include<stdio.h>

int main() 
{
	int n_devices;
	cudaGetDeviceCount(&n_devices);
  	for (int i = 0; i < n_devices; i++)
  	{
    		cudaDeviceProp prop;
    		cudaGetDeviceProperties(&prop, i);
    		printf("  Device number: %d\n", i);
    		printf("  Device name: %s\n", prop.name);
    		printf("  Memory clock rate (KHz): %d\n", prop.memoryClockRate);
    		printf("  Memory bus width (bits): %d\n", prop.memoryBusWidth);
    		printf("  Peak memory bandwidth (GB/s): %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
		printf("  Maximum number of grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("  Total constant memory: %d\n", prop.totalConstMem);
		printf("  Warp size: %d\n", prop.warpSize);
  	}
	return 0;
}
