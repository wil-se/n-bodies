#include <cuda.h>
#include <stdio.h>
#include "common.h"
#include "../sequential/common.h"

#define G 6.67e-11
#define dt 1000000


__managed__ double *x_d, *y_d, *z_d, *new_x_d, *new_y_d, *new_z_d, *new_sx_d, *new_sy_d, *new_sz_d, *mass_d, *sx_d, *sy_d, *sz_d;
__managed__ int n_d;

void set_memory_cuda(){
	//cudaMalloc((void**)&n_d, sizeof(int));
    cudaMalloc((void**)&x_d, sizeof(double)*n);
    cudaMalloc((void**)&y_d, sizeof(double)*n);
    cudaMalloc((void**)&z_d, sizeof(double)*n);
    cudaMalloc((void**)&mass_d, sizeof(double)*n);
    cudaMalloc((void**)&sx_d, sizeof(double)*n);
    cudaMalloc((void**)&sy_d, sizeof(double)*n);
    cudaMalloc((void**)&sz_d, sizeof(double)*n);
    
    cudaMemcpy(&n_d, &n, sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMemcpy(x_d, x, sizeof(double)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, sizeof(double)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(z_d, z, sizeof(double)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(mass_d, mass, sizeof(double)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(sx_d, sx, sizeof(double)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(sy_d, sy, sizeof(double)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(sz_d, sz, sizeof(double)*n, cudaMemcpyHostToDevice);


    cudaDeviceSynchronize();
}

void swap_memory(){
	cudaMemcpy(x, x_d, sizeof(double)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, y_d, sizeof(double)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(z, z_d, sizeof(double)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(mass, mass_d, sizeof(double)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(sx, sx_d, sizeof(double)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(sy, sy_d, sizeof(double)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(sz, sz_d, sizeof(double)*n, cudaMemcpyDeviceToHost);
	// cudaDeviceSynchronize();
}

void free_memory_cuda(){
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
    cudaFree(mass_d);
    cudaFree(sx_d);
    cudaFree(sy_d);
    cudaFree(sz_d);
}

__global__ void set_new_memory_cuda(){
	cudaMalloc((void**)&new_x_d, sizeof(double)*n_d);
	cudaMalloc((void**)&new_y_d, sizeof(double)*n_d);
	cudaMalloc((void**)&new_z_d, sizeof(double)*n_d);
	cudaMalloc((void**)&new_sx_d, sizeof(double)*n_d);
	cudaMalloc((void**)&new_sy_d, sizeof(double)*n_d);
	cudaMalloc((void**)&new_sz_d, sizeof(double)*n_d);
}

__global__ void free_new_memory_cuda(){
	cudaFree(new_x_d);
	cudaFree(new_y_d);
	cudaFree(new_z_d);
	cudaFree(new_sx_d);
	cudaFree(new_sy_d);
	cudaFree(new_sz_d);
}

__global__ void set_new_vectors_cuda(){
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    new_x_d[i] = x_d[i];
    new_y_d[i] = y_d[i];
    new_z_d[i] = z_d[i];
    new_sx_d[i] = sx_d[i];
    new_sy_d[i] = sy_d[i];
    new_sz_d[i] = sz_d[i];      
}

__global__ void set_vectors_cuda(){
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    x_d[i] = new_x_d[i];
    y_d[i] = new_y_d[i];
    z_d[i] = new_z_d[i];
    sx_d[i] = new_sx_d[i];
    sy_d[i] = new_sy_d[i];
    sz_d[i] = new_sz_d[i];      
}
