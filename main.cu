#include <stdio.h>
#include <cuda.h>
#include "sequential/render.h"
#include "cuda/render.h"



int main(int argc, char* argv[]){
    set_memory();
    set_memory_cuda();
    set_new_memory_cuda<<<1,1>>>();
  
    cudaDeviceSynchronize();
    
    // render_sequential_exhaustive(argc, argv);
    // render_sequential_barneshut(argc, argv);
    render_cuda_exhaustive(argc, argv);
    

    cudaDeviceSynchronize();
    
    free_new_memory_cuda<<<1,1>>>();
    free_memory();
    return 0;
}