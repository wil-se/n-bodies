#define G 6.67e-11
#define dt 1000000

extern __device__ __managed__ double *x_d, *y_d, *z_d, *new_x_d, *new_y_d, *new_z_d, *new_sx_d, *new_sy_d, *new_sz_d, *mass_d, *sx_d, *sy_d, *sz_d;
extern __device__ __managed__ int n_d;

void set_memory_cuda();
void swap_memory();
void free_memory_cuda();
__device__ void set_new_memory_cuda();
__device__ void free_new_memory_cuda();
__global__ void set_new_vectors_cuda();
__global__ void set_vectors_cuda();
