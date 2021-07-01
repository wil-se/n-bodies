#include <cuda.h>
#include <stdio.h>
#include "common.h"
#include "compute-barneshut.h"

__device__ int QUEUE_SIZE = 2048;

__device__ int *semaphore, *semaphore_mutex;

__device__ void print_node_cuda(bnode_cuda* node){
	printf("================================\nBODY: %d\nDEPTH: %d\nLOCK: %d\nMAX X: %d\nMAX Y: %d\nMAX Z: %d\nMIN X: %d\nMIN Y: %d\nMIN Z: %d\nX: %f\nY: %f\nZ: %f\nMASS: %f\n", node->body, node->depth, *node->mutex, node->max_x, node->max_y, node->max_z, node->min_x, node->min_y, node->min_z, node->x, node->y, node->z, node->mass);
}


__device__ void enqueue_cuda(queue_cuda* q, bnode_cuda* node){
        queue_node_cuda* q_node = create_queue_node_cuda(node);
        if(q->size >= q->max_size){
                return;
        }

        q_node->previous = NULL;

        if(q->size == 0){
                q->head = q_node;
                q->tail = q_node;
        }else{
                q->tail->previous = q_node;
                q->tail = q_node;
        }
        q->size++;

        return;
}

/*
   estrae un nodo dalla coda
 */
__device__ bnode_cuda* dequeue_cuda(queue_cuda* q){
        queue_node_cuda* head = q->head;
        bnode_cuda* node = head->body;
        q->head = head->previous;
        q->size--;
        cudaFree(head);
        return node;
}

/*
   crea la coda
 */
__device__ queue_cuda* create_queue_cuda(int max_size){
        queue_cuda* q;
        cudaMalloc((void**)&q, sizeof(queue_cuda));
        q->max_size = max_size;
        q->size = 0;
        q->head = NULL;
        q->tail = NULL;
        return q;
}

/*
   libera la coda
 */
__device__ void destruct_queue_cuda(queue_cuda* q){
        if(q->head == NULL){return;}
        queue_node_cuda* node = q->head;
        while(node->previous != NULL){
                queue_node_cuda* previous = node->previous;
                free(node);
                node = previous;
        }
        free(q);
}

/*
   crea un elemento della coda conente un nodo dell'albero
 */
__device__ queue_node_cuda* create_queue_node_cuda(bnode_cuda* node){
        queue_node_cuda* n;
        cudaMalloc((void**)&n, sizeof(queue_node_cuda));
        n->body = node;
        n->previous = NULL;
        return n;
}

/*
   stampa l'albero di barnes-hut con una visita in ampiezza
 */
__device__ void print_tree_cuda(bnode_cuda* node){
        queue_cuda* q = create_queue_cuda(QUEUE_SIZE);
        enqueue_cuda(q, node);
        while(q->size != 0){
                bnode_cuda* curr = dequeue_cuda(q);
                print_node_cuda(curr);
                if(curr->body == -2){
                        enqueue_cuda(q, curr->o0);
                        enqueue_cuda(q, curr->o1);
                        enqueue_cuda(q, curr->o2);
                        enqueue_cuda(q, curr->o3);
                        enqueue_cuda(q, curr->o4);
                        enqueue_cuda(q, curr->o5);
                        enqueue_cuda(q, curr->o6);
                        enqueue_cuda(q, curr->o7);
                }
        }
        destruct_queue_cuda(q);
}

/*
   visita l'albero di barnes-hut in ampiezza e libera la memoria
   per ogni nodo visitato
 */
__device__ void destruct_tree(bnode_cuda* root){
        queue_cuda* q = create_queue_cuda(QUEUE_SIZE);
        enqueue_cuda(q, root->o0);
        enqueue_cuda(q, root->o1);
        enqueue_cuda(q, root->o2);
        enqueue_cuda(q, root->o3);
        enqueue_cuda(q, root->o4);
        enqueue_cuda(q, root->o5);
        enqueue_cuda(q, root->o6);
        enqueue_cuda(q, root->o7);
        while(q->size != 0){
                bnode_cuda* curr = dequeue_cuda(q);
                if(curr->body == -2){
                        enqueue_cuda(q, curr->o0);
                        enqueue_cuda(q, curr->o1);
                        enqueue_cuda(q, curr->o2);
                        enqueue_cuda(q, curr->o3);
                        enqueue_cuda(q, curr->o4);
                        enqueue_cuda(q, curr->o5);
                        enqueue_cuda(q, curr->o6);
                        enqueue_cuda(q, curr->o7);
                }
                cudaFree(curr);
        }
        destruct_queue_cuda(q);
}


// un vettore diviso n thread
__global__ void get_max_x_cuda(double *result){
    extern __shared__ double sdata_x[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= n_d) return;
    
    sdata_x[tid] = fabsf(x_d[i]);
    
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
	    if (tid < s) {
            if(fabsf(sdata_x[tid + s]) > fabsf(sdata_x[tid])){
                sdata_x[tid] = fabsf(sdata_x[tid + s]);
            }
        }
	    __syncthreads();
    }

    cudaDeviceSynchronize();
    if(tid == 0) {
        *result = fabsf(sdata_x[0]);
    }
}

__global__ void get_max_y_cuda(double *result){
    extern __shared__ double sdata_y[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= n_d) return;
    
    sdata_y[tid] = fabsf(y_d[i]);
    
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
	    if (tid < s) {
            if(fabsf(sdata_y[tid + s]) > fabsf(sdata_y[tid])){
                sdata_y[tid] = fabsf(sdata_y[tid + s]);
            }
        }
	    __syncthreads();
    }

    cudaDeviceSynchronize();
    if(tid == 0) {
        *result = fabsf(sdata_y[0]);
    }
}


__global__ void get_max_z_cuda(double *result){
    extern __shared__ double sdata_z[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= n_d) return;
    
    sdata_z[tid] = fabsf(z_d[  i]);
    
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
	    if (tid < s) {
            if(fabsf(sdata_z[tid + s]) > fabsf(sdata_z[tid])){
                sdata_z[tid] = fabsf(sdata_z[tid + s]);
            }
        }
	    __syncthreads();
    }

    cudaDeviceSynchronize();

    if(tid == 0) {
        *result = fabsf(sdata_z[0]);
    }
}


__device__ void generate_empty_children_cuda(bnode_cuda *node){
    int depth = node->depth+1;
	int scalar = fabsf(node->max_x - node->min_x)/2;
    bnode_cuda *o0, *o1, *o2, *o3, *o4, *o5, *o6, *o7;

    cudaMalloc((void**)&o0, sizeof(bnode_cuda));
    cudaMalloc((void**)&o1, sizeof(bnode_cuda));
    cudaMalloc((void**)&o2, sizeof(bnode_cuda));
    cudaMalloc((void**)&o3, sizeof(bnode_cuda));
    cudaMalloc((void**)&o4, sizeof(bnode_cuda));
    cudaMalloc((void**)&o5, sizeof(bnode_cuda));
    cudaMalloc((void**)&o6, sizeof(bnode_cuda));
    cudaMalloc((void**)&o7, sizeof(bnode_cuda));


    o0->depth = depth;
    o0->body = -1;
    cudaMalloc((void**)&o0->mutex, sizeof(int));
    *o0->mutex = 0;
    o0->min_x = node->min_x + scalar;
    o0->max_x = node->max_x;
    o0->min_y = node->min_y + scalar;
    o0->max_y = node->max_y;
    o0->min_z = node->min_z + scalar;
    o0->max_z = node->max_z;
	o0->x = 0;
	o0->y = 0;
	o0->z = 0;
	o0->mass = 0;

    o1->depth = depth;
    o1->body = -1;
    cudaMalloc((void**)&o1->mutex, sizeof(int));
    *o1->mutex = 0;
    o1->min_x = node->min_x;
    o1->max_x = node->max_x - scalar;
    o1->min_y = node->min_y + scalar;
    o1->max_y = node->max_y;
    o1->min_z = node->min_z + scalar;
    o1->max_z = node->max_z;
	o1->x = 0;
	o1->y = 0;
	o1->z = 0;
	o1->mass = 0;

    o2->depth = depth;
    o2->body = -1;
    cudaMalloc((void**)&o2->mutex, sizeof(int));
    *o2->mutex = 0;
    o2->min_x = node->min_x;
    o2->max_x = node->max_x - scalar;
    o2->min_y = node->min_y;
    o2->max_y = node->max_y - scalar;
    o2->min_z = node->min_z + scalar;
    o2->max_z = node->max_z;
	o2->x = 0;
	o2->y = 0;
	o2->z = 0;
	o2->mass = 0;

	o3->depth = depth;
	o3->body = -1;
    cudaMalloc((void**)&o3->mutex, sizeof(int));
    *o3->mutex = 0;
	o3->min_x = node->min_x + scalar;
	o3->max_x = node->max_x;
	o3->min_y = node->min_y;
	o3->max_y = node->max_y - scalar;
	o3->min_z = node->min_z + scalar;
	o3->max_z = node->max_z;
	o3->x = 0;
	o3->y = 0;
	o3->z = 0;
	o3->mass = 0;

	o4->depth = depth;
	o4->body = -1;
    cudaMalloc((void**)&o4->mutex, sizeof(int));
    *o4->mutex = 0;
	o4->min_x = node->min_x + scalar;
	o4->max_x = node->max_x;
	o4->min_y = node->min_y + scalar;
	o4->max_y = node->max_y;
	o4->min_z = node->min_z;
	o4->max_z = node->max_z - scalar;
	o4->x = 0;
	o4->y = 0;
	o4->z = 0;
	o4->mass = 0;

	o5->depth = depth;
	o5->body = -1;
    cudaMalloc((void**)&o5->mutex, sizeof(int));
    *o5->mutex = 0;
	o5->min_x = node->min_x;
	o5->max_x = node->max_x - scalar;
	o5->min_y = node->min_y + scalar;
	o5->max_y = node->max_y;
	o5->min_z = node->min_z;
	o5->max_z = node->max_z - scalar;
	o5->x = 0;
	o5->y = 0;
	o5->z = 0;
	o5->mass = 0;

	o6->depth = depth;
	o6->body = -1;
    cudaMalloc((void**)&o6->mutex, sizeof(int));
    *o6->mutex = 0;
	o6->min_x = node->min_x;
	o6->max_x = node->max_x - scalar;
	o6->min_y = node->min_y;
	o6->max_y = node->max_y - scalar;
	o6->min_z = node->min_z;
	o6->max_z = node->max_z - scalar;
	o6->x = 0;
	o6->y = 0;
	o6->z = 0;
	o6->mass = 0;

	o7->depth = depth;
	o7->body = -1;
    cudaMalloc((void**)&o7->mutex, sizeof(int));
    *o7->mutex = 0;
	o7->min_x = node->min_x + scalar;
	o7->max_x = node->max_x;
	o7->min_y = node->min_y;
	o7->max_y = node->max_y - scalar;
	o7->min_z = node->min_z;
	o7->max_z = node->max_z - scalar;
	o7->x = 0;
	o7->y = 0;
	o7->z = 0;
	o7->mass = 0;

    node->o0 = o0;
    node->o1 = o1;
    node->o2 = o2;
    node->o3 = o3;
    node->o4 = o4;
    node->o5 = o5;
    node->o6 = o6;
    node->o7 = o7;
}


__device__ bnode_cuda* get_octant_cuda(bnode_cuda* node, double x, double y, double z){
	int scalar = fabsf(node->max_x - node->min_x)/2;
    bnode_cuda* result;
    // printf("thread %d get octant\n", (blockIdx.x * blockDim.x) + threadIdx.x);
    // print_node_cuda(node);
	if(node->min_x + scalar <= x && x <= node->max_x && node->min_y + scalar <= y && y <= node->max_y && node->min_z + scalar <= z && z <= node->max_z){
        // printf("thread %d returning o0..\n", (blockIdx.x * blockDim.x) + threadIdx.x);
        result = node->o0;
    }
    if(node->min_x <= x && x <= node->max_x - scalar && node->min_y + scalar <= y && y <= node->max_y && node->min_z + scalar <= z && z <= node->max_z){
        // printf("thread %d returning o1..\n", (blockIdx.x * blockDim.x) + threadIdx.x);
        result = node->o1;
    }
    if(node->min_x <= x && x <= node->max_x - scalar && node->min_y <= y && y <= node->max_y - scalar && node->min_z + scalar <= z && z <= node->max_z){
        // printf("thread %d returning o2..\n", (blockIdx.x * blockDim.x) + threadIdx.x);
        result = node->o2;
    }
    if(node->min_x + scalar <= x && x <= node->max_x && node->min_y <= y && y <= node->max_y - scalar && node->min_z + scalar <= z && z <= node->max_z){
        // printf("thread %d returning o3..\n", (blockIdx.x * blockDim.x) + threadIdx.x);
        result = node->o3;
    }
    if(node->min_x + scalar <= x && x <= node->max_x && node->min_y + scalar <= y && y <= node->max_y && node->min_z <= z && z <= node->max_z - scalar){
        // printf("thread %d returning o4..\n", (blockIdx.x * blockDim.x) + threadIdx.x);
        result = node->o4;
    }
    if(node->min_x <= x && x <= node->max_x - scalar && node->min_y + scalar <= y && y <= node->max_y && node->min_z <= z && z <= node->max_z - scalar){
        // printf("thread %d returning o5..\n", (blockIdx.x * blockDim.x) + threadIdx.x);
        result = node->o5;
    }
    if(node->min_x <= x && x <= node->max_x - scalar && node->min_y <= y && y <= node->max_y - scalar && node->min_z <= z && z <= node->max_z - scalar){
        // printf("thread %d returning o6..\n", (blockIdx.x * blockDim.x) + threadIdx.x);
        result = node->o6;
    }
    if(node->min_x + scalar <= x && x <= node->max_x && node->min_y <= y && y <= node->max_y - scalar && node->min_z <= z && z <= node->max_z - scalar){
		// printf("thread %d returning o7..\n", (blockIdx.x * blockDim.x) + threadIdx.x);
        result = node->o7;
    }
    return result;
}


__global__ void insert_body_cuda(bnode_cuda* node){
    int body = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(body >= n_d) return;
    double bx = x_d[body], by = y_d[body], bz = z_d[body], bmass = mass_d[body];

    bnode_cuda *current = node;
    int done = 0;

    // printf("THREAD %d started\n", body);

    while (*semaphore > 0){
        // printf("THREAD: %d SEM VALUE: %d\n", body, *semaphore);
        if(done == 1 && current->body == -2){
            done = 0;
            atomicAdd(semaphore, 1);
            // printf("%p\n", current);
            current = get_octant_cuda(current, bx, by, bz);
            // printf("%p\n", current);
        }
        if( done == 1 && current->body >= 0){
            continue;
        }
        //printf("THREAD %d trying to acquire lock NODE LOCK STATUS: %d SEM VALUE: %d\n", body, *current->mutex, *semaphore);
        if (atomicCAS(current->mutex, 0, 1) == 0){
            // printf("THREAD %d acquired lock NODE LOCK STATUS start: %d SEM VALUE: %d\n", body, *current->mutex, *semaphore);
            if(current->body >= 0){
                generate_empty_children_cuda(current);
                update_cuda(current, body, bx, by, bz, bmass);
                *current->mutex = 0;
                current = get_octant_cuda(current, bx, by, bz);
                // printf("THREAD %d found a full leaf NODE LOCK STATUS end: %d SEM VALUE: %d\n", body, *current->mutex, *semaphore);
                continue;
            }
            if(current->body == -2){
                update_cuda(current, body, bx, by, bz, bmass);
                *current->mutex = 0;
                current = get_octant_cuda(current, bx, by, bz);
                // printf("THREAD %d found an internal node NODE LOCK STATUS end: %d SEM VALUE: %d\n", body, *current->mutex, *semaphore);
                continue;
            }
            if(current->body == -1){
                update_cuda(current, body, bx, by, bz, bmass);

                atomicSub(semaphore, 1);

                done = 1;
                *semaphore_mutex = 0;
                *current->mutex = 0;
                // printf("THREAD %d found an empty leaf NODE LOCK STATUS end: %d SEM VALUE: %d\n", body, *current->mutex, *semaphore);
                continue;
            }
        }


    }

    cudaDeviceSynchronize();

}

__device__ void update_cuda(bnode_cuda *node, int body, double body_x, double body_y, double body_z, double body_mass){
    if(node->body >= 0){
        node->body = -2;
    }
    if(node->body == -1){
        node->body = body;
    }
    double c_mass = node->mass + body_mass;
	double c_x = ((node->mass*node->x)+(body_mass*body_x))/c_mass;
	double c_y = ((node->mass*node->y)+(body_mass*body_y))/c_mass;
	double c_z = ((node->mass*node->z)+(body_mass*body_z))/c_mass;
	node->mass = c_mass;
	node->x = c_x;
	node->y = c_y;
	node->z = c_z;
}


__global__ void set_pos(float4 *pos){
    int body = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(body >= n_d) return;
    pos[body] = make_float4(x_d[body], y_d[body], z_d[body], 1);
}


__global__ void compute_barnes_forces_cuda(float4 *pos){

    

    dim3 grid(ceil(n_d/512.0f), 1, 1);
    
    int blockdim = 512;

    if (n_d < 512){
        blockdim = n_d;
    }
    
    dim3 block(blockdim, 1, 1);

    set_pos<<<grid, block>>>(pos);
    cudaDeviceSynchronize();

    // compute bounding box
    double *max_x, *max_y, *max_z;
    cudaMalloc((void**)&max_x, sizeof(double));
    cudaMalloc((void**)&max_y, sizeof(double));
    cudaMalloc((void**)&max_z, sizeof(double));

    get_max_x_cuda<<<grid, block, n_d*sizeof(double)>>>(max_x);
    cudaDeviceSynchronize();
    get_max_y_cuda<<<grid, block, n_d*sizeof(double)>>>(max_y);
    cudaDeviceSynchronize();
    get_max_z_cuda<<<grid, block, n_d*sizeof(double)>>>(max_z);

    cudaDeviceSynchronize();    

    double max = 0;
    if (*max_x > *max_y) {
        if (*max_x > *max_z){
            max = *max_x;
        } else {
            max = *max_z;
        }
    } else {
        if (*max_y > *max_z) {
            max = *max_y;
        } else {
            max = *max_z;
        }
    }

    // build barnes root
    bnode_cuda* root;
	cudaMalloc((void**)&root, sizeof(bnode_cuda));

    root->body = -1;
    root->depth = 0;
    cudaMalloc((void**)&root->mutex, sizeof(int));
    *root->mutex = 0;
    root->max_x = max;
    root->max_y = max;
    root->max_z = max;
    root->min_x = -max;
    root->min_y = -max;
    root->min_z = -max;
    root->x = 0;
    root->y = 0;
    root->z = 0;
    root->mass = 0;

    cudaMalloc((void**)&semaphore, sizeof(int));
    *semaphore = n_d;
    cudaMalloc((void**)&semaphore_mutex, sizeof(int));
    *semaphore_mutex = 0;

    // insert_body_cuda<<<grid, block>>>(root);

    // compute_force_cuda<<<grid, block>>>(root, 0.5, pos);
    // cudaDeviceSynchronize();


    cudaFree(max_x);
    cudaFree(max_y);
    cudaFree(max_z);

    destruct_tree(root);

    cudaFree(semaphore);
    cudaFree(semaphore_mutex);
    
    
}

__global__ void compute_force_cuda(bnode_cuda* root, double theta, float4 *pos){
        int body = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(body >= n_d) return;
        queue_cuda* queue = create_queue_cuda(n_d);
        enqueue_cuda(queue, root);

        while(queue->size > 0){
            bnode_cuda* current = dequeue_cuda(queue);
            if(current->body == body || current->body == -1) continue;
            double ratio = fabsf(current->max_x - current->min_x);
	        double line_distance = sqrt(pow(x_d[body] - current->x,2) + pow(y_d[body] - current->y,2) + pow(z_d[body] - current->z,2));;

            if(ratio/line_distance < theta || current->body >= 0){
                double acc[3] = {0, 0, 0};
		        double force[3] = {0, 0, 0};
		        double distance[3] = {x_d[body] - current->x, y_d[body] - current->y, z_d[body] - current->z};
		        double dist = sqrt(pow(x_d[body] - current->x,2) + pow(y_d[body] - current->y,2) + pow(z_d[body] - current->z,2));
                double unit_vector[3] = {distance[0]/fabsf(distance[0]), distance[1]/fabsf(distance[1]), distance[2]/fabsf(distance[2])};

                if(distance[0] == 0){
		            unit_vector[0] = 0;
	            }
	            if(distance[1] == 0){
		            unit_vector[1] = 0;
	            }
	            if(distance[2] == 0){
		            unit_vector[2] = 0;
    	        }

		        force[0] = -G*((current->mass*mass_d[body]/pow(dist, 2)))*unit_vector[0];
		        force[1] = -G*((current->mass*mass_d[body]/pow(dist, 2)))*unit_vector[1];
		        force[2] = -G*((current->mass*mass_d[body]/pow(dist, 2)))*unit_vector[2];

		        acc[0] = force[0]/mass_d[body];
		        acc[1] = force[1]/mass_d[body];
		        acc[2] = force[2]/mass_d[body];

                new_x_d[body] += sx_d[body]*dt + (acc[0])*dt*dt*0.5;
		        new_y_d[body] += sy_d[body]*dt + (acc[1])*dt*dt*0.5;
		        new_z_d[body] += sz_d[body]*dt + (acc[2])*dt*dt*0.5;

                double x_res = x_d[body] + (sx_d[body]*dt + (acc[0])*dt*dt*0.5);
	            double y_res = y_d[body] + (sy_d[body]*dt + (acc[1])*dt*dt*0.5);
	            double z_res = z_d[body] + (sz_d[body]*dt + (acc[2])*dt*dt*0.5);

		        double new_acc[3] = {0, 0, 0};
		        double new_force[3] = {0, 0, 0};
		        double new_distance[3] = {new_x_d[body] - current->x, new_y_d[body] - current->y, new_z_d[body] - current->z};
		        double new_dist = sqrt(pow(new_x_d[body] - current->x,2) + pow(new_y_d[body] - current->y,2) + pow(new_z_d[body] - current->z,2));
                double new_unit_vector[3] = {new_distance[0]/fabsf(new_distance[0]), new_distance[1]/fabsf(new_distance[1]), new_distance[2]/fabsf(new_distance[2])};

                if(new_distance[0] == 0){
		            new_unit_vector[0] = 0;
	            }
	            if(new_distance[1] == 0){
	            	new_unit_vector[1] = 0;
	            }
	            if(new_distance[2] == 0){
		            new_unit_vector[2] = 0;
	            }

		        new_force[0] = -G*((current->mass*mass_d[body]/pow(dist, 2)))*new_unit_vector[0];
		        new_force[1] = -G*((current->mass*mass_d[body]/pow(dist, 2)))*new_unit_vector[1];
		        new_force[2] = -G*((current->mass*mass_d[body]/pow(dist, 2)))*new_unit_vector[2];

		        new_acc[0] = new_force[0]/mass_d[body];
		        new_acc[1] = new_force[1]/mass_d[body];
		        new_acc[2] = new_force[2]/mass_d[body];

		        new_sx_d[body] += 0.5*(acc[0] + new_acc[0])*dt;
		        new_sy_d[body] += 0.5*(acc[1] + new_acc[1])*dt;
		        new_sz_d[body] += 0.5*(acc[2] + new_acc[2])*dt;

                pos[body] = make_float4(x_res, y_res, z_res, 0);

            } else {
                enqueue_cuda(queue, current->o0);
                enqueue_cuda(queue, current->o1);
                enqueue_cuda(queue, current->o2);
                enqueue_cuda(queue, current->o3);
                enqueue_cuda(queue, current->o4);
                enqueue_cuda(queue, current->o5);
                enqueue_cuda(queue, current->o6);
                enqueue_cuda(queue, current->o7);
            }
        }

        cudaDeviceSynchronize();

}
