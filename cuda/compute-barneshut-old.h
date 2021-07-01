typedef struct bnode_cuda{
        int body; // >= 0 se contiene un corpo, -1 se contiene zero corpi, -2 se contiene più di un corpo
        int depth;
        int *mutex; // 0 free 1 locked
        int max_x, max_y, max_z, min_x, min_y, min_z;
        double x, y, z;
        double mass;
        struct bnode_cuda *o0, *o1, *o2, *o3, *o4, *o5, *o6, *o7;
} bnode_cuda;

typedef struct queue_node_cuda{
        bnode_cuda* body;
        struct queue_node_cuda* previous;
} queue_node_cuda;

typedef struct queue_cuda{
        int size, max_size;
        struct queue_node_cuda* head;
        struct queue_node_cuda* tail;        
} queue_cuda;

__global__ void get_max_x_cuda(double *result);
__global__ void get_max_y_cuda(double *result);
__global__ void get_max_z_cuda(double *result);
__global__ void compute_barnes_forces_cuda(float4 *pos);
__global__ void insert_body_cuda(bnode_cuda* root);
__global__ void compute_force_cuda(bnode_cuda* root, double theta, float4 *pos);
__global__ void set_pos(float4 *pos);

__device__ void generate_empty_children_cuda(bnode_cuda *node);
__device__ bnode_cuda* get_octant_cuda(bnode_cuda* node, double x, double y, double z);
__device__ void print_node_cuda(bnode_cuda* node);
__device__ void update_cuda(bnode_cuda *node, int body, double body_x, double body_y, double body_z, double body_mass);

__device__ void enqueue_cuda(queue_cuda* q, bnode_cuda* node);
__device__ bnode_cuda* dequeue_cuda(queue_cuda* q);
__device__ queue_cuda* create_queue_cuda(int max_size);
__device__ void destruct_queue_cuda(queue_cuda* q);
__device__ queue_node_cuda* create_queue_node_cuda(bnode_cuda* node);
__device__ void print_tree_cuda(bnode_cuda* node);
__device__ void destruct_tree_cuda(bnode_cuda* root);
