#ifndef BARNES_STRUCT
#define BARNES_STRUCT

typedef struct bnode{
        int body; // >= 0 se contiene un corpo, -1 se contiene zero corpi, -2 se contiene più di un corpo
        int depth;
        int max_x, max_y, max_z, min_x, min_y, min_z;
        double x, y, z;
        double mass; 
        struct bnode *o0, *o1, *o2, *o3, *o4, *o5, *o6, *o7;
} bnode;

typedef struct queue_node{
        bnode* body;
        struct queue_node* previous;
} queue_node;

typedef struct queue{
        int size, max_size;
        struct queue_node* head;
        struct queue_node* tail;
} queue;

#endif


void print_node(bnode* node);
queue* create_queue(int max_size);
queue_node* create_queue_node(bnode* node);
void enqueue(queue* q, bnode* node);
bnode* dequeue(queue* q);
void destruct_queue(queue* q);
void print_tree(bnode* node);
int get_bound();
void generate_empty_children(bnode *node);
void destroy_barnes_tree(bnode* root);
bnode* get_octant(bnode* node, double x, double y, double z);
void update(bnode* node, int body, double x, double y, double z, double mass);
void insert_body(bnode* node, int body);
void build_barnes_tree(bnode* root);
void compute_barnes_forces(bnode* node, int body, double theta);
void compute_barnes_forces_all(bnode* root, double theta);
