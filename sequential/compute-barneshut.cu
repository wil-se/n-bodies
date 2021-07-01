#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"
#include "compute-barneshut.h"



void print_node(bnode* node){
	printf("================================\nBODY: %d\nDEPTH: %d\nMAX X: %d\nMAX Y: %d\nMAX Z: %d\nMIN X: %d\nMIN Y: %d\nMIN Z: %d\nX: %f\nY: %f\nZ: %f\nMASS: %f\n", node->body, node->depth, node->max_x, node->max_y, node->max_z, node->min_x, node->min_y, node->min_z, node->x, node->y, node->z, node->mass);
}

queue* create_queue(int max_size){
    queue* q;
    q = (queue*)malloc(sizeof(queue));
    q->max_size = max_size;
    q->size = 0;
    q->head = NULL;
    q->tail = NULL;
    return q;
}

queue_node* create_queue_node(bnode* node){
    queue_node* n;
    n = (queue_node*)malloc(sizeof(queue_node));
    n->body = node;
    n->previous = NULL;
    return n;
}

void enqueue(queue* q, bnode* node){
    queue_node* q_node = create_queue_node(node);
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

bnode* dequeue(queue* q){
    queue_node* head = q->head;
    bnode* node = head->body;
    q->head = head->previous;
    q->size--;
    free(head);
    return node;
}

void destruct_queue(queue* q){
    if(q->head == NULL){return;}
    queue_node* node = q->head;
    while(node->previous != NULL){
        queue_node* previous = node->previous;
        free(node);
        node = previous;
    }
    free(q);
}

void print_tree(bnode* node){
    queue* q = create_queue(1024);
    enqueue(q, node);
    while(q->size != 0){
        bnode* curr = dequeue(q);
        print_node(curr);
        if(curr->body == -2){
            enqueue(q, curr->o0);
            enqueue(q, curr->o1);
            enqueue(q, curr->o2);
            enqueue(q, curr->o3);
            enqueue(q, curr->o4);
            enqueue(q, curr->o5);
            enqueue(q, curr->o6);
            enqueue(q, curr->o7);
        }
    }
	destruct_queue(q);
}


int get_bound(){
	double max = 0;
    for(int i=0; i<n; i++){
		if(fabs(x[i]) > max){max = fabs(x[i]);}
        if(fabs(y[i]) > max){max = fabs(y[i]);}
        if(fabs(z[i]) > max){max = fabs(z[i]);}
    }
	return ((int)max%2==0)?(int)max+2:(int)max+3;
}

void generate_empty_children(bnode *node){
    int depth = node->depth+1;
	int scalar = fabs(node->max_x - node->min_x)/2;
    bnode *o0, *o1, *o2, *o3, *o4, *o5, *o6, *o7;

    o0 = (bnode*)malloc(sizeof(bnode));
    o1 = (bnode*)malloc(sizeof(bnode));
    o2 = (bnode*)malloc(sizeof(bnode));
    o3 = (bnode*)malloc(sizeof(bnode));
    o4 = (bnode*)malloc(sizeof(bnode));
    o5 = (bnode*)malloc(sizeof(bnode));
    o6 = (bnode*)malloc(sizeof(bnode));
    o7 = (bnode*)malloc(sizeof(bnode));

    o0->depth = depth;
    o0->body = -1;
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


void destroy_barnes_tree(bnode* root){
	if(root->body >= 0 || root->body == -1){
		free(root);
	} else {
		destroy_barnes_tree(root->o0);
		destroy_barnes_tree(root->o1);
		destroy_barnes_tree(root->o2);
		destroy_barnes_tree(root->o3);
		destroy_barnes_tree(root->o4);
		destroy_barnes_tree(root->o5);
		destroy_barnes_tree(root->o6);
		destroy_barnes_tree(root->o7);
		free(root);
	}
}

bnode* get_octant(bnode* node, double x, double y, double z){
	int scalar = fabs(node->max_x - node->min_x)/2;
    bnode* result;

	if(node->min_x + scalar <= x && x <= node->max_x && node->min_y + scalar <= y && y <= node->max_y && node->min_z + scalar <= z && z <= node->max_z){
        result = node->o0;
    }
    if(node->min_x <= x && x <= node->max_x - scalar && node->min_y + scalar <= y && y <= node->max_y && node->min_z + scalar <= z && z <= node->max_z){
        result = node->o1;
    }
    if(node->min_x <= x && x <= node->max_x - scalar && node->min_y <= y && y <= node->max_y - scalar && node->min_z + scalar <= z && z <= node->max_z){
        result = node->o2;
    }
    if(node->min_x + scalar <= x && x <= node->max_x && node->min_y <= y && y <= node->max_y - scalar && node->min_z + scalar <= z && z <= node->max_z){
        result = node->o3;
    }
    if(node->min_x + scalar <= x && x <= node->max_x && node->min_y + scalar <= y && y <= node->max_y && node->min_z <= z && z <= node->max_z - scalar){
        result = node->o4;
    }
    if(node->min_x <= x && x <= node->max_x - scalar && node->min_y + scalar <= y && y <= node->max_y && node->min_z <= z && z <= node->max_z - scalar){
        result = node->o5;
    }
    if(node->min_x <= x && x <= node->max_x - scalar && node->min_y <= y && y <= node->max_y - scalar && node->min_z <= z && z <= node->max_z - scalar){
        result = node->o6;
    }
    if(node->min_x + scalar <= x && x <= node->max_x && node->min_y <= y && y <= node->max_y - scalar && node->min_z <= z && z <= node->max_z - scalar){
		result = node->o7;
    }
    return result;
}

void update(bnode* node, int body, double x, double y, double z, double mass){
	if(node->body >= 0){
        node->body = -2;
    }
    if(node->body == -1){
        node->body = body;
    }
    double tmass = node->mass + mass;
    double tx = ((node->mass*node->x)+(mass*x))/tmass;
	double ty = ((node->mass*node->y)+(mass*y))/tmass;
    double tz = ((node->mass*node->z)+(mass*z))/tmass;
    node->mass = tmass;
    node->x = tx;
    node->y = ty;
    node->z = tz;
}

void insert_body(bnode* node, int body){
	double bx = x[body], by = y[body], bz = z[body], bmass=mass[body];
	if(node->body == -1){
		update(node, body, bx, by, bz, bmass);
		return;
	}
	if(node->body >= 0){
		int old_body = node->body;
		generate_empty_children(node);
		bnode* next = get_octant(node, bx, by, bz);
		bnode* old_next = get_octant(node, x[old_body], y[old_body], z[old_body]);
		update(node, body, bx, by, bz, bmass);
		insert_body(old_next, old_body);
		insert_body(next, body);
		return;
	}
	if(node->body == -2){
        	update(node, body, bx, by, bz, bmass);
        	bnode* next = get_octant(node, bx, by, bz);
        	insert_body(next, body);
        	return;
	}
}

void build_barnes_tree(bnode* root){
    int bound = get_bound();
	root->body = -1;
    root->depth = 0;
    root->max_x = bound;
    root->max_y = bound;
    root->max_z = bound;
    root->min_x = -bound;
    root->min_y = -bound;
    root->min_z = -bound;
    root->x = 0;
    root->y = 0;
    root->z = 0;
    root->mass = 0;
    generate_empty_children(root);
    for(int i=0; i<n; i++){
		insert_body(root, i);
    }
}


void compute_barnes_forces(bnode* node, int body, double theta){
	if(node->body == body || node->body == -1){return;}
	double ratio = fabs(node->max_x - node->min_x);
	double line_distance = sqrt(pow(x[body] - node->x,2) + pow(y[body] - node->y,2) + pow(z[body] - node->z,2));;
		
	if(ratio/line_distance < theta || node->body >= 0){
		double acc[3] = {0, 0, 0};
		double force[3] = {0, 0, 0};
		double distance[3] = {x[body] - node->x, y[body] - node->y, z[body] - node->z};
		double dist = sqrt(pow(x[body] - node->x,2) + pow(y[body] - node->y,2) + pow(z[body] - node->z,2));
        double unit_vector[3] = {distance[0]/fabs(distance[0]), distance[1]/fabs(distance[1]), distance[2]/fabs(distance[2])};
		
        if(distance[0] == 0){
		    unit_vector[0] = 0;
	    }
	    if(distance[1] == 0){
		    unit_vector[1] = 0;
	    }
	    if(distance[2] == 0){
		    unit_vector[2] = 0;
    	}

		force[0] = -G*((node->mass*mass[body]/pow(dist, 2)))*unit_vector[0];
		force[1] = -G*((node->mass*mass[body]/pow(dist, 2)))*unit_vector[1];
		force[2] = -G*((node->mass*mass[body]/pow(dist, 2)))*unit_vector[2];

		acc[0] = force[0]/mass[body];
		acc[1] = force[1]/mass[body];
		acc[2] = force[2]/mass[body];

		new_x[body] += sx[body]*dt + (acc[0])*dt*dt*0.5;
		new_y[body] += sy[body]*dt + (acc[1])*dt*dt*0.5;
		new_z[body] += sz[body]*dt + (acc[2])*dt*dt*0.5;

		double new_acc[3] = {0, 0, 0};
		double new_force[3] = {0, 0, 0};
		double new_distance[3] = {new_x[body] - node->x, new_y[body] - node->y, new_z[body] - node->z};
		double new_dist = sqrt(pow(new_x[body] - node->x,2) + pow(new_y[body] - node->y,2) + pow(new_z[body] - node->z,2));
        double new_unit_vector[3] = {new_distance[0]/fabs(new_distance[0]), new_distance[1]/fabs(new_distance[1]), new_distance[2]/fabs(new_distance[2])};

        if(new_distance[0] == 0){
		    new_unit_vector[0] = 0;
	    }
	    if(new_distance[1] == 0){
	    	new_unit_vector[1] = 0;
	    }
	    if(new_distance[2] == 0){
		    new_unit_vector[2] = 0;
	    }

		new_force[0] = -G*((node->mass*mass[body]/pow(dist, 2)))*new_unit_vector[0];
		new_force[1] = -G*((node->mass*mass[body]/pow(dist, 2)))*new_unit_vector[1];
		new_force[2] = -G*((node->mass*mass[body]/pow(dist, 2)))*new_unit_vector[2];

		new_acc[0] = new_force[0]/mass[body];
		new_acc[1] = new_force[1]/mass[body];
		new_acc[2] = new_force[2]/mass[body];

		new_sx[body] += 0.5*(acc[0] + new_acc[0])*dt;
		new_sy[body] += 0.5*(acc[1] + new_acc[1])*dt;
		new_sz[body] += 0.5*(acc[2] + new_acc[2])*dt;

		
	} else {
		compute_barnes_forces(node->o0, body, theta);
		compute_barnes_forces(node->o1, body, theta);
		compute_barnes_forces(node->o2, body, theta);
		compute_barnes_forces(node->o3, body, theta);
		compute_barnes_forces(node->o4, body, theta);
		compute_barnes_forces(node->o5, body, theta);	
		compute_barnes_forces(node->o6, body, theta);
		compute_barnes_forces(node->o7, body, theta);
	}
}

void compute_barnes_forces_all(bnode* root, double theta){
	set_new_memory();
	set_new_vectors();
	for(int i=0; i<n; i++){
		compute_barnes_forces(root, i, theta);
	}
	set_vectors();
	free_new_memory();
}
