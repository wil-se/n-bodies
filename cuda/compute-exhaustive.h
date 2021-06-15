#include <cuda.h>
#include <stdio.h>
#include "common.h"


// forza applicata al corpo 2 esercitata dal corpo 1
__device__ void compute_ex_force_cuda(float4 *pos, int body2, int body1){
	double acc[3] = {0, 0, 0};
	double force[3] = {0, 0, 0};
	double distance[3] = {x_d[body2] - x_d[body1], y_d[body2] - y_d[body1], z_d[body2] - z_d[body1]};
	double dist = sqrt(pow(x_d[body2] - x_d[body1],2) + pow(y_d[body2] - y_d[body1],2) + pow(z_d[body2] - z_d[body1],2));
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

	force[0] = -G*((mass_d[body1]*mass_d[body2]/pow(dist, 2)))*unit_vector[0];
	force[1] = -G*((mass_d[body1]*mass_d[body2]/pow(dist, 2)))*unit_vector[1];
	force[2] = -G*((mass_d[body1]*mass_d[body2]/pow(dist, 2)))*unit_vector[2];
	
	acc[0] = force[0]/mass_d[body2];
	acc[1] = force[1]/mass_d[body2];
	acc[2] = force[2]/mass_d[body2];
	
	new_x_d[body2] += sx_d[body2]*dt + (acc[0])*dt*dt*0.5;
	new_y_d[body2] += sy_d[body2]*dt + (acc[1])*dt*dt*0.5;
	new_z_d[body2] += sz_d[body2]*dt + (acc[2])*dt*dt*0.5;
		
	double new_acc[3] = {0, 0, 0};
	double new_force[3] = {0, 0, 0};
	double new_distance[3] = {new_x_d[body2] - x_d[body1], new_y_d[body2] - y_d[body1], new_z_d[body2] - z_d[body1]};
	double new_dist = sqrt(pow(new_x_d[body2] - x_d[body1],2) + pow(new_y_d[body2] - y_d[body1],2) + pow(new_z_d[body2] - z_d[body1],2));
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

	new_force[0] = -G*((mass_d[body1]*mass_d[body2]/pow(dist, 2)))*new_unit_vector[0];
	new_force[1] = -G*((mass_d[body1]*mass_d[body2]/pow(dist, 2)))*new_unit_vector[1];
	new_force[2] = -G*((mass_d[body1]*mass_d[body2]/pow(dist, 2)))*new_unit_vector[2];

	new_acc[0] = new_force[0]/mass_d[body2];
	new_acc[1] = new_force[1]/mass_d[body2];
	new_acc[2] = new_force[2]/mass_d[body2];
	
	new_sx_d[body2] += 0.5*(acc[0] + new_acc[0])*dt;
	new_sy_d[body2] += 0.5*(acc[1] + new_acc[1])*dt;
	new_sz_d[body2] += 0.5*(acc[2] + new_acc[2])*dt;

	pos[body2] = make_float4(new_x_d[body2], new_y_d[body2], new_z_d[body2], 10.0);
}

__global__ void compute_ex_forces_cuda(float4 *pos, unsigned int width){
	int body = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(body >= n_d) return;
	for(int k=0; k<n_d; k++){
		if(body != k){
			compute_ex_force_cuda(pos, body, k);
		}
	}
}
