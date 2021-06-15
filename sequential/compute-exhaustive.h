#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cassert>
#include "common.h"


// forza applicata al corpo 2 esercitata dal corpo 1
void compute_ex_force(int body2, int body1){
	double acc[3] = {0, 0, 0};
	double force[3] = {0, 0, 0};
	double distance[3] = {x[body2] - x[body1], y[body2] - y[body1], z[body2] - z[body1]};
	double dist = sqrt(pow(x[body2] - x[body1],2) + pow(y[body2] - y[body1],2) + pow(z[body2] - z[body1],2));
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

	force[0] = -G*((mass[body1]*mass[body2]/pow(dist, 2)))*unit_vector[0];
	force[1] = -G*((mass[body1]*mass[body2]/pow(dist, 2)))*unit_vector[1];
	force[2] = -G*((mass[body1]*mass[body2]/pow(dist, 2)))*unit_vector[2];
	
	acc[0] = force[0]/mass[body2];
	acc[1] = force[1]/mass[body2];
	acc[2] = force[2]/mass[body2];
	
	new_x[body2] += sx[body2]*dt + (acc[0])*dt*dt*0.5;
	new_y[body2] += sy[body2]*dt + (acc[1])*dt*dt*0.5;
	new_z[body2] += sz[body2]*dt + (acc[2])*dt*dt*0.5;
		
	double new_acc[3] = {0, 0, 0};
	double new_force[3] = {0, 0, 0};
	double new_distance[3] = {new_x[body2] - x[body1], new_y[body2] - y[body1], new_z[body2] - z[body1]};
	double new_dist = sqrt(pow(new_x[body2] - x[body1],2) + pow(new_y[body2] - y[body1],2) + pow(new_z[body2] - z[body1],2));
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

	new_force[0] = -G*((mass[body1]*mass[body2]/pow(dist, 2)))*new_unit_vector[0];
	new_force[1] = -G*((mass[body1]*mass[body2]/pow(dist, 2)))*new_unit_vector[1];
	new_force[2] = -G*((mass[body1]*mass[body2]/pow(dist, 2)))*new_unit_vector[2];

	new_acc[0] = new_force[0]/mass[body2];
	new_acc[1] = new_force[1]/mass[body2];
	new_acc[2] = new_force[2]/mass[body2];
	
	new_sx[body2] += 0.5*(acc[0] + new_acc[0])*dt;
	new_sy[body2] += 0.5*(acc[1] + new_acc[1])*dt;
	new_sz[body2] += 0.5*(acc[2] + new_acc[2])*dt;
}



void compute_ex_forces(){
	set_new_memory();	
	set_new_vectors();
	for(int j=0; j<n; j++){
		for(int k=0; k<n; k++){
			if(j != k){
				compute_ex_force(j, k);
			}
		}
	}
	set_vectors();
	free_new_memory();
}
