#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"

int n;
double *x, *y, *z, *new_x, *new_y, *new_z, *new_sx, *new_sy, *new_sz, *mass, *sx, *sy, *sz;

void print_csv_body(int i) {
    printf("%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", i, x[i], y[i], z[i], mass[i], sx[i], sy[i], sz[i]);
}

void print_csv_bodies(){
    for(int i=0; i<n; i++){
        print_csv_body(i);
    }
}

void set_memory(){
    scanf("%d", &n);
    y = (double*)malloc(sizeof(double)*n);
    x = (double*)malloc(sizeof(double)*n);
    z = (double*)malloc(sizeof(double)*n);
    mass = (double*)malloc(sizeof(double)*n);
    sx = (double*)malloc(sizeof(double)*n);
    sy = (double*)malloc(sizeof(double)*n);
    sz = (double*)malloc(sizeof(double)*n);
    for(int i=0; i<n; i++ ){
        scanf("%lf %lf %lf %lf %lf %lf %lf", &x[i], &y[i], &z[i], &mass[i], &sx[i], &sy[i], &sz[i]);
	}
}

void free_memory(){
    free(x);
    free(y);
    free(z);
    free(mass);
    free(sx);
    free(sy);
    free(sz);
}

void set_new_memory(){
	new_x = (double*)malloc(sizeof(double)*n);
	new_y = (double*)malloc(sizeof(double)*n);
	new_z = (double*)malloc(sizeof(double)*n);
	new_sx = (double*)malloc(sizeof(double)*n);
	new_sy = (double*)malloc(sizeof(double)*n);
	new_sz = (double*)malloc(sizeof(double)*n);
}

void free_new_memory(){
	free(new_x);
	free(new_y);
	free(new_z);
	free(new_sx);
	free(new_sy);
	free(new_sz);
}

void set_new_vectors(){
	for(int i=0; i<n; i++){
        new_x[i] = x[i];
        new_y[i] = y[i];
        new_z[i] = z[i];
        new_sx[i] = sx[i];
        new_sy[i] = sy[i];
        new_sz[i] = sz[i];
	}       
}

void set_vectors(){
	for(int i=0; i<n; i++){
        x[i] = new_x[i];
		y[i] = new_y[i];
        z[i] = new_z[i];
        sx[i] = new_sx[i];
        sy[i] = new_sy[i];
        sz[i] = new_sz[i];
	}       
}