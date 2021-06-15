#include "../opengl/render.h"
#include "../sequential/compute-exhaustive.h"
#include "../sequential/compute-barneshut.h"


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


void display_seq_ex() {
  // print_csv_bodies();
  compute_ex_forces();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  draw_axis();
  for(int i=0; i<n; i++){
      draw_body(i);
  }
  glFlush();
  glutSwapBuffers();
}

void display_seq_bh() {
  // print_csv_bodies();
  bnode* root;
	root = (bnode*)malloc(sizeof(bnode));
	build_barnes_tree(root);
  compute_barnes_forces_all(root, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  display_tree(root);
  for(int i=0; i<n; i++){
      draw_body(i);
  }
  // draw_axis();
  glFlush();
  glutSwapBuffers();
	destroy_barnes_tree(root);
}

void render_sequential_exhaustive(int argc, char** argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(1600, 900);
  glutCreateWindow("space");
  glutDisplayFunc(display_seq_ex);
  glutReshapeFunc(reshape);
	glutSpecialFunc(processSpecialKeys);
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);
  glutTimerFunc(1, timerfunc, 0);
  glEnable(GL_DEPTH_TEST);
  glutMainLoop();
}

void render_sequential_barneshut(int argc, char** argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(1600, 900);
  glutCreateWindow("space");
  glutDisplayFunc(display_seq_bh);
  glutReshapeFunc(reshape);
	glutSpecialFunc(processSpecialKeys);
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);
  glutTimerFunc(1, timerfunc, 0);
  glEnable(GL_DEPTH_TEST);
  glutMainLoop();
}