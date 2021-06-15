#include <GL/glut.h>
#include "../sequential/compute-barneshut.h"

extern float angle;
extern float lxcam, lzcam, lycam;
extern float xcam,zcam,ycam;
extern float deltaAngle;
extern int xOrigin;
extern int yOrigin;
extern float camera_speed;
extern float camera_x, camera_y, camera_z;

void processSpecialKeys(int key, int xx, int yy);
void mouseButton(int button, int state, int xcam, int ycam);
void mouseMove(int xcam, int ycam);
void reshape(GLint w, GLint h);
void timerfunc(int v);
void draw_axis();
void draw_body(int i);
void motion(int xcam, int ycam);
void display_tree(bnode* node);