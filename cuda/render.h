#include "compute-exhaustive.h"
#include "compute-barneshut.h"
#include "lib/exception.h"
#include "lib/helper_cuda.h"
#include "lib/helper_functions.h"
#include "lib/helper_gl.h"
#include "lib/helper_image.h"
#include "lib/helper_string.h"
#include "lib/helper_timer.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>




// OpenGL Graphics includes
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


#include <vector_types.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms




const unsigned int window_width  = 1536;
const unsigned int window_height = 1024;

const unsigned int mesh_width    = 256;
const unsigned int mesh_height   = 256;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

float g_fAnim = 0.0;


StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;



void launch_kernel_exhaustive(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time)
{
    dim3 grid(ceil(n_d/512.0f), 1, 1);
    dim3 block(512, 1, 1);
    compute_ex_forces_cuda<<<grid,block>>>(pos);
}



void launch_kernel_barneshut(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time)
{
    compute_barnes_forces_cuda<<<1,1>>>(pos);
    cudaDeviceSynchronize();
}


void run_exhaustive(struct cudaGraphicsResource **vbo_resource)
{
    float4 *dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, *vbo_resource));
    launch_kernel_exhaustive(dptr, mesh_width, mesh_height, g_fAnim);
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

void run_barneshut(struct cudaGraphicsResource **vbo_resource)
{
    float4 *dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, *vbo_resource));
    launch_kernel_barneshut(dptr, mesh_width, mesh_height, g_fAnim);
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);
        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}


void display_exhaustive()
{
    sdkStartTimer(&timer);
    run_exhaustive(&cuda_vbo_resource);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPointSize(2);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(xcam,ycam, zcam, xcam+lxcam,ycam+lycam,zcam+lzcam, 0.0f,1.0f,0.0f);
    glutPostRedisplay();
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 1.0, 0.0);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);
    glutSwapBuffers();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}

void display_barneshut()
{
    sdkStartTimer(&timer);
    run_barneshut(&cuda_vbo_resource);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPointSize(2);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(xcam,ycam, zcam, xcam+lxcam,ycam+lycam,zcam+lzcam, 0.0f,1.0f,0.0f);
    glutPostRedisplay();
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 1.0, 0.0);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);
    glutSwapBuffers();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
        // W
        case (119) :
            xcam += lxcam * camera_speed;
            zcam += lzcam * camera_speed;
            return;
        // A
        case (97) :
            xcam += lxcam * camera_speed;
			ycam += lycam * camera_speed;
            return;
        // S
        case (115) :
            xcam -= lxcam * camera_speed;
            zcam -= lzcam * camera_speed;
            return;
        // D
        case (100) :
            xcam -= lxcam * camera_speed;
			ycam -= lycam * camera_speed;
            return;
    }
}

void motion(int xcam, int ycam)
{
    if (xOrigin >= 0) {
    deltaAngle = (xcam - xOrigin) * 0.005f;
    lxcam = sin(angle - deltaAngle) * camera_speed;
    lzcam = -cos(angle - deltaAngle) * camera_speed;
  }
  if (yOrigin >= 0) {
    deltaAngle = (ycam - yOrigin) * 0.005f;
    lycam = tan(angle - deltaAngle) * camera_speed;
  }
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

void mouse(int button, int state, int xcam, int ycam)
{
    if (button == GLUT_LEFT_BUTTON) {
    if (state == GLUT_UP) {
      angle -= deltaAngle;
      xOrigin = -1;
      yOrigin = -1;
    }
    else {
      xOrigin = xcam;
      yOrigin = ycam;
    }
  }
}

void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}



void render_cuda_exhaustive(int argc, char** argv) {
  sdkCreateTimer(&timer);
  int devID = findCudaDevice(argc, (const char **)argv);
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(1600, 900);
  glutCreateWindow("space");
  glutTimerFunc(REFRESH_DELAY, timerEvent,0);
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glDisable(GL_DEPTH_TEST);
  glViewport(0, 0, window_width, window_height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 1000000.0);
  SDK_CHECK_ERROR_GL();
  glutDisplayFunc(display_exhaustive);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
  glutMainLoop();
}


void render_cuda_barneshut(int argc, char** argv) {
  sdkCreateTimer(&timer);
  int devID = findCudaDevice(argc, (const char **)argv);
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(1600, 900);
  glutCreateWindow("space");
  glutTimerFunc(REFRESH_DELAY, timerEvent,0);
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glDisable(GL_DEPTH_TEST);
  glViewport(0, 0, window_width, window_height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 1000000.0);
  SDK_CHECK_ERROR_GL();
  glutDisplayFunc(display_barneshut);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
  glutMainLoop();
}