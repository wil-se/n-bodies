CFLAGS = -lm -lglut -lGL -lGLU
CC = nvcc
DEBUGGER = cuda-gdb
CUFILES = main.cu sequential/compute-barneshut.cu sequential/common.cu opengl/render.cu cuda/common.cu cuda/compute-barneshut.cu

.PHONY: clean nbody test


clean:
	rm *.o

nbody: main.cu
	$(CC) -rdc=true $(CUFILES) -o main.o $(CFLAGS)
debug: main.cu
	$(CC) -rdc=true -g -G $(CUFILES) -o main.o $(CFLAGS)

run: nbody
	./main.o
