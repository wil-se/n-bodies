CFLAGS = -lm -lglut -lGL -lGLU
CC = nvcc
CUFILES = main.cu sequential/compute-barneshut.cu sequential/common.cu opengl/render.cu cuda/common.cu

.PHONY: clean nbody test


clean:
	rm *.o

nbody: main.cu
	$(CC) -rdc=true $(CUFILES) -o main.o $(CFLAGS)

run: nbody
	./main.o