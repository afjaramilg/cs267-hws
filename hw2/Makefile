SHELL='bash'
#
# Bridges - PSC
#
# Intel Compilers are loaded by default
# You will need to specifically switch to GNU Modules
# With with `modules.sh`
#

CC = g++
MPCC = mpic++
OPENMP = -fopenmp
CFLAGS = -O3
LIBS =


TARGETS = serial pthreads openmp autograder my_serial1 my_serial2 my_openmp

all:	$(TARGETS)


# my solutions
my_openmp: my_openmp.o common.o dcomp2d1.o
	$(CC) -o $@ $(LIBS) -fopenmp my_openmp.o dcomp2d1.o common.o

my_serial2: my_serial2.o common.o dcomp2d1.o
	$(CC) -o $@ $(LIBS) -fopenmp my_serial2.o dcomp2d1.o common.o

my_serial1: my_serial1.o common.o dcomp2d1.o
	$(CC) -o $@ $(LIBS) -fopenmp my_serial1.o dcomp2d1.o common.o
# -------------------

serial: serial.o common.o
	$(CC) -o $@ $(LIBS) serial.o common.o

autograder: autograder.o common.o
	$(CC) -o $@ $(LIBS) autograder.o common.o

pthreads: pthreads.o common.o
	$(CC) -o $@ $(LIBS) -lpthread pthreads.o common.o

openmp: openmp.o common.o
	$(CC) -o $@ $(LIBS) $(OPENMP) openmp.o common.o

mpi: mpi.o common.o
	$(MPCC) -o $@ $(LIBS) $(MPILIBS) mpi.o common.o


# my solutions
my_openmp.o: my_openmp.cpp
	$(CC) -c $(CFLAGS) my_openmp.cpp

my_serial2.o: my_serial2.cpp
	$(CC) -c $(CFLAGS) my_serial2.cpp

my_serial1.o: my_serial1.cpp
	$(CC) -c $(CFLAGS) my_serial1.cpp

dcomp2d1.o: dcomp2d1.h dcomp2d1.cpp
	$(CC) -c $(CFLAGS) dcomp2d1.cpp
# -------------------


autograder.o: autograder.cpp common.h
	$(CC) -c $(CFLAGS) autograder.cpp

openmp.o: openmp.cpp common.h
	$(CC) -c $(OPENMP) $(CFLAGS) openmp.cpp

serial.o: serial.cpp common.h
	$(CC) -c $(CFLAGS) serial.cpp

pthreads.o: pthreads.cpp common.h
	$(CC) -c $(CFLAGS) pthreads.cpp

mpi.o: mpi.cpp common.h
	$(MPCC) -c $(CFLAGS) mpi.cpp

common.o: common.cpp common.h
	$(CC) -c $(CFLAGS) common.cpp


clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
