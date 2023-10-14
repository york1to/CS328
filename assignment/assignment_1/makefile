EXECS=mpi_matrix_fox
MPICC?=mpicc

all: ${EXECS}

mpi_matrix_fox: mpi_matrix_fox.c
	${MPICC} -o mpi_matrix_fox mpi_matrix_fox.c -lm

clean:
	rm -f ${EXECS}