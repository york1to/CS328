EXECS=mpi_matrix_solved mpi_matrix_fox mpi_matrix_brute
MPICC?=mpicc

all: ${EXECS}

mpi_matrix_fox: mpi_matrix_fox.c
	${MPICC} -o mpi_matrix_fox mpi_matrix_fox.c -lm

mpi_matrix_solved: mpi_matrix_solved.c
	${MPICC} -o mpi_matrix_solved mpi_matrix_solved.c -lm

mpi_matrix_brute: mpi_matrix_brute.c
	${MPICC} -o mpi_matrix_brute mpi_matrix_brute.c -lm

clean:
	rm -f ${EXECS}