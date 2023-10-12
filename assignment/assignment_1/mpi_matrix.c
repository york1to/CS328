#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define MAT_SIZE 500

typedef struct Matrix{
    size_t cols;
    size_t rows;
    double* datap;
}Mat;

void brute_force_matmul(Mat* mat1, Mat* mat2, Mat* mat3) {
    //there should be a row col check but I ignored it cause we're not doing a library
    for (int i = 0; i < MAT_SIZE; ++i) {
        for (int j = 0; j < MAT_SIZE; ++j) {
            mat3->datap[i+j*MAT_SIZE] = 0;
            for (int k = 0; k < MAT_SIZE; ++k) {
                mat3->datap[i+j*MAT_SIZE] += mat1->datap[i*k*MAT_SIZE] * mat2->datap[k*j*MAT_SIZE];
            }
        }
    }
}

int main(int argc, char *argv[])
{
   int rank, mpiSize;
   Mat a = { .cols = MAT_SIZE, .rows = MAT_SIZE, .datap=(double*)malloc(MAT_SIZE*MAT_SIZE*sizeof(double))};
   Mat b = { .cols = MAT_SIZE, .rows = MAT_SIZE, .datap=(double*)malloc(MAT_SIZE*MAT_SIZE*sizeof(double))};
   Mat result = { .cols = MAT_SIZE, .rows = MAT_SIZE, .datap=(double*)malloc(MAT_SIZE*MAT_SIZE*sizeof(double))};
   Mat local_a = { .cols = 1, .rows = MAT_SIZE, .datap=(double*)malloc(MAT_SIZE*MAT_SIZE*sizeof(double))};
   Mat local_b = { .cols = MAT_SIZE, .rows = 1, .datap=(double*)malloc(MAT_SIZE*MAT_SIZE*sizeof(double))};

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
   double start;
   double finish;

   if (rank == 0)
   {
      for (int i = 0; i < MAT_SIZE; i++)
         for (int j = 0; j < MAT_SIZE; j++)
            a->datap[i*j*MAT_SIZE] = i + j;

      for (int i = 0; i < MAT_SIZE; i++)
         for (int j = 0; j < MAT_SIZE; j++)
            b->datap[i*j*MAT_SIZE] = i * j;

      start = MPI_Wtime();
      MPI_Scatter(a, MAT_SIZE*MAT_SIZE/mpiSize, MPI_DOUBLE, local_a, MAT_SIZE*MAT_SIZE/mpiSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(b, MAT_SIZE*MAT_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   }
   else
   {
      MPI_Scatter(a, MAT_SIZE*MAT_SIZE/mpiSize, MPI_DOUBLE, local_a, MAT_SIZE*MAT_SIZE/mpiSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(b, MAT_SIZE*MAT_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   }

   // Each process computes its local result
   brute_force_matmul(local_a, b, local_c);

   if (rank == 0)
   {
      MPI_Gather(local_c, MAT_SIZE*MAT_SIZE/mpiSize, MPI_DOUBLE, c, MAT_SIZE*MAT_SIZE/mpiSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      finish = MPI_Wtime();
      printf("Done in %f seconds.\n", finish - start);

      brute_force_matmul(a, b, bfRes);
      // Here, you can compare c and bfRes for correctness
   }
   else
   {
      MPI_Gather(local_c, MAT_SIZE*MAT_SIZE/mpiSize, MPI_DOUBLE, c, MAT_SIZE*MAT_SIZE/mpiSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   }

   MPI_Finalize();
   return 0;
}
