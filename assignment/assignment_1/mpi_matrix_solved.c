/******************************************************************************
 * FILE: mpi_mm.c
 * DESCRIPTION:
 *   MPI Matrix Multiply - C Version
 *   In this code, the master task distributes a matrix multiply
 *   operation to numtasks-1 worker tasks.
 *   NOTE:  C and Fortran versions of this code differ because of the way
 *   arrays are stored/passed.  C arrays are row-major order but Fortran
 *   arrays are column-major order.
 * AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
 *   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
 * LAST REVISED: 04/13/05
 ******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAT_SIZE 500
#define m MAT_SIZE   /* number of rows in matrix A */
#define n MAT_SIZE   /* number of columns in matrix A */
#define p MAT_SIZE   /* number of columns in matrix B */
#define MASTER 0      /* rank of first task */
#define FROM_MASTER 1 /* setting a message type */
#define FROM_SLAVE 2 /* setting a message type */

void brute_force_matmul(double** mat1, double** mat2, double** res)
{
  /* matrix multiplication of mat1 and mat2, store the result in res */
  for (int i = 0; i < MAT_SIZE; ++i)
  {
    for (int j = 0; j < MAT_SIZE; ++j)
    {
      res[i][j] = 0;
      for (int k = 0; k < MAT_SIZE; ++k)
      {
        res[i][j] += mat1[i][k] * mat2[k][j];
      }
    }
  }
}

int main(int argc, char *argv[])
{
  int numtasks,              /* number of tasks in partition */
      rank,                /* a task identifier */
      numworkers,            /* number of worker tasks */
      source,                /* task id of message source */
      dest,                  /* task id of message destination */
      mtype,                 /* message type */
      rows,                  /* rows of matrix A sent to each worker */
      averow, extra, offset, /* used to determine rows sent to each worker */
      i, j, k, rc;           /* misc */
  double ** a;
  double ** b;
  double ** c;
  a = (double **)malloc(m * sizeof(double *));
  b = (double **)malloc(n * sizeof(double *));
  c = (double **)malloc(m * sizeof(double *));
  double *ap = malloc(m* n * sizeof(double));
  double *bp = malloc(n* p * sizeof(double));
  double *cp = malloc(m* p * sizeof(double));
  for (int i = 0; i < m; i++)
    a[i] = &ap[i * n];
  for (int i = 0; i < n; i++)
    b[i] = &bp[i * p];
  for (int i = 0; i < m; i++)
    c[i] = &cp[i * p];

  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  if (numtasks < 2)
  {
    printf("Need at least two MPI tasks. Quitting...\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
  }
  numworkers = numtasks - 1;

  if (rank == MASTER)
  {
    /* master */

    printf("mpi_mm has started with %d tasks.\n", numtasks);

    /* First, fill some numbers into the matrix */
    printf("Initializing arrays...\n");
    for (i = 0; i < m; i++)
      for (j = 0; j < n; j++)
        a[i][j] = i + j;
    for (i = 0; i < n; i++)
      for (j = 0; j < p; j++)
        b[i][j] = i * j;

    /* Measure start time */
    double start = MPI_Wtime();

    /* Send matrix data to the worker tasks */
    averow = m / numworkers;
    extra = m % numworkers;
    offset = 0;
    mtype = FROM_MASTER;

    for (dest = 1; dest <= numworkers; dest++)
    {
      /* If the destination index is Smaller than the residue give one more row to this task */
      
      rows = (dest <= extra) ? averow + 1 : averow;
      printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
      MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      MPI_Send(&a[offset][0], rows * n, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
      MPI_Send(&b[0][0], n * p, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
      // for (int i = 0; i < n; i++)
      // {
      //   for (int j = 0; j < p; j++)
      //   {
      //     printf("before b[%d][%d] = %f\n", i, j, b[i][j]);
      //   }
      // }
      offset = offset + rows;
    }

    /* Receive results from worker tasks */
    mtype = FROM_SLAVE;
    for (i = 1; i <= numworkers; i++)
    {
      source = i;
      MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&c[offset][0], rows * p, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
      // printf("Received results from task %d\n",source);
    }

    /* Print results */
    // printf("******************************************************\n");
    // printf("Result Matrix:\n");
    // for (i=0; i<m; i++)
    // {
    //    printf("\n");
    //    for (j=0; j<p; j++)
    //       printf("%6.2f   ", c[i][j]);
    // }
    // printf("\n******************************************************\n");

    /* Measure finish time */
    double finish = MPI_Wtime();
    printf("Done in %f seconds.\n", finish - start);

    /* Compare the result with Brutal force */
    double **d = (double **)malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++)
      d[i] = (double *)malloc(p * sizeof(double));
    double begin = clock();
    brute_force_matmul(a, b, d);
    double end = clock();
    printf("Brutal force done in %f seconds.\n", (end - begin) / CLOCKS_PER_SEC);

    /* Check the result of the two methods */
    for (int i = 0; i < m; i++)
      for (int j = 0; j < p; j++)
        if (c[i][j] != d[i][j])
        {
          printf("Error: c[%d][%d] != d[%d][%d]\n", i, j, i, j);
          exit(1);
        }

    for (int i = 0; i < m; i++)
      free(d[i]);
    free(d);
  }

  /**************************** worker task ************************************/
  if (rank > MASTER)
  {
    for (int i = 0; i < n; i++)
      b[i] = (double *)malloc(p * sizeof(double));
    mtype = FROM_MASTER;
    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&a[0][0], rows * n, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&b[0][0], n * p, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
    // for(int i = 0; i< n; i++)
    // {
    //   for (int j = 0; j < p; j++)
    //   {
    //     printf("b[%d][%d] = %f\n", i, j, b[i][j]);
    //   }
    // }


    for (k = 0; k < p; k++)
      for (i = 0; i < rows; i++)
      {
        c[i][k] = 0.0;
        for (j = 0; j < n; j++)
          c[i][k] = c[i][k] + a[i][j] * b[j][k];
      }
    mtype = FROM_SLAVE;
    MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&c[0][0], rows * p, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  }
  MPI_Finalize();

  /* Free */
  free(ap);
  free(bp);
  free(cp);
  free(a);
  free(b);
  free(c);
}