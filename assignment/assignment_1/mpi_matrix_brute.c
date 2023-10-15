#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAT_SIZE 480  /* Size of matrices */
#define m MAT_SIZE    /* number of rows in matrix A */
#define n MAT_SIZE    /* number of columns in matrix A */
#define p MAT_SIZE    /* number of columns in matrix B */
#define MASTER 0      /* rank of first task */
#define FROM_MASTER 1 /* setting a message type */
#define FROM_SLAVE 2  /* setting a message type */

void brute_force_matmul(double mat1[MAT_SIZE][MAT_SIZE], double mat2[MAT_SIZE][MAT_SIZE],
                        double res[MAT_SIZE][MAT_SIZE])
{
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
      rank,                  /* a task identifier */
      numworkers,            /* number of worker tasks */
      source,                /* task id of message source */
      dest,                  /* task id of message destination */
      mtype,                 /* message type */
      rows,                  /* rows of matrix A sent to each worker */
      averow, extra, offset, /* used to determine rows sent to each worker */
      i, j, k, rc;           /* misc */
  double a[m][n];
  double b[n][p];
  double c[m][p];
  double start;

  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  if (numtasks < 2)
  {
    double begin = clock();
    brute_force_matmul(a, b, c);
    double end = clock();
    printf("ELAPSED_TIME: %f\n", (end - begin) / CLOCKS_PER_SEC);
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
  }
  numworkers = numtasks;

  if (rank == MASTER)
  {
    /* master */
    FILE *file;
    char *filename1 = "matrixA.txt";
    file = fopen(filename1, "r");
    if (!file)
    {
      fprintf(stderr, "Unable to open file %s\n", filename1);
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < MAT_SIZE; i++)
      for (int j = 0; j < MAT_SIZE; j++)
        fscanf(file, "%lf", &a[i][j]);
    fclose(file);
    char *filename2 = "matrixB.txt";
    file = fopen(filename2, "r");
    if (!file)
    {
      fprintf(stderr, "Unable to open file %s\n", filename2);
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < MAT_SIZE; i++)
      for (int j = 0; j < MAT_SIZE; j++)
        fscanf(file, "%lf", &b[i][j]);

    /* Measure start time */
    start = MPI_Wtime();

    // Adjusted calculation for rows
    averow = m / numworkers;
    extra = m % numworkers;
    offset = 0;
    mtype = FROM_MASTER;

    for (dest = 0; dest < numworkers; dest++)
    {
      rows = (dest < extra) ? averow + 1 : averow;
      if (dest == MASTER)
      {
        // Skip sending data to itself (Master)
        offset += rows;
        continue;
      }
      printf("Sending %d rows to task %d offset=%d\n", rows, dest, offset);
      MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      MPI_Send(&a[offset][0], rows * n, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
      MPI_Send(&b[0][0], n * p, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
      offset += rows;
    }
  }

  // Both Master and Worker will execute the following code
  if (rank == MASTER)
  {
    rows = averow + (rank < extra ? 1 : 0); // Master's share of rows
    offset = 0;
  }
  else
  {
    mtype = FROM_MASTER;
    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&a[0][0], rows * n, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&b[0][0], n * p, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
  }

  for (k = 0; k < p; k++)
    for (i = 0; i < rows; i++)
    {
      c[i][k] = 0.0;
      for (j = 0; j < n; j++)
        c[i][k] = c[i][k] + a[i][j] * b[j][k];
    }

  if (rank != MASTER)
  {
    mtype = FROM_SLAVE;
    MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&c[0][0], rows * p, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  }
  else
  {
    // Directly store Master's computed results without MPI communication
    for (i = 0; i < rows; i++)
      for (j = 0; j < p; j++)
        c[i + offset][j] = c[i][j];
  }

  // If it's MASTER, finalize the process by collecting results and displaying time elapsed.
  if (rank == MASTER)
  {
    mtype = FROM_SLAVE;
    for (i = 1; i < numworkers; i++)
    {
      source = i;
      MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&c[offset][0], rows * p, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
    }

    double finish = MPI_Wtime();
    printf("ELAPSED_TIME: %f\n", finish - start);

    /* Compare the result with Brutal force */
    double d[m][p];
    brute_force_matmul(a, b, d);

    /* Check the result of the two methods */
    for (int i = 0; i < m; i++)
      for (int j = 0; j < p; j++)
        if (c[i][j] != d[i][j])
        {
          printf("Error: c[%d][%d] != d[%d][%d]\n", i, j, i, j);
          exit(1);
        }
    printf("The result is correct!\n");
  }

  MPI_Finalize();
  return 0;
}
