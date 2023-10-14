#include <stdio.h>
#include <stdlib.h>

int main() {
  int m = 3, n = 2, p = 4;
  double **A, **B, **C;

  // Allocate memory for matrices A, B, and C
  A = (double **)malloc(m * sizeof(double *));
  B = (double **)malloc(n * sizeof(double *));
  C = (double **)malloc(m * sizeof(double *));
  for (int i = 0; i < m; i++) {
    A[i] = (double *)malloc(n * sizeof(double));
    C[i] = (double *)malloc(p * sizeof(double));
  }
  for (int i = 0; i < n; i++) {
    B[i] = (double *)malloc(p * sizeof(double));
  }

  // Initialize matrices A and B
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] = i + j;
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      B[i][j] = i * j;
    }
  }

  // Compute matrix multiplication C = A * B
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      C[i][j] = 0;
      for (int k = 0; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  // Print matrix C
  printf("Matrix C:\n");
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      printf("%f ", C[i][j]);
    }
    printf("\n");
  }

  // Free memory
  for (int i = 0; i < m; i++) {
    free(A[i]);
    free(C[i]);
  }
  for (int i = 0; i < n; i++) {
    free(B[i]);
  }
  free(A);
  free(B);
  free(C);

  return 0;
}
