#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <mpi.h>

#define WIDTH 800
#define HEIGHT 600
#define MAX_ITERATIONS 1000

int main(int argc, char **argv) {
    int i, j, rank, size;
    double realPart, imaginaryPart;
    int *calculatedMandelbrotSet;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0) {
            printf("Error: Number of processes must be 4.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int startRow, endRow;
    int rowsPerProcess = HEIGHT / size;
    int remainingRows = HEIGHT % size;

    if (rank < remainingRows) {
        startRow = rank * (rowsPerProcess + 1);
        endRow = startRow + rowsPerProcess + 1;
    } else {
        startRow = rank * rowsPerProcess + remainingRows;
        endRow = startRow + rowsPerProcess;
    }

    calculatedMandelbrotSet = (int *) malloc(WIDTH * (endRow - startRow) * sizeof(int));

    for (i = startRow; i < endRow; i++) {
        for (j = 0; j < WIDTH; j++) {
            realPart = (double) (j - WIDTH/2) / (WIDTH/4);
            imaginaryPart = (double) (i - HEIGHT/2) / (HEIGHT/4);

            double complex c = realPart + imaginaryPart * I;
            double complex z = 0 + 0 * I;
            int iterations = 0;

            while (cabs(z) < 2 && iterations < MAX_ITERATIONS) {
                z = z * z + c;
                iterations++;
            }

            calculatedMandelbrotSet[(i - startRow) * WIDTH + j] = iterations;
        }
    }

    // Gather results from each process
    int *counts = (int *) malloc(size * sizeof(int));
    int *displs = (int *) malloc(size * sizeof(int));
    for (i = 0; i < size; i++) {
        if (i < remainingRows) {
            counts[i] = (rowsPerProcess + 1) * WIDTH;
            displs[i] = i * (rowsPerProcess + 1) * WIDTH;
        } else {
            counts[i] = rowsPerProcess * WIDTH;
            displs[i] = i * rowsPerProcess * WIDTH + remainingRows * (rowsPerProcess + 1) * WIDTH;
        }
    }

    MPI_Gatherv(calculatedMandelbrotSet, (endRow - startRow) * WIDTH, MPI_INT,
                calculatedMandelbrotSet, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Print final result
        for (i = 0; i < HEIGHT; i++) {
            for (j = 0; j < WIDTH; j++) {
                printf("%d ", calculatedMandelbrotSet[i * WIDTH + j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
