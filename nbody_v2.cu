#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "timer.h" // Include the timer header
#include "matric.h" // Include your custom matric.h header

#define SOFTENING 1e-9f

typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

void randomizeBodies(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

__global__ void bodyForce(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            if (i != j) {
                float dx = p[j].x - p[i].x;
                float dy = p[j].y - p[i].y;
                float dz = p[j].z - p[i].z;
                float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

__global__ void integratePositions(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

void saveForcesToFile(const char *filename, int nBodies, Body *p) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Unable to open file %s for writing.\n", filename);
        return;
    }
    for (int i = 0; i < nBodies; i++) {
        fprintf(file, "Body %d: x = %.3f, y = %.3f, z = %.3f, vx = %.3f, vy = %.3f, vz = %.3f\n",
                i, p[i].x, p[i].y, p[i].z, p[i].vx, p[i].vy, p[i].vz);
    }
    fclose(file);
}

int main(int argc, char **argv) {
    int nBodies = 200000;
    if (argc > 1) nBodies = atoi(argv[1]);

    const float dt = 0.01f; // time step
    const int nIters = 500;  // simulation iterations

    int bytes = nBodies * sizeof(Body);
    Body *p = (Body *)malloc(bytes);

    if (p == NULL) {
        fprintf(stderr, "Unable to allocate memory for bodies.\n");
        return 1;
    }

    float *buf = (float *)malloc(6 * nBodies * sizeof(float));
    if (buf == NULL) {
        fprintf(stderr, "Unable to allocate memory for buffer.\n");
        free(p);
        return 1;
    }

    randomizeBodies(buf, 6 * nBodies); // Init pos / vel data
    for (int i = 0; i < nBodies; i++) {
        p[i].x = buf[6 * i];
        p[i].y = buf[6 * i + 1];
        p[i].z = buf[6 * i + 2];
        p[i].vx = buf[6 * i + 3];
        p[i].vy = buf[6 * i + 4];
        p[i].vz = buf[6 * i + 5];
    }

    free(buf);

    // Allocate device memory
    Body *d_p;
    cudaMalloc(&d_p, bytes);

    cudaMemcpy(d_p, p, bytes, cudaMemcpyHostToDevice);

    double totalTime = 0.0;

    int blockSize = 256; // adjust value for performance tuning
    int numBlocks = (nBodies + blockSize - 1) / blockSize;

    for (int iter = 1; iter <= nIters; iter++) {
        StartTimer();

        bodyForce<<<numBlocks, blockSize>>>(d_p, dt, nBodies);
        cudaDeviceSynchronize();
        integratePositions<<<numBlocks, blockSize>>>(d_p, dt, nBodies);
        cudaDeviceSynchronize();

        const double tElapsed = GetTimer() / 1000.0;
        if (iter > 1) { // First iter is warm up
            totalTime += tElapsed;
        }
        printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
    }

    cudaMemcpy(p, d_p, bytes, cudaMemcpyDeviceToHost);

    saveForcesToFile("forces.txt", nBodies, p);

    double avgTime = totalTime / (double)(nIters - 1);
    double rate = (double)nBodies / avgTime;

    printf("Average rate for iterations 2 through %d: %.3f steps per second.\n",
           nIters, rate);
    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);

    free(p);

    cudaFree(d_p);

    return 0;
}
