#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <stdio.h>    // Include for file handling functions
#include <stdlib.h>   // Include for rand function

#define N_POINTS 10000  // Update this to the size of your dataset
#define DIMENSIONS 2
#define K 15
#define MAX_ITERATIONS 100

__mram_noinit float points[N_POINTS][DIMENSIONS];
__mram_noinit float centroids[K][DIMENSIONS];
__mram_noinit int clusters[N_POINTS];

BARRIER_INIT(my_barrier, NR_TASKLETS);

// Simple function to calculate square root using Newton's method
float sqrt_approx(float number) {
    float x = number;
    float y = 1.0;
    float e = 0.01;  // error threshold

    while (x - y > e) {
        x = (x + y) / 2;
        y = number / x;
    }
    return x;
}

// Simulate loading points from a file (this would be done on the host normally)
void load_points_from_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N_POINTS; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            if (fscanf(file, "%f", &points[i][j]) != 1) {
                printf("Error reading point %d, dimension %d from file\n", i, j);
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(file);
}

void assign_clusters() {
    for (int i = 0; i < N_POINTS; i++) {
        float min_distance = 1e30;
        int closest_centroid = 0;
        for (int j = 0; j < K; j++) {
            float distance = 0.0;
            for (int d = 0; d < DIMENSIONS; d++) {
                float diff = points[i][d] - centroids[j][d];
                distance += diff * diff;
            }
            distance = sqrt_approx(distance);
            if (distance < min_distance) {
                min_distance = distance;
                closest_centroid = j;
            }
        }
        clusters[i] = closest_centroid;
    }
}

void update_centroids() {
    float new_centroids[K][DIMENSIONS] = {0};
    int count[K] = {0};

    for (int i = 0; i < N_POINTS; i++) {
        int cluster_id = clusters[i];
        for (int d = 0; d < DIMENSIONS; d++) {
            new_centroids[cluster_id][d] += points[i][d];
        }
        count[cluster_id]++;
    }

    for (int j = 0; j < K; j++) {
        if (count[j] != 0) {
            for (int d = 0; d < DIMENSIONS; d++) {
                centroids[j][d] = new_centroids[j][d] / count[j];
            }
        }
    }
}

void print_centroids(const char* title) {
    printf("%s:\n", title);
    for (int i = 0; i < K; i++) {
        printf("Centroid %d: (", i);
        for (int j = 0; j < DIMENSIONS; j++) {
            printf("%f", centroids[i][j]);
            if (j < DIMENSIONS - 1) printf(", ");
        }
        printf(")\n");
    }
}

int main() {
    // Load points from the file
    load_points_from_file("points.txt");

    barrier_wait(&my_barrier);

    // Initialize centroids with random points
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            centroids[i][j] = points[rand() % N_POINTS][j];
        }
    }

    // Print initial centroids
    print_centroids("Initial Centroids");

    for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
        printf("\nIteration %d:\n", iteration + 1);

        assign_clusters();
        barrier_wait(&my_barrier);

        update_centroids();
        barrier_wait(&my_barrier);

        // Print centroids after update
        print_centroids("Updated Centroids");
    }

    // Print final centroids
    print_centroids("Final Centroids");

    return 0;
}
