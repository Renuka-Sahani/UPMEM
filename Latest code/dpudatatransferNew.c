#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <stdio.h>
#include <stdlib.h>
#include <perfcounter.h>

#define N_POINTS 10000  // Number of points
#define DIMENSIONS 2
#define K 15           // Number of clusters
#define MAX_ITERATIONS 15
#define TRANSFER_SIZE 256  // Transfer size in bytes

__mram_noinit float points[N_POINTS][DIMENSIONS];
__mram_noinit float centroids[K][DIMENSIONS];
__mram_noinit int clusters[N_POINTS];
__dma_aligned float buffer[TRANSFER_SIZE / sizeof(float)];

BARRIER_INIT(my_barrier, NR_TASKLETS);

// Function to calculate square root using Newton's method
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

// Generate predefined clusters around fixed centroids
void generate_fixed_clusters() {
    float predefined_centroids[K][DIMENSIONS] = {
        {1.0, 1.0},  // Centroid for cluster 1
        {5.0, 5.0},  // Centroid for cluster 2
        {9.0, 9.0},  // Centroid for cluster 3
        {13.0, 13.0},// Centroid for cluster 4
        {17.0, 17.0},// Centroid for cluster 5
        {21.0, 21.0}, // Centroid for cluster 6
        {25.0, 25.0}, // Centroid for cluster 7
        {29.0, 29.0}, // Centroid for cluster 8
        {33.0, 33.0}, // Centroid for cluster 9
        {37.0, 37.0}, // Centroid for cluster 10
        {41.0, 41.0}, // Centroid for cluster 11
        {45.0, 45.0}, // Centroid for cluster 12
        {49.0, 49.0}, // Centroid for cluster 13
        {53.0, 53.0}, // Centroid for cluster 14
        {57.0, 57.0}  // Centroid for cluster 15 
    };

    int points_per_cluster = N_POINTS / K;

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < points_per_cluster; j++) {
            int index = i * points_per_cluster + j;
            for (int d = 0; d < DIMENSIONS; d++) {
                points[index][d] = predefined_centroids[i][d];
            }
        }
    }
}

// K-means clustering functions (assign clusters and update centroids)
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

// Function to manage data transfer
void transfer_data_to_wram() {
    mram_read(points, buffer, TRANSFER_SIZE);
    // Process the data in WRAM if necessary
}

// Function to print centroids
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
    perfcounter_t start_time, end_time;
    perfcounter_t assign_start, assign_end;
    perfcounter_t update_start, update_end;
    perfcounter_t transfer_start, transfer_end;

    // Initialize performance counters
    perfcounter_config(COUNT_CYCLES, true);

    // Start total time measurement
    start_time = perfcounter_get();

    // Generate fixed clusters
    generate_fixed_clusters();

    // Initialize centroids with predefined values (same as Python code)
    float initial_centroids[K][DIMENSIONS] = {
        {1.1, 1.1},
        {5.1, 5.1},
        {9.1, 9.1},
        {13.1, 13.1},
        {17.1, 17.1},
        {21.1, 21.1}
    };
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            centroids[i][j] = initial_centroids[i][j];
        }
    }

    barrier_wait(&my_barrier);

    // Print initial centroids
    print_centroids("Initial Centroids");

    // Perform K-means clustering
    for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
        printf("\nIteration %d:\n", iteration + 1);

        // Data Transfer (for illustration purposes)
        transfer_start = perfcounter_get();
        transfer_data_to_wram();
        transfer_end = perfcounter_get();
        printf("Data transfer duration: %lu cycles\n", transfer_end - transfer_start);

        // Measure assign clusters duration
        assign_start = perfcounter_get();
        assign_clusters();
        assign_end = perfcounter_get();
        printf("Assign clusters duration: %lu cycles\n", assign_end - assign_start);
        
        barrier_wait(&my_barrier);

        // Measure update centroids duration
        update_start = perfcounter_get();
        update_centroids();
        update_end = perfcounter_get();
        printf("Update centroids duration: %lu cycles\n", update_end - update_start);
        
        barrier_wait(&my_barrier);

        // Print centroids after update
        print_centroids("Updated Centroids");
    }

    // End total time measurement
    end_time = perfcounter_get();
    printf("Total duration: %lu cycles\n", end_time - start_time);

    // Print final centroids
    print_centroids("Final Centroids");

    return 0;
}
