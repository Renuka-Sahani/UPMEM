#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <stdio.h>
#include <stdlib.h>
#include <perfcounter.h>

#define N_POINTS 10000
#define DIMENSIONS 2
#define K 6
#define TRANSFER_SIZE 2048

__mram_noinit float points[N_POINTS][DIMENSIONS];
__mram_noinit float centroids[K][DIMENSIONS];
__mram_noinit int clusters[N_POINTS];
__dma_aligned float buffer[TRANSFER_SIZE / sizeof(float)];
__dma_aligned float buffer2[TRANSFER_SIZE / sizeof(float)];

BARRIER_INIT(my_barrier, NR_TASKLETS);

// Function to generate predefined clusters around fixed centroids
void generate_fixed_clusters() {
    float predefined_centroids[K][DIMENSIONS] = {
        {1.0, 1.0},
        {5.0, 5.0},
        {9.0, 9.0},
        {13.0, 13.0},
        {17.0, 17.0},
        {21.0, 21.0}
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

// Assign points to the closest centroid
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

// Update centroids based on assigned clusters
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
        for (int d = 0; d < DIMENSIONS; d++) {
            if (count[j] > 0) {
                centroids[j][d] = new_centroids[j][d] / count[j];
            }
        }
    }
}

// Original data transfer function (placeholder)
void original_data_transfer(__mram_ptr float* mram_addr, float* wram_buffer, size_t size) {
    mram_read(mram_addr, wram_buffer, size);
}

// Optimized data transfer function (placeholder)
void optimized_data_transfer(__mram_ptr float* mram_addr, float* wram_buffer1, float* wram_buffer2, size_t size) {
    int half_size = size / 2;
    mram_read(mram_addr, wram_buffer1, half_size);
    mram_read(mram_addr + half_size / sizeof(float), wram_buffer2, half_size);
}

int main() {
    // Initialize performance counters
    perfcounter_config(COUNT_CYCLES, true);

    // Initialize MRAM data (for testing purposes)
    for(int i = 0; i < N_POINTS; i++) {
        points[i][0] = (float)i;
        points[i][1] = (float)i;
    }

    // Generate initial clusters
    generate_fixed_clusters();

    // Perform clustering iterations
    for (int iteration = 0; iteration < 10; iteration++) {
        assign_clusters();
        update_centroids();
    }

    // Measure execution time of original and optimized data transfer functions
    perfcounter_t start, end;

    start = perfcounter_get();
    original_data_transfer((__mram_ptr float*)points, buffer, TRANSFER_SIZE);
    end = perfcounter_get();
    unsigned long original_duration = end - start;

    start = perfcounter_get();
    optimized_data_transfer((__mram_ptr float*)points, buffer, buffer2, TRANSFER_SIZE);
    end = perfcounter_get();
    unsigned long optimized_duration = end - start;

    printf("Original data transfer duration: %lu cycles\n", original_duration);
    printf("Optimized data transfer duration: %lu cycles\n", optimized_duration);

    // Print final message
    printf("Benchmarking is complete.\n");

    return 0;
}
