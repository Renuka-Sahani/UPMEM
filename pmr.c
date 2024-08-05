#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <stdio.h>
#include <stdlib.h>

#define N_POINTS 10000  // Number of points
#define DIMENSIONS 2
#define K 6            // Number of clusters
#define MAX_ITERATIONS 15
#define LOCAL_POINTS_PER_TASKLET_SIZE 100  // Fixed size per tasklet

__mram_noinit float points[N_POINTS][DIMENSIONS];
__mram_noinit float centroids[K][DIMENSIONS];
__mram_noinit int clusters[N_POINTS];

BARRIER_INIT(my_barrier, NR_TASKLETS);

// Simple Linear Congruential Generator (LCG)
static unsigned long next = 1;
int my_rand(void) {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next / 65536) % 32768;
}

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

// Generate simple clusters around predefined centroids
void generate_simple_clusters() {
    float predefined_centroids[K][DIMENSIONS] = {
        {1.0, 1.0},  // Centroid for cluster 1
        {5.0, 5.0},  // Centroid for cluster 2
        {9.0, 9.0},  // Centroid for cluster 3
        {13.0, 13.0},// Centroid for cluster 4
        {17.0, 17.0},// Centroid for cluster 5
        {21.0, 21.0} // Centroid for cluster 6
    };

    int points_per_cluster = N_POINTS / K;

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < points_per_cluster; j++) {
            int index = i * points_per_cluster + j;
            for (int d = 0; d < DIMENSIONS; d++) {
                points[index][d] = predefined_centroids[i][d] + (float)(my_rand() % 100) / 1000.0;  // Small random variation
            }
        }
    }
}

// K-means clustering functions (assign clusters and update centroids)
void assign_clusters() {
    __dma_aligned float local_points[LOCAL_POINTS_PER_TASKLET_SIZE][DIMENSIONS];

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
    // Generate simple clusters
    generate_simple_clusters();

    // Initialize centroids randomly from points
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            centroids[i][j] = points[my_rand() % N_POINTS][j];
        }
    }

    barrier_wait(&my_barrier);

    // Print initial centroids
    print_centroids("Initial Centroids");

    // Perform K-means clustering
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
