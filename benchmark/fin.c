#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <stdio.h>
#include <stdlib.h>
#include <perfcounter.h>

#define N_POINTS 4000000  // Adjust this as needed
#define DIMENSIONS 2
#define K 15           // Number of clusters
#define TRANSFER_SIZE 2048  // Adjust this as needed

__mram_noinit float points[N_POINTS][DIMENSIONS];
__mram_noinit float centroids[K][DIMENSIONS];
__mram_noinit int clusters[N_POINTS];
__dma_aligned float buffer[TRANSFER_SIZE / sizeof(float)];
__dma_aligned float buffer2[TRANSFER_SIZE / sizeof(float)];

BARRIER_INIT(my_barrier, NR_TASKLETS);

// Original data transfer function (from dpudatatransfer.c)
void transfer_data_to_wram() {
    mram_read(points, buffer, TRANSFER_SIZE);
    // Process the data in WRAM if necessary
}

// Optimized data transfer function (from dpuOptimizedDataTransfer.c)
void transfer_data_to_wram_optimized(__mram_ptr float* mram_addr, float* wram_buffer, size_t size) {
    mram_read(mram_addr, wram_buffer, size);
    // Additional processing in WRAM if necessary
}

int main() {
    perfcounter_config(COUNT_CYCLES, true);
    perfcounter_t start, end;
    unsigned long original_duration, optimized_duration;

    // Benchmark original data transfer
    start = perfcounter_get();
    transfer_data_to_wram((__mram_ptr float*)points, buffer); // Assuming the function takes two arguments
    end = perfcounter_get();
    original_duration = end - start;
    printf("Original data transfer duration: %lu cycles\n", original_duration);

    // Benchmark optimized data transfer
    start = perfcounter_get();
    transfer_data_to_wram_optimized((__mram_ptr float*)points, buffer, TRANSFER_SIZE); // Assuming the function takes three arguments
    end = perfcounter_get();
    optimized_duration = end - start;
    printf("Optimized data transfer duration: %lu cycles\n", optimized_duration);

    return 0;
}

