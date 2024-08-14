#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <stdio.h>
#include <stdlib.h>
#include <perfcounter.h>

#define N_POINTS 4000000  // Number of points
#define DIMENSIONS 5
#define K 150        // Number of clusters
#define TRANSFER_SIZE 2048  // Transfer size in bytes (must be multiple of 256)

// MRAM and WRAM buffers
__mram_noinit float points[N_POINTS][DIMENSIONS];
__dma_aligned float buffer[TRANSFER_SIZE / sizeof(float)];
__dma_aligned float buffer2[TRANSFER_SIZE / sizeof(float)];  // Second buffer for double buffering

BARRIER_INIT(my_barrier, NR_TASKLETS);

// Optimized data transfer function with double buffering
void transfer_data_to_wram_optimized(__mram_ptr float* mram_addr, float* wram_buffer1, float* wram_buffer2, size_t size) {
    int half_size = size / 2;

    // Start transferring the first half to buffer 1
    mram_read(mram_addr, wram_buffer1, half_size);
    
    // Transfer the second half to buffer 2 while processing buffer 1
    mram_read(mram_addr + half_size / sizeof(float), wram_buffer2, half_size);
    
    // Process buffer 1 in WRAM if necessary
    // process(wram_buffer1);

    // Process buffer 2 in WRAM if necessary
    // process(wram_buffer2);
}

int main() {
    // Initialize performance counters
    perfcounter_config(COUNT_CYCLES, true);  

    // Initialize MRAM data (for testing purposes)
    for(int i = 0; i < N_POINTS; i++) {
        points[i][0] = (float)i;
        points[i][1] = (float)i;
    }

    perfcounter_t transfer_start, transfer_end;
    unsigned long transfer_duration_original = 0;
    unsigned long transfer_duration_optimized = 0;

    // Measure original data transfer
    transfer_start = perfcounter_get();
    mram_read(points, buffer, TRANSFER_SIZE);  // Original transfer function
    transfer_end = perfcounter_get();
    transfer_duration_original = transfer_end - transfer_start;

    // Measure optimized data transfer
    transfer_start = perfcounter_get();
    transfer_data_to_wram_optimized((__mram_ptr float*)points, buffer, buffer2, TRANSFER_SIZE);  // Optimized transfer function
    transfer_end = perfcounter_get();
    transfer_duration_optimized = transfer_end - transfer_start;

    // Output results
    printf("Original data transfer duration: %lu cycles\n", transfer_duration_original);
    printf("Optimized data transfer duration: %lu cycles\n", transfer_duration_optimized);

    return 0;
}
