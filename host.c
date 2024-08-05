#include <stdio.h>
#include <stdlib.h>
#include <dpu.h>
#include <dpu_log.h>

#define N_POINTS 10000
#define DIMENSIONS 2

float points[N_POINTS][DIMENSIONS];

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

int main() {
    struct dpu_set_t dpus, dpu;

    // Load points from the file
    load_points_from_file("points.txt");

    // Allocate a simulated DPU
    DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, NULL, &dpus));

    // Load the DPU program onto the simulator
    DPU_ASSERT(dpu_load(dpus, "new_advanced_dpu", NULL));

    // Transfer the points to the simulated DPU's MRAM
    DPU_FOREACH(dpus, dpu) {
        DPU_ASSERT(dpu_copy_to(dpu, "points", 0, points, sizeof(points)));
    }

    // Run the DPU program on the simulator
    DPU_ASSERT(dpu_launch(dpus, DPU_SYNCHRONOUS));

    // Retrieve and print logs from the simulator
    DPU_FOREACH(dpus, dpu) {
        dpu_log_read(dpu, stdout);
    }

    // Free the simulated DPUs
    DPU_ASSERT(dpu_free(dpus));

    return 0;
}
