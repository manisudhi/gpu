//
// src/amr/kernels/kmeans_assign.metal
// Apple Metal Compute Shader for K-Means Nearest-Centroid Assignment
//
// Thread Mapping Architecture (Report Sections 14, 17 & 22):
// - 1D Grid of GPU threads dispatched over num_records.
// - Each thread evaluates all K centroids for one record: thread_id = thread_position_in_grid.x.
// - Threadgroup size: 256 threads.
// - Input Buffer 0: Contiguous Feature Matrix [N x D] (float32)
// - Input Buffer 1: Centroids Matrix [K x D] (float32)
// - Output Buffer 2: Cluster Assignments [N] (uint32)
// - Output Buffer 3: Minimum Squared Distances [N] (float32)
//

#include <metal_stdlib>
using namespace metal;

kernel void amr_kmeans_assign(
    device const float* feature_matrix      [[buffer(0)]],
    device const float* centroids           [[buffer(1)]],
    device uint*        cluster_assignments [[buffer(2)]],
    device float*       min_sq_distances    [[buffer(3)]],
    constant uint&      num_records         [[buffer(4)]],
    constant uint&      num_features        [[buffer(5)]],
    constant uint&      num_clusters        [[buffer(6)]],
    uint                gid                 [[thread_position_in_grid]]
) {
    if (gid >= num_records) {
        return;
    }

    uint record_offset = gid * num_features;
    uint best_cluster = 0;
    float min_dist_sq = 1e30f;

    for (uint k = 0; k < num_clusters; ++k) {
        uint centroid_offset = k * num_features;
        float current_dist_sq = 0.0f;

        for (uint f = 0; f < num_features; ++f) {
            float diff = feature_matrix[record_offset + f] - centroids[centroid_offset + f];
            current_dist_sq += diff * diff;
        }

        if (current_dist_sq < min_dist_sq) {
            min_dist_sq = current_dist_sq;
            best_cluster = k;
        }
    }

    cluster_assignments[gid] = best_cluster;
    min_sq_distances[gid] = min_dist_sq;
}
