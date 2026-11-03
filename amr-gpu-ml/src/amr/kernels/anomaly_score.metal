//
// src/amr/kernels/anomaly_score.metal
// Apple Metal Compute Shader for Per-Record Distance-to-Assigned-Centroid Anomaly Scoring
//
// Thread Mapping Architecture (Report Sections 14, 18 & 22):
// - 1D Grid of GPU threads dispatched over num_records.
// - Each thread reads assigned centroid and computes Euclidean distance: thread_id = thread_position_in_grid.x.
// - Threadgroup size: 256 threads.
// - Input Buffer 0: Contiguous Feature Matrix [N x D] (float32)
// - Input Buffer 1: Centroids Matrix [K x D] (float32)
// - Input Buffer 2: Cluster Assignments [N] (uint32)
// - Output Buffer 3: Anomaly Scores [N] (float32, Euclidean distance to centroid)
//

#include <metal_stdlib>
using namespace metal;

kernel void amr_anomaly_score(
    device const float* feature_matrix      [[buffer(0)]],
    device const float* centroids           [[buffer(1)]],
    device const uint*  cluster_assignments [[buffer(2)]],
    device float*       out_anomaly_scores  [[buffer(3)]],
    constant uint&      num_records         [[buffer(4)]],
    constant uint&      num_features        [[buffer(5)]],
    uint                gid                 [[thread_position_in_grid]]
) {
    if (gid >= num_records) {
        return;
    }

    uint assigned_k = cluster_assignments[gid];
    uint record_offset = gid * num_features;
    uint centroid_offset = assigned_k * num_features;

    float sum_sq = 0.0f;
    for (uint f = 0; f < num_features; ++f) {
        float diff = feature_matrix[record_offset + f] - centroids[centroid_offset + f];
        sum_sq += diff * diff;
    }

    out_anomaly_scores[gid] = sqrt(sum_sq);
}
