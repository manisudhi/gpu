//
// src/amr/kernels/similarity.metal
// Apple Metal Compute Shader for Parallel Query-vs-All AMR Distance and Similarity
//
// Thread Mapping Architecture (Report Sections 14 & 22):
// - 1D Grid of GPU threads dispatched over num_records.
// - Each thread maps 1-to-1 to a record: thread_id = thread_position_in_grid.x.
// - Threadgroup size: 256 threads (aligned with Apple Silicon SIMD width of 32 threads, 8 SIMD-groups).
// - Input Buffer 0: Query vector [D] (float32)
// - Input Buffer 1: Contiguous Feature Matrix [N x D] (float32, row-major)
// - Output Buffer: Distance / Similarity scores [N] (float32)
//

#include <metal_stdlib>
using namespace metal;

kernel void amr_similarity_euclidean(
    device const float* query          [[buffer(0)]],
    device const float* feature_matrix [[buffer(1)]],
    device float*       out_distances  [[buffer(2)]],
    constant uint&      num_records    [[buffer(3)]],
    constant uint&      num_features   [[buffer(4)]],
    uint                gid            [[thread_position_in_grid]]
) {
    if (gid >= num_records) {
        return;
    }

    uint row_offset = gid * num_features;
    float sum_sq = 0.0f;

    for (uint f = 0; f < num_features; ++f) {
        float diff = query[f] - feature_matrix[row_offset + f];
        sum_sq += diff * diff;
    }

    out_distances[gid] = sqrt(sum_sq);
}

kernel void amr_similarity_cosine(
    device const float* query          [[buffer(0)]],
    device const float* feature_matrix [[buffer(1)]],
    device float*       out_similarities [[buffer(2)]],
    constant uint&      num_records    [[buffer(3)]],
    constant uint&      num_features   [[buffer(4)]],
    uint                gid            [[thread_position_in_grid]]
) {
    if (gid >= num_records) {
        return;
    }

    uint row_offset = gid * num_features;
    float dot_prod = 0.0f;
    float norm_q_sq = 0.0f;
    float norm_x_sq = 0.0f;

    for (uint f = 0; f < num_features; ++f) {
        float q_val = query[f];
        float x_val = feature_matrix[row_offset + f];
        dot_prod += q_val * x_val;
        norm_q_sq += q_val * q_val;
        norm_x_sq += x_val * x_val;
    }

    float denom = sqrt(norm_q_sq) * sqrt(norm_x_sq);
    out_similarities[gid] = (denom > 1e-7f) ? (dot_prod / denom) : 0.0f;
}
