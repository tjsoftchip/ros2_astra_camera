#pragma once

#include <cuda_runtime.h>
#include "cuda_utils.h"

namespace astra_camera {
namespace cuda {

void launchUndistort(
    const uint8_t* d_input,
    uint8_t* d_output,
    const float* d_camera_matrix,
    const float* d_dist_coeffs,
    int width,
    int height,
    int channels,
    cudaStream_t stream);

void launchImageEnhance(
    const uint8_t* d_input,
    uint8_t* d_output,
    int width,
    int height,
    int channels,
    float brightness,
    float contrast,
    cudaStream_t stream);

}
}
