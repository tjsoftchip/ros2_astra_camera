#pragma once

#include <cuda_runtime.h>
#include "cuda_utils.h"

namespace astra_camera {
namespace cuda {

void launchBilateralFilter(
    const uint16_t* d_input,
    uint16_t* d_output,
    int width,
    int height,
    float spatial_sigma,
    float range_sigma,
    cudaStream_t stream);

void launchMedianFilter(
    const uint16_t* d_input,
    uint16_t* d_output,
    int width,
    int height,
    cudaStream_t stream);

}
}
