#pragma once

#include <cuda_runtime.h>
#include "cuda_utils.h"

namespace astra_camera {
namespace cuda {

void launchDepthToPointCloudXYZ(
    const uint16_t* d_depth,
    float* d_pointcloud,
    const CameraIntrinsics& intrinsics,
    int width,
    int height,
    float depth_scale,
    cudaStream_t stream);

void launchDepthToPointCloudXYZRGB(
    const uint16_t* d_depth,
    const uint8_t* d_rgb,
    float* d_pointcloud,
    const CameraIntrinsics& intrinsics,
    int width,
    int height,
    float depth_scale,
    cudaStream_t stream);

}
}
