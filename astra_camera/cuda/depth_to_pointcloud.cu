#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "astra_camera/cuda/cuda_utils.h"

namespace astra_camera {
namespace cuda {

__global__ void depthToPointCloudXYZKernel(
    const uint16_t* depth_data,
    float* pointcloud_data,
    const CameraIntrinsics intrinsics,
    int width,
    int height,
    float depth_scale)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx >= width * height) return;
  
  int u = idx % width;
  int v = idx / width;
  
  uint16_t depth_raw = depth_data[idx];
  
  if (depth_raw == 0) {
    pointcloud_data[idx * 3 + 0] = __int_as_float(0x7fc00000);
    pointcloud_data[idx * 3 + 1] = __int_as_float(0x7fc00000);
    pointcloud_data[idx * 3 + 2] = __int_as_float(0x7fc00000);
    return;
  }
  
  float z = static_cast<float>(depth_raw) * depth_scale;
  
  float x = (static_cast<float>(u) - intrinsics.cx) * z / intrinsics.fx;
  float y = (static_cast<float>(v) - intrinsics.cy) * z / intrinsics.fy;
  
  pointcloud_data[idx * 3 + 0] = x;
  pointcloud_data[idx * 3 + 1] = y;
  pointcloud_data[idx * 3 + 2] = z;
}

__global__ void depthToPointCloudXYZRGBKernel(
    const uint16_t* depth_data,
    const uint8_t* rgb_data,
    float* pointcloud_data,
    const CameraIntrinsics intrinsics,
    int width,
    int height,
    float depth_scale)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx >= width * height) return;
  
  int u = idx % width;
  int v = idx / width;
  
  uint16_t depth_raw = depth_data[idx];
  
  if (depth_raw == 0) {
    pointcloud_data[idx * 6 + 0] = __int_as_float(0x7fc00000);
    pointcloud_data[idx * 6 + 1] = __int_as_float(0x7fc00000);
    pointcloud_data[idx * 6 + 2] = __int_as_float(0x7fc00000);
    pointcloud_data[idx * 6 + 3] = 0.0f;
    pointcloud_data[idx * 6 + 4] = 0.0f;
    pointcloud_data[idx * 6 + 5] = 0.0f;
    return;
  }
  
  float z = static_cast<float>(depth_raw) * depth_scale;
  
  float x = (static_cast<float>(u) - intrinsics.cx) * z / intrinsics.fx;
  float y = (static_cast<float>(v) - intrinsics.cy) * z / intrinsics.fy;
  
  pointcloud_data[idx * 6 + 0] = x;
  pointcloud_data[idx * 6 + 1] = y;
  pointcloud_data[idx * 6 + 2] = z;
  
  int rgb_idx = idx * 3;
  pointcloud_data[idx * 6 + 3] = static_cast<float>(rgb_data[rgb_idx + 0]) / 255.0f;
  pointcloud_data[idx * 6 + 4] = static_cast<float>(rgb_data[rgb_idx + 1]) / 255.0f;
  pointcloud_data[idx * 6 + 5] = static_cast<float>(rgb_data[rgb_idx + 2]) / 255.0f;
}

void launchDepthToPointCloudXYZ(
    const uint16_t* d_depth,
    float* d_pointcloud,
    const CameraIntrinsics& intrinsics,
    int width,
    int height,
    float depth_scale,
    cudaStream_t stream)
{
  int total_pixels = width * height;
  
  dim3 blockSize(256);
  dim3 gridSize((total_pixels + blockSize.x - 1) / blockSize.x);
  
  depthToPointCloudXYZKernel<<<gridSize, blockSize, 0, stream>>>(
      d_depth, d_pointcloud, intrinsics, width, height, depth_scale);
  
  CUDA_CHECK_LAST_ERROR();
}

void launchDepthToPointCloudXYZRGB(
    const uint16_t* d_depth,
    const uint8_t* d_rgb,
    float* d_pointcloud,
    const CameraIntrinsics& intrinsics,
    int width,
    int height,
    float depth_scale,
    cudaStream_t stream)
{
  int total_pixels = width * height;
  
  dim3 blockSize(256);
  dim3 gridSize((total_pixels + blockSize.x - 1) / blockSize.x);
  
  depthToPointCloudXYZRGBKernel<<<gridSize, blockSize, 0, stream>>>(
      d_depth, d_rgb, d_pointcloud, intrinsics, width, height, depth_scale);
  
  CUDA_CHECK_LAST_ERROR();
}

}
}
