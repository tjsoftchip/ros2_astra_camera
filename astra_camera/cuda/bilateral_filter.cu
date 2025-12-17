#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "astra_camera/cuda/cuda_utils.h"

namespace astra_camera {
namespace cuda {

__global__ void bilateralFilterKernel(
    const uint16_t* input,
    uint16_t* output,
    int width,
    int height,
    float spatial_sigma,
    float range_sigma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    uint16_t center = input[idx];
    
    if (center == 0) {
        output[idx] = 0;
        return;
    }
    
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    const int radius = 2;
    const float inv_spatial_sigma2 = -0.5f / (spatial_sigma * spatial_sigma);
    const float inv_range_sigma2 = -0.5f / (range_sigma * range_sigma);
    
    #pragma unroll
    for (int dy = -radius; dy <= radius; dy++) {
        int ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        
        #pragma unroll
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            if (nx < 0 || nx >= width) continue;
            
            int neighbor_idx = ny * width + nx;
            uint16_t neighbor = input[neighbor_idx];
            
            if (neighbor == 0) continue;
            
            float spatial_dist = static_cast<float>(dx * dx + dy * dy);
            float spatial_weight = __expf(spatial_dist * inv_spatial_sigma2);
            
            int range_dist = static_cast<int>(center) - static_cast<int>(neighbor);
            float range_weight = __expf(static_cast<float>(range_dist * range_dist) * inv_range_sigma2);
            
            float weight = spatial_weight * range_weight;
            sum += static_cast<float>(neighbor) * weight;
            weight_sum += weight;
        }
    }
    
    if (weight_sum > 0.0f) {
        output[idx] = static_cast<uint16_t>(sum / weight_sum + 0.5f);
    } else {
        output[idx] = center;
    }
}

__global__ void medianFilterKernel(
    const uint16_t* input,
    uint16_t* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    uint16_t center = input[idx];
    
    if (center == 0) {
        output[idx] = 0;
        return;
    }
    
    const int radius = 1;
    uint16_t values[9];
    int count = 0;
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                uint16_t val = input[ny * width + nx];
                if (val > 0) {
                    values[count++] = val;
                }
            }
        }
    }
    
    if (count == 0) {
        output[idx] = 0;
        return;
    }
    
    for (int i = 0; i < count - 1; i++) {
        for (int j = 0; j < count - i - 1; j++) {
            if (values[j] > values[j + 1]) {
                uint16_t temp = values[j];
                values[j] = values[j + 1];
                values[j + 1] = temp;
            }
        }
    }
    
    output[idx] = values[count / 2];
}

void launchBilateralFilter(
    const uint16_t* d_input,
    uint16_t* d_output,
    int width,
    int height,
    float spatial_sigma,
    float range_sigma,
    cudaStream_t stream)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    
    bilateralFilterKernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_output, width, height, spatial_sigma, range_sigma);
    
    CUDA_CHECK_LAST_ERROR();
}

void launchMedianFilter(
    const uint16_t* d_input,
    uint16_t* d_output,
    int width,
    int height,
    cudaStream_t stream)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    
    medianFilterKernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_output, width, height);
    
    CUDA_CHECK_LAST_ERROR();
}

}
}
