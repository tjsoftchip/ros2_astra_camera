#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "astra_camera/cuda/cuda_utils.h"

namespace astra_camera {
namespace cuda {

__global__ void undistortKernel(
    const uint8_t* input,
    uint8_t* output,
    const float* camera_matrix,
    const float* dist_coeffs,
    int width,
    int height,
    int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float fx = camera_matrix[0];
    float fy = camera_matrix[4];
    float cx = camera_matrix[2];
    float cy = camera_matrix[5];
    
    float x_norm = (static_cast<float>(x) - cx) / fx;
    float y_norm = (static_cast<float>(y) - cy) / fy;
    
    float r2 = x_norm * x_norm + y_norm * y_norm;
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    
    float k1 = dist_coeffs[0];
    float k2 = dist_coeffs[1];
    float p1 = dist_coeffs[2];
    float p2 = dist_coeffs[3];
    float k3 = dist_coeffs[4];
    
    float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
    float x_distorted = x_norm * radial + 2.0f * p1 * x_norm * y_norm + p2 * (r2 + 2.0f * x_norm * x_norm);
    float y_distorted = y_norm * radial + p1 * (r2 + 2.0f * y_norm * y_norm) + 2.0f * p2 * x_norm * y_norm;
    
    int x_src = __float2int_rn(x_distorted * fx + cx);
    int y_src = __float2int_rn(y_distorted * fy + cy);
    
    int out_idx = (y * width + x) * channels;
    
    if (x_src >= 0 && x_src < width && y_src >= 0 && y_src < height) {
        int in_idx = (y_src * width + x_src) * channels;
        for (int c = 0; c < channels; c++) {
            output[out_idx + c] = input[in_idx + c];
        }
    } else {
        for (int c = 0; c < channels; c++) {
            output[out_idx + c] = 0;
        }
    }
}

__global__ void imageEnhanceKernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    int channels,
    float brightness,
    float contrast)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height * channels;
    
    if (idx >= total) return;
    
    float value = static_cast<float>(input[idx]);
    value = (value - 128.0f) * contrast + 128.0f + brightness;
    value = fminf(255.0f, fmaxf(0.0f, value));
    
    output[idx] = static_cast<uint8_t>(value);
}

void launchUndistort(
    const uint8_t* d_input,
    uint8_t* d_output,
    const float* d_camera_matrix,
    const float* d_dist_coeffs,
    int width,
    int height,
    int channels,
    cudaStream_t stream)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    
    undistortKernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_output, d_camera_matrix, d_dist_coeffs, width, height, channels);
    
    CUDA_CHECK_LAST_ERROR();
}

void launchImageEnhance(
    const uint8_t* d_input,
    uint8_t* d_output,
    int width,
    int height,
    int channels,
    float brightness,
    float contrast,
    cudaStream_t stream)
{
    int total = width * height * channels;
    dim3 blockSize(256);
    dim3 gridSize((total + 255) / 256);
    
    imageEnhanceKernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_output, width, height, channels, brightness, contrast);
    
    CUDA_CHECK_LAST_ERROR();
}

}
}
