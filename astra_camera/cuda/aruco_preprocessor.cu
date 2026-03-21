#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include "astra_camera/cuda/cuda_utils.h"

namespace astra_camera {
namespace cuda {

// 自适应阈值计算的共享内存版本
__global__ void adaptiveThresholdKernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    int block_size,
    float C)
{
    extern __shared__ uint8_t block[];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // 加载块数据到共享内存
    int block_idx = threadIdx.y * blockDim.x + threadIdx.x;
    block[block_idx] = input[idx];
    __syncthreads();
    
    // 计算局部均值
    int half_block = block_size / 2;
    int count = 0;
    int sum = 0;
    
    for (int dy = -half_block; dy <= half_block; dy++) {
        for (int dx = -half_block; dx <= half_block; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += input[ny * width + nx];
                count++;
            }
        }
    }
    
    float mean = static_cast<float>(sum) / count;
    float threshold = mean - C;
    
    output[idx] = (input[idx] > threshold) ? 255 : 0;
}

// 形态学腐蚀操作
__global__ void erodeKernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    uint8_t min_val = 255;
    
    int half_kernel = kernel_size / 2;
    
    for (int dy = -half_kernel; dy <= half_kernel; dy++) {
        for (int dx = -half_kernel; dx <= half_kernel; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                min_val = min(min_val, input[ny * width + nx]);
            }
        }
    }
    
    output[idx] = min_val;
}

// 形态学膨胀操作
__global__ void dilateKernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    uint8_t max_val = 0;
    
    int half_kernel = kernel_size / 2;
    
    for (int dy = -half_kernel; dy <= half_kernel; dy++) {
        for (int dx = -half_kernel; dx <= half_kernel; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny < height) {
                max_val = max(max_val, input[ny * width + nx]);
            }
        }
    }
    
    output[idx] = max_val;
}

// 高斯模糊（用于降噪）
__global__ void gaussianBlurKernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    float sigma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // 3x3高斯核
    const float kernel[3][3] = {
        {1.0f/16, 2.0f/16, 1.0f/16},
        {2.0f/16, 4.0f/16, 2.0f/16},
        {1.0f/16, 2.0f/16, 1.0f/16}
    };
    
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float weight = kernel[dy + 1][dx + 1];
                sum += input[ny * width + nx] * weight;
                weight_sum += weight;
            }
        }
    }
    
    output[idx] = static_cast<uint8_t>(sum / weight_sum);
}

// 边缘增强
__global__ void edgeEnhanceKernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    float strength)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Sobel算子
    const int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    const int sobel_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };
    
    float grad_x = 0.0f;
    float grad_y = 0.0f;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float pixel = static_cast<float>(input[ny * width + nx]);
                grad_x += pixel * sobel_x[dy + 1][dx + 1];
                grad_y += pixel * sobel_y[dy + 1][dx + 1];
            }
        }
    }
    
    float magnitude = sqrtf(grad_x * grad_x + grad_y * grad_y);
    float enhanced = static_cast<float>(input[idx]) + strength * magnitude;
    
    output[idx] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, enhanced)));
}

// 主预处理函数
void preprocessArUcoImageGPU(
    const uint8_t* d_input,
    uint8_t* d_output,
    uint8_t* d_temp1,
    uint8_t* d_temp2,
    int width,
    int height,
    bool enable_adaptive_threshold,
    bool enable_morphology,
    bool enable_blur,
    bool enable_edge_enhance,
    cudaStream_t stream)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    
    // 1. 高斯模糊（降噪）
    if (enable_blur) {
        gaussianBlurKernel<<<gridSize, blockSize, 0, stream>>>(
            d_input, d_temp1, width, height, 1.0f);
        CUDA_CHECK_LAST_ERROR();
    } else {
        cudaMemcpyAsync(d_temp1, d_input, width * height, cudaMemcpyDeviceToDevice, stream);
    }
    
    // 2. 自适应阈值
    if (enable_adaptive_threshold) {
        adaptiveThresholdKernel<<<gridSize, blockSize, blockSize.x * blockSize.y * sizeof(uint8_t), stream>>>(
            d_temp1, d_temp2, width, height, 15, 5.0f);
        CUDA_CHECK_LAST_ERROR();
    } else {
        cudaMemcpyAsync(d_temp2, d_temp1, width * height, cudaMemcpyDeviceToDevice, stream);
    }
    
    // 3. 形态学开运算（腐蚀+膨胀）
    if (enable_morphology) {
        erodeKernel<<<gridSize, blockSize, 0, stream>>>(
            d_temp2, d_temp1, width, height, 3);
        CUDA_CHECK_LAST_ERROR();
        
        dilateKernel<<<gridSize, blockSize, 0, stream>>>(
            d_temp1, d_temp2, width, height, 3);
        CUDA_CHECK_LAST_ERROR();
    }
    
    // 4. 边缘增强
    if (enable_edge_enhance) {
        edgeEnhanceKernel<<<gridSize, blockSize, 0, stream>>>(
            d_temp2, d_output, width, height, 0.5f);
        CUDA_CHECK_LAST_ERROR();
    } else {
        cudaMemcpyAsync(d_output, d_temp2, width * height, cudaMemcpyDeviceToDevice, stream);
    }
}

}
}