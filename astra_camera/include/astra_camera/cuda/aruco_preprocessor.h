#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace astra_camera {
namespace cuda {

/**
 * @brief GPU加速的ArUco图像预处理
 * 
 * @param d_input 输入图像（GPU设备内存）
 * @param d_output 输出图像（GPU设备内存）
 * @param d_temp1 临时缓冲区1（GPU设备内存）
 * @param d_temp2 临时缓冲区2（GPU设备内存）
 * @param width 图像宽度
 * @param height 图像高度
 * @param enable_adaptive_threshold 是否启用自适应阈值
 * @param enable_morphology 是否启用形态学操作
 * @param enable_blur 是否启用高斯模糊
 * @param enable_edge_enhance 是否启用边缘增强
 * @param stream CUDA流
 */
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
    cudaStream_t stream = 0);

}
}