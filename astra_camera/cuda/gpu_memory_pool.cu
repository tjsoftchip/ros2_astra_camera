#include "astra_camera/cuda/gpu_memory_pool.h"
#include <iostream>

namespace astra_camera {
namespace cuda {

GPUMemoryPool::GPUMemoryPool(int width, int height)
    : width_(width),
      height_(height),
      d_depth_buffer_(nullptr),
      d_pointcloud_buffer_(nullptr),
      d_rgb_buffer_(nullptr),
      d_filtered_depth_buffer_(nullptr),
      stream_(nullptr)
{
  CUDA_CHECK(cudaStreamCreate(&stream_));
  allocateBuffers();
  
  std::cout << "[GPUMemoryPool] Initialized for " << width << "x" << height << std::endl;
  std::cout << "[GPUMemoryPool] Total GPU memory: " 
            << getTotalMemoryUsage() / (1024.0 * 1024.0) << " MB" << std::endl;
}

GPUMemoryPool::~GPUMemoryPool() {
  freeBuffers();
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

void GPUMemoryPool::allocateBuffers() {
  size_t depth_size = width_ * height_ * sizeof(uint16_t);
  size_t pointcloud_xyz_size = width_ * height_ * 3 * sizeof(float);
  size_t rgb_size = width_ * height_ * 3 * sizeof(uint8_t);
  
  CUDA_CHECK(cudaMalloc(&d_depth_buffer_, depth_size));
  CUDA_CHECK(cudaMalloc(&d_pointcloud_buffer_, pointcloud_xyz_size * 2));
  CUDA_CHECK(cudaMalloc(&d_rgb_buffer_, rgb_size));
  CUDA_CHECK(cudaMalloc(&d_filtered_depth_buffer_, depth_size));
  
  CUDA_CHECK(cudaMemset(d_depth_buffer_, 0, depth_size));
  CUDA_CHECK(cudaMemset(d_pointcloud_buffer_, 0, pointcloud_xyz_size * 2));
  CUDA_CHECK(cudaMemset(d_rgb_buffer_, 0, rgb_size));
  CUDA_CHECK(cudaMemset(d_filtered_depth_buffer_, 0, depth_size));
}

void GPUMemoryPool::freeBuffers() {
  if (d_depth_buffer_) {
    cudaFree(d_depth_buffer_);
    d_depth_buffer_ = nullptr;
  }
  if (d_pointcloud_buffer_) {
    cudaFree(d_pointcloud_buffer_);
    d_pointcloud_buffer_ = nullptr;
  }
  if (d_rgb_buffer_) {
    cudaFree(d_rgb_buffer_);
    d_rgb_buffer_ = nullptr;
  }
  if (d_filtered_depth_buffer_) {
    cudaFree(d_filtered_depth_buffer_);
    d_filtered_depth_buffer_ = nullptr;
  }
}

void GPUMemoryPool::uploadDepth(const void* host_data, size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  CUDA_CHECK(cudaMemcpyAsync(d_depth_buffer_, host_data, size, 
                             cudaMemcpyHostToDevice, stream_));
}

void GPUMemoryPool::uploadRGB(const void* host_data, size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  CUDA_CHECK(cudaMemcpyAsync(d_rgb_buffer_, host_data, size, 
                             cudaMemcpyHostToDevice, stream_));
}

void GPUMemoryPool::downloadPointCloud(void* host_data, size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  CUDA_CHECK(cudaMemcpyAsync(host_data, d_pointcloud_buffer_, size, 
                             cudaMemcpyDeviceToHost, stream_));
}

void GPUMemoryPool::downloadFilteredDepth(void* host_data, size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  CUDA_CHECK(cudaMemcpyAsync(host_data, d_filtered_depth_buffer_, size, 
                             cudaMemcpyDeviceToHost, stream_));
}

void GPUMemoryPool::synchronize() {
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

size_t GPUMemoryPool::getTotalMemoryUsage() const {
  size_t depth_size = width_ * height_ * sizeof(uint16_t);
  size_t pointcloud_xyz_size = width_ * height_ * 3 * sizeof(float);
  size_t rgb_size = width_ * height_ * 3 * sizeof(uint8_t);
  
  return depth_size + (pointcloud_xyz_size * 2) + rgb_size + depth_size;
}

}
}
