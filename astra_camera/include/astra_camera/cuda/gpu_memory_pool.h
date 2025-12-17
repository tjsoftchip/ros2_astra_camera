#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include "cuda_utils.h"

namespace astra_camera {
namespace cuda {

class GPUMemoryPool {
public:
  GPUMemoryPool(int width, int height);
  ~GPUMemoryPool();

  GPUMemoryPool(const GPUMemoryPool&) = delete;
  GPUMemoryPool& operator=(const GPUMemoryPool&) = delete;

  void* getDepthBuffer() { return d_depth_buffer_; }
  void* getPointCloudBuffer() { return d_pointcloud_buffer_; }
  void* getRGBBuffer() { return d_rgb_buffer_; }
  void* getFilteredDepthBuffer() { return d_filtered_depth_buffer_; }
  
  cudaStream_t getStream() { return stream_; }
  
  void uploadDepth(const void* host_data, size_t size);
  void uploadRGB(const void* host_data, size_t size);
  void downloadPointCloud(void* host_data, size_t size);
  void downloadFilteredDepth(void* host_data, size_t size);
  
  void synchronize();
  
  int getWidth() const { return width_; }
  int getHeight() const { return height_; }
  size_t getTotalMemoryUsage() const;

private:
  int width_;
  int height_;
  
  void* d_depth_buffer_;
  void* d_pointcloud_buffer_;
  void* d_rgb_buffer_;
  void* d_filtered_depth_buffer_;
  
  cudaStream_t stream_;
  std::mutex mutex_;
  
  void allocateBuffers();
  void freeBuffers();
};

}
}
