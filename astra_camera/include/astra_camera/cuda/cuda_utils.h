#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "   \
                << cudaGetErrorString(error) << std::endl;                     \
      throw std::runtime_error(cudaGetErrorString(error));                    \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_LAST_ERROR()                                                \
  do {                                                                         \
    cudaError_t error = cudaGetLastError();                                    \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "   \
                << cudaGetErrorString(error) << std::endl;                     \
      throw std::runtime_error(cudaGetErrorString(error));                    \
    }                                                                          \
  } while (0)

namespace astra_camera {
namespace cuda {

struct CameraIntrinsics {
  float fx;
  float fy;
  float cx;
  float cy;
};

inline void printGPUInfo() {
  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  
  if (deviceCount == 0) {
    std::cerr << "No CUDA devices found!" << std::endl;
    return;
  }
  
  for (int dev = 0; dev < deviceCount; ++dev) {
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    
    std::cout << "CUDA Device " << dev << ": " << deviceProp.name << std::endl;
    std::cout << "  Compute Capability: " << deviceProp.major << "." 
              << deviceProp.minor << std::endl;
    std::cout << "  Total Global Memory: " 
              << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "  CUDA Cores: " << deviceProp.multiProcessorCount * 128 << std::endl;
    std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "  Clock Rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
  }
}

}
}
