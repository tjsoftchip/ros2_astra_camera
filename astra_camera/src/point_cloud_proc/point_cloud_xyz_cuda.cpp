#include "astra_camera/point_cloud_proc/point_cloud_xyz_cuda.h"
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <chrono>

namespace astra_camera {

PointCloudXyzCudaNode::PointCloudXyzCudaNode(rclcpp::Node *const node,
                                             std::shared_ptr<Parameters> parameters)
    : node_(node), 
      parameters_(std::move(parameters)),
      initialized_(false),
      image_width_(0),
      image_height_(0),
      depth_scale_(0.001f),
      enable_depth_filter_(true),
      depth_filter_spatial_sigma_(2.0f),
      depth_filter_range_sigma_(50.0f)
{
  setAndGetNodeParameter<int>(parameters_, queue_size_, "queue_size", 5);
  std::string point_cloud_qos;
  std::string depth_qos;
  setAndGetNodeParameter<std::string>(parameters_, point_cloud_qos, "point_cloud_qos", "default");
  setAndGetNodeParameter<std::string>(parameters_, depth_qos, "depth_qos", "default");
  depth_qos_profile_ = getRMWQosProfileFromString(depth_qos);
  
  RCLCPP_INFO(logger_, "Initializing CUDA-accelerated point cloud node");
  
  cuda::printGPUInfo();
  
  connectCb();

  std::scoped_lock<decltype(connect_mutex_)> lock(connect_mutex_);
  point_cloud_qos_profile_ = getRMWQosProfileFromString(point_cloud_qos);
  pub_point_cloud_ = node_->create_publisher<PointCloud2>(
      "depth/points", rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(point_cloud_qos_profile_),
                                  point_cloud_qos_profile_));
}

PointCloudXyzCudaNode::~PointCloudXyzCudaNode() {
  gpu_memory_pool_.reset();
  RCLCPP_INFO(logger_, "CUDA point cloud node destroyed");
}

void PointCloudXyzCudaNode::initializeGPU(int width, int height) {
  if (initialized_ && image_width_ == width && image_height_ == height) {
    return;
  }
  
  RCLCPP_INFO(logger_, "Initializing GPU memory pool for %dx%d images", width, height);
  
  try {
    gpu_memory_pool_ = std::make_unique<cuda::GPUMemoryPool>(width, height);
    image_width_ = width;
    image_height_ = height;
    initialized_ = true;
    
    RCLCPP_INFO(logger_, "GPU initialization successful");
  } catch (const std::exception& e) {
    RCLCPP_ERROR(logger_, "Failed to initialize GPU: %s", e.what());
    initialized_ = false;
    throw;
  }
}

void PointCloudXyzCudaNode::updateIntrinsics(const CameraInfo::ConstSharedPtr& info_msg) {
  model_.fromCameraInfo(info_msg);
  
  intrinsics_.fx = model_.fx();
  intrinsics_.fy = model_.fy();
  intrinsics_.cx = model_.cx();
  intrinsics_.cy = model_.cy();
}

void PointCloudXyzCudaNode::connectCb() {
  std::scoped_lock<decltype(connect_mutex_)> lock(connect_mutex_);
  if (!sub_depth_) {
    auto custom_qos = depth_qos_profile_;
    custom_qos.depth = queue_size_;

    sub_depth_ = image_transport::create_camera_subscription(
        node_, "depth/image_raw",
        [this](const sensor_msgs::msg::Image::ConstSharedPtr &msg,
               const sensor_msgs::msg::CameraInfo::ConstSharedPtr &info) { depthCb(msg, info); },
        "raw", custom_qos);
  }
}

void PointCloudXyzCudaNode::depthCb(const Image::ConstSharedPtr &depth_msg,
                                    const CameraInfo::ConstSharedPtr &info_msg) {
  auto start_time = std::chrono::high_resolution_clock::now();
  
  if (depth_msg->encoding != enc::TYPE_16UC1) {
    RCLCPP_ERROR(logger_, "Unsupported depth encoding: %s (expected TYPE_16UC1)", 
                 depth_msg->encoding.c_str());
    return;
  }
  
  int width = depth_msg->width;
  int height = depth_msg->height;
  
  if (!initialized_ || image_width_ != width || image_height_ != height) {
    try {
      initializeGPU(width, height);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(logger_, "GPU initialization failed, skipping frame");
      return;
    }
  }
  
  updateIntrinsics(info_msg);
  
  try {
    size_t depth_size = width * height * sizeof(uint16_t);
    gpu_memory_pool_->uploadDepth(depth_msg->data.data(), depth_size);
    
    const uint16_t* depth_input = static_cast<const uint16_t*>(gpu_memory_pool_->getDepthBuffer());
    
    if (enable_depth_filter_) {
      cuda::launchBilateralFilter(
          depth_input,
          static_cast<uint16_t*>(gpu_memory_pool_->getFilteredDepthBuffer()),
          width,
          height,
          depth_filter_spatial_sigma_,
          depth_filter_range_sigma_,
          gpu_memory_pool_->getStream()
      );
      depth_input = static_cast<const uint16_t*>(gpu_memory_pool_->getFilteredDepthBuffer());
    }
    
    cuda::launchDepthToPointCloudXYZ(
        depth_input,
        static_cast<float*>(gpu_memory_pool_->getPointCloudBuffer()),
        intrinsics_,
        width,
        height,
        depth_scale_,
        gpu_memory_pool_->getStream()
    );
    
    auto cloud_msg = std::make_shared<PointCloud2>();
    cloud_msg->header = depth_msg->header;
    cloud_msg->height = height;
    cloud_msg->width = width;
    cloud_msg->is_dense = false;
    cloud_msg->is_bigendian = false;
    
    sensor_msgs::PointCloud2Modifier pcd_modifier(*cloud_msg);
    pcd_modifier.setPointCloud2FieldsByString(1, "xyz");
    
    size_t pointcloud_size = width * height * 3 * sizeof(float);
    gpu_memory_pool_->downloadPointCloud(cloud_msg->data.data(), pointcloud_size);
    gpu_memory_pool_->synchronize();
    
    pub_point_cloud_->publish(*cloud_msg);
    
    static bool first_publish = true;
    if (first_publish) {
      RCLCPP_INFO(logger_, "CUDA Point Cloud node started successfully");
      first_publish = false;
    }
    
  } catch (const std::exception& e) {
    RCLCPP_ERROR(logger_, "CUDA processing error: %s", e.what());
  }
}

}
