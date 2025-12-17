#pragma once

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <memory>
#include "depth_traits.h"
#include "astra_camera/utils.h"
#include "astra_camera/dynamic_params.h"
#include "astra_camera/cuda/gpu_memory_pool.h"
#include "astra_camera/cuda/depth_to_pointcloud.h"
#include "astra_camera/cuda/bilateral_filter.h"

namespace astra_camera {

namespace enc = sensor_msgs::image_encodings;

class PointCloudXyzCudaNode {
 public:
  explicit PointCloudXyzCudaNode(rclcpp::Node* const node, std::shared_ptr<Parameters> parameters);
  ~PointCloudXyzCudaNode();

 private:
  using PointCloud2 = sensor_msgs::msg::PointCloud2;
  using Image = sensor_msgs::msg::Image;
  using CameraInfo = sensor_msgs::msg::CameraInfo;
  
  rclcpp::Node* const node_;
  std::shared_ptr<Parameters> parameters_;
  rclcpp::Logger logger_ = rclcpp::get_logger("PointCloudXyzCudaNode");
  
  image_transport::CameraSubscriber sub_depth_;
  int queue_size_ = 5;
  rmw_qos_profile_t point_cloud_qos_profile_ = rmw_qos_profile_sensor_data;
  rmw_qos_profile_t depth_qos_profile_ = rmw_qos_profile_sensor_data;

  std::mutex connect_mutex_;
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_point_cloud_;

  image_geometry::PinholeCameraModel model_;
  
  std::unique_ptr<cuda::GPUMemoryPool> gpu_memory_pool_;
  cuda::CameraIntrinsics intrinsics_;
  float depth_scale_;
  
  bool initialized_;
  int image_width_;
  int image_height_;
  
  bool enable_depth_filter_;
  float depth_filter_spatial_sigma_;
  float depth_filter_range_sigma_;

  void connectCb();
  void depthCb(const Image::ConstSharedPtr& depth_msg, const CameraInfo::ConstSharedPtr& info_msg);
  void initializeGPU(int width, int height);
  void updateIntrinsics(const CameraInfo::ConstSharedPtr& info_msg);
};

}
