#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <memory>
#include "astra_camera/cuda/qr_detector_cuda.h"
#include "astra_camera/utils.h"
#include "astra_camera/dynamic_params.h"

namespace astra_camera {

class QRCodeDetectorNode {
public:
    explicit QRCodeDetectorNode(rclcpp::Node* node, std::shared_ptr<Parameters> parameters);
    ~QRCodeDetectorNode();

private:
    using Image = sensor_msgs::msg::Image;
    using CameraInfo = sensor_msgs::msg::CameraInfo;
    using PoseStamped = geometry_msgs::msg::PoseStamped;
    using Marker = visualization_msgs::msg::Marker;
    
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image, CameraInfo>;
    
    rclcpp::Node* node_;
    std::shared_ptr<Parameters> parameters_;
    rclcpp::Logger logger_;
    
    std::shared_ptr<message_filters::Subscriber<Image>> rgb_sub_;
    std::shared_ptr<message_filters::Subscriber<Image>> depth_sub_;
    std::shared_ptr<message_filters::Subscriber<CameraInfo>> camera_info_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    
    rclcpp::Publisher<PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<Marker>::SharedPtr marker_pub_;
    rclcpp::Publisher<Image>::SharedPtr debug_image_pub_;
    
    std::unique_ptr<cuda::QRDetectorCUDA> qr_detector_;
    
    float qr_size_;
    bool enable_debug_image_;
    bool enable_visualization_;
    
    void imageCb(
        const Image::ConstSharedPtr& rgb_msg,
        const Image::ConstSharedPtr& depth_msg,
        const CameraInfo::ConstSharedPtr& camera_info_msg);
    
    void publishPose(const cuda::QRCodeResult& result, const std_msgs::msg::Header& header);
    void publishMarker(const cuda::QRCodeResult& result, const std_msgs::msg::Header& header);
    void publishDebugImage(const cv::Mat& image, const cuda::QRCodeResult& result, 
                          const std_msgs::msg::Header& header);
};

}
