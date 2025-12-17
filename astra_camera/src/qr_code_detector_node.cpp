#include "astra_camera/qr_code_detector_node.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <opencv2/opencv.hpp>

namespace astra_camera {

QRCodeDetectorNode::QRCodeDetectorNode(rclcpp::Node* node, std::shared_ptr<Parameters> parameters)
    : node_(node),
      parameters_(std::move(parameters)),
      logger_(node->get_logger()),
      qr_size_(0.1f),
      enable_debug_image_(true),
      enable_visualization_(true)
{
    setAndGetNodeParameter(parameters_, qr_size_, "qr_code_size", 0.1f);
    setAndGetNodeParameter(parameters_, enable_debug_image_, "enable_qr_debug_image", true);
    setAndGetNodeParameter(parameters_, enable_visualization_, "enable_qr_visualization", true);
    
    qr_detector_ = std::make_unique<cuda::QRDetectorCUDA>(qr_size_);
    
    rgb_sub_ = std::make_shared<message_filters::Subscriber<Image>>(
        node_, "color/image_raw", rmw_qos_profile_sensor_data);
    depth_sub_ = std::make_shared<message_filters::Subscriber<Image>>(
        node_, "depth/image_raw", rmw_qos_profile_sensor_data);
    camera_info_sub_ = std::make_shared<message_filters::Subscriber<CameraInfo>>(
        node_, "color/camera_info", rmw_qos_profile_sensor_data);
    
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), *rgb_sub_, *depth_sub_, *camera_info_sub_);
    sync_->registerCallback(std::bind(&QRCodeDetectorNode::imageCb, this,
                                     std::placeholders::_1,
                                     std::placeholders::_2,
                                     std::placeholders::_3));
    
    pose_pub_ = node_->create_publisher<PoseStamped>("qr_code/pose", 10);
    marker_pub_ = node_->create_publisher<Marker>("qr_code/marker", 10);
    debug_image_pub_ = node_->create_publisher<Image>("qr_code/debug_image", 10);
    
    RCLCPP_INFO(logger_, "QR Code Detector Node initialized (QR size: %.3f m)", qr_size_);
}

QRCodeDetectorNode::~QRCodeDetectorNode() {
}

void QRCodeDetectorNode::imageCb(
    const Image::ConstSharedPtr& rgb_msg,
    const Image::ConstSharedPtr& depth_msg,
    const CameraInfo::ConstSharedPtr& camera_info_msg)
{
    try {
        cv_bridge::CvImageConstPtr rgb_ptr = cv_bridge::toCvShare(rgb_msg, "bgr8");
        cv_bridge::CvImageConstPtr depth_ptr = cv_bridge::toCvShare(depth_msg, "16UC1");
        
        cv::Mat camera_matrix = cv::Mat(3, 3, CV_64F);
        camera_matrix.at<double>(0, 0) = camera_info_msg->k[0];
        camera_matrix.at<double>(0, 1) = camera_info_msg->k[1];
        camera_matrix.at<double>(0, 2) = camera_info_msg->k[2];
        camera_matrix.at<double>(1, 0) = camera_info_msg->k[3];
        camera_matrix.at<double>(1, 1) = camera_info_msg->k[4];
        camera_matrix.at<double>(1, 2) = camera_info_msg->k[5];
        camera_matrix.at<double>(2, 0) = camera_info_msg->k[6];
        camera_matrix.at<double>(2, 1) = camera_info_msg->k[7];
        camera_matrix.at<double>(2, 2) = camera_info_msg->k[8];
        
        cv::Mat dist_coeffs = cv::Mat(5, 1, CV_64F);
        for (int i = 0; i < 5; i++) {
            dist_coeffs.at<double>(i, 0) = camera_info_msg->d[i];
        }
        
        cuda::QRCodeResult result;
        bool detected = qr_detector_->detectAndDecode(
            rgb_ptr->image,
            depth_ptr->image,
            camera_matrix,
            dist_coeffs,
            result);
        
        if (detected) {
            RCLCPP_INFO(logger_, "QR Code detected: '%s', Distance: %.3f m, Angle: %.1f deg",
                       result.data.c_str(), result.distance, result.angle);
            
            publishPose(result, rgb_msg->header);
            
            if (enable_visualization_) {
                publishMarker(result, rgb_msg->header);
            }
            
            if (enable_debug_image_) {
                publishDebugImage(rgb_ptr->image, result, rgb_msg->header);
            }
        }
        
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(logger_, "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "Exception in QR detection: %s", e.what());
    }
}

void QRCodeDetectorNode::publishPose(const cuda::QRCodeResult& result, 
                                    const std_msgs::msg::Header& header)
{
    auto pose_msg = std::make_unique<PoseStamped>();
    pose_msg->header = header;
    
    if (!result.tvec.empty() && !result.rvec.empty()) {
        pose_msg->pose.position.x = result.tvec.at<double>(0);
        pose_msg->pose.position.y = result.tvec.at<double>(1);
        pose_msg->pose.position.z = result.tvec.at<double>(2);
        
        cv::Mat rotation_matrix;
        cv::Rodrigues(result.rvec, rotation_matrix);
        
        tf2::Matrix3x3 tf_rotation(
            rotation_matrix.at<double>(0, 0), rotation_matrix.at<double>(0, 1), rotation_matrix.at<double>(0, 2),
            rotation_matrix.at<double>(1, 0), rotation_matrix.at<double>(1, 1), rotation_matrix.at<double>(1, 2),
            rotation_matrix.at<double>(2, 0), rotation_matrix.at<double>(2, 1), rotation_matrix.at<double>(2, 2)
        );
        
        tf2::Quaternion quaternion;
        tf_rotation.getRotation(quaternion);
        
        pose_msg->pose.orientation.x = quaternion.x();
        pose_msg->pose.orientation.y = quaternion.y();
        pose_msg->pose.orientation.z = quaternion.z();
        pose_msg->pose.orientation.w = quaternion.w();
    }
    
    pose_pub_->publish(std::move(pose_msg));
}

void QRCodeDetectorNode::publishMarker(const cuda::QRCodeResult& result,
                                       const std_msgs::msg::Header& header)
{
    auto marker = std::make_unique<Marker>();
    marker->header = header;
    marker->ns = "qr_code";
    marker->id = 0;
    marker->type = Marker::CUBE;
    marker->action = Marker::ADD;
    
    if (!result.tvec.empty()) {
        marker->pose.position.x = result.tvec.at<double>(0);
        marker->pose.position.y = result.tvec.at<double>(1);
        marker->pose.position.z = result.tvec.at<double>(2);
    }
    
    marker->scale.x = qr_size_;
    marker->scale.y = qr_size_;
    marker->scale.z = 0.01;
    
    marker->color.r = 0.0f;
    marker->color.g = 1.0f;
    marker->color.b = 0.0f;
    marker->color.a = 0.8f;
    
    marker->lifetime = rclcpp::Duration::from_seconds(0.5);
    
    marker_pub_->publish(std::move(marker));
}

void QRCodeDetectorNode::publishDebugImage(const cv::Mat& image, 
                                          const cuda::QRCodeResult& result,
                                          const std_msgs::msg::Header& header)
{
    cv::Mat debug_image = image.clone();
    
    if (result.corners.size() == 4) {
        for (size_t i = 0; i < 4; i++) {
            cv::line(debug_image, result.corners[i], result.corners[(i + 1) % 4], 
                    cv::Scalar(0, 255, 0), 3);
        }
        
        cv::Point2f center(0, 0);
        for (const auto& corner : result.corners) {
            center.x += corner.x;
            center.y += corner.y;
        }
        center.x /= 4.0f;
        center.y /= 4.0f;
        
        cv::circle(debug_image, center, 5, cv::Scalar(0, 0, 255), -1);
        
        std::string text = result.data + " | " + 
                          std::to_string(static_cast<int>(result.distance * 1000)) + "mm | " +
                          std::to_string(static_cast<int>(result.angle)) + "deg";
        cv::putText(debug_image, text, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }
    
    auto debug_msg = cv_bridge::CvImage(header, "bgr8", debug_image).toImageMsg();
    debug_image_pub_->publish(*debug_msg);
}

}
