#include "astra_camera/aruco_detector_node.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <opencv2/opencv.hpp>

namespace astra_camera {

ArUcoDetectorNode::ArUcoDetectorNode(rclcpp::Node* node, std::shared_ptr<Parameters> parameters)
    : node_(node),
      parameters_(std::move(parameters)),
      logger_(node->get_logger()),
      marker_size_(0.1685f),
      enable_debug_image_(true),
      enable_visualization_(true),
      enable_preprocessing_(true),
      dictionary_id_(10) // DICT_6X6_250 = 10
{
    // 启用参数设置
    setAndGetNodeParameter(parameters_, marker_size_, "aruco_marker_size", 0.1685f);
    setAndGetNodeParameter(parameters_, enable_debug_image_, "enable_aruco_debug_image", true);
    setAndGetNodeParameter(parameters_, enable_visualization_, "enable_aruco_visualization", true);
    setAndGetNodeParameter(parameters_, enable_preprocessing_, "enable_aruco_preprocessing", true);
    setAndGetNodeParameter(parameters_, dictionary_id_, "aruco_dictionary_id", 10);
    
    aruco_detector_ = std::make_unique<cuda::ArUcoDetectorCUDA>(marker_size_);
    aruco_detector_->setDictionary(dictionary_id_);
    aruco_detector_->setDebugMode(enable_debug_image_);
    
    RCLCPP_INFO(logger_, "Creating subscribers...");
    rgb_sub_ = std::make_shared<message_filters::Subscriber<Image>>(
        node_, "camera/color/image_raw", rmw_qos_profile_sensor_data);
    depth_sub_ = std::make_shared<message_filters::Subscriber<Image>>(
        node_, "camera/depth/image_raw", rmw_qos_profile_sensor_data);
    camera_info_sub_ = std::make_shared<message_filters::Subscriber<CameraInfo>>(
        node_, "camera/color/camera_info", rmw_qos_profile_sensor_data);
    RCLCPP_INFO(logger_, "Subscribers created");
    
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), *rgb_sub_, *depth_sub_, *camera_info_sub_);
    sync_->registerCallback(std::bind(&ArUcoDetectorNode::imageCb, this,
                                     std::placeholders::_1,
                                     std::placeholders::_2,
                                     std::placeholders::_3));
    RCLCPP_INFO(logger_, "Synchronizer created and callback registered");
    
    pose_pub_ = node_->create_publisher<PoseStamped>("qr_code/pose", 10);
    marker_pub_ = node_->create_publisher<Marker>("qr_code/marker", 10);
    debug_image_pub_ = node_->create_publisher<Image>("qr_code/debug_image", 10);
    
    RCLCPP_INFO(logger_, "ArUco Detector Node initialized (Marker size: %.3f m, Dictionary: 6x6_250)", marker_size_);
}

ArUcoDetectorNode::~ArUcoDetectorNode() {
}

void ArUcoDetectorNode::imageCb(
    const Image::ConstSharedPtr& rgb_msg,
    const Image::ConstSharedPtr& depth_msg,
    const CameraInfo::ConstSharedPtr& camera_info_msg)
{
    static int count = 0;
    count++;
    
    if (count % 10 == 0) {
        RCLCPP_INFO(logger_, "Image callback called %d times", count);
    }
    
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
        
        cuda::ArUcoResult result;
        bool detected = aruco_detector_->detectAndDecode(
            rgb_ptr->image,
            depth_ptr->image,
            camera_matrix,
            dist_coeffs,
            result);
        
        if (detected) {
            RCLCPP_INFO(logger_, "ArUco marker detected: ID=%d, Distance=%.3f m, Angle=%.1f deg, Confidence=%.2f",
                       result.marker_id, result.distance, result.angle, result.confidence);
            
            publishPose(result, rgb_msg->header);
            
            if (enable_visualization_) {
                publishMarker(result, rgb_msg->header);
            }
            
            if (enable_debug_image_) {
                publishDebugImage(rgb_ptr->image, result, rgb_msg->header);
            }
        } else {
            if (count % 30 == 0) {
                RCLCPP_INFO(logger_, "No ArUco marker detected (callback %d)", count);
            }
        }
        
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(logger_, "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "Exception in ArUco detection: %s", e.what());
    }
}

void ArUcoDetectorNode::publishPose(const cuda::ArUcoResult& result, 
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

void ArUcoDetectorNode::publishMarker(const cuda::ArUcoResult& result,
                                       const std_msgs::msg::Header& header)
{
    auto marker = std::make_unique<Marker>();
    marker->header = header;
    marker->ns = "qr_code";
    marker->id = result.marker_id;
    marker->type = Marker::CUBE;
    marker->action = Marker::ADD;
    
    if (!result.tvec.empty()) {
        marker->pose.position.x = result.tvec.at<double>(0);
        marker->pose.position.y = result.tvec.at<double>(1);
        marker->pose.position.z = result.tvec.at<double>(2);
    }
    
    marker->scale.x = marker_size_;
    marker->scale.y = marker_size_;
    marker->scale.z = 0.01;
    
    marker->color.r = 0.0f;
    marker->color.g = 1.0f;
    marker->color.b = 0.0f;
    marker->color.a = 0.8f;
    
    marker->lifetime = rclcpp::Duration::from_seconds(0.5);
    
    marker_pub_->publish(std::move(marker));
}

void ArUcoDetectorNode::publishDebugImage(const cv::Mat& image, 
                                          const cuda::ArUcoResult& result,
                                          const std_msgs::msg::Header& header)
{
    cv::Mat debug_image = image.clone();
    
    if (result.corners.size() == 4) {
        // 绘制ArUco标记边框
        for (size_t i = 0; i < 4; i++) {
            cv::line(debug_image, result.corners[i], result.corners[(i + 1) % 4], 
                    cv::Scalar(0, 255, 0), 3);
        }
        
        // 绘制中心点
        cv::Point2f center(0, 0);
        for (const auto& corner : result.corners) {
            center.x += corner.x;
            center.y += corner.y;
        }
        center.x /= 4.0f;
        center.y /= 4.0f;
        
        cv::circle(debug_image, center, 5, cv::Scalar(0, 0, 255), -1);
        
        // 添加文本信息
        std::string text = "ID:" + std::to_string(result.marker_id) + 
                          " | " + std::to_string(static_cast<int>(result.distance * 1000)) + "mm | " +
                          std::to_string(static_cast<int>(result.angle)) + "deg | " +
                          std::to_string(static_cast<int>(result.confidence * 100)) + "%";
        cv::putText(debug_image, text, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        // 添加预处理状态
        if (enable_preprocessing_) {
            cv::putText(debug_image, "Preprocessing: ON", cv::Point(10, 60), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
        }
    }
    
    auto debug_msg = cv_bridge::CvImage(header, "bgr8", debug_image).toImageMsg();
    debug_image_pub_->publish(*debug_msg);
}

}  // namespace astra_camera