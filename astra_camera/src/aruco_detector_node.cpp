#include "astra_camera/aruco_detector_node.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace astra_camera {

ArUcoDetectorNode::ArUcoDetectorNode(rclcpp::Node* node, std::shared_ptr<Parameters> parameters)
    : node_(node),
      parameters_(std::move(parameters)),
      logger_(node->get_logger()),
      marker_size_(0.57f),
      enable_debug_image_(true),
      enable_visualization_(true),
      enable_preprocessing_(true),
      dictionary_id_(0), // DICT_4X4_50 = 0
      frame_counter_(0),
      process_every_n_frames_(5)
{
    // 启用参数设置
    setAndGetNodeParameter(parameters_, marker_size_, "aruco_marker_size", 0.57f);
    setAndGetNodeParameter(parameters_, enable_debug_image_, "enable_aruco_debug_image", true);
    setAndGetNodeParameter(parameters_, enable_visualization_, "enable_aruco_visualization", true);
    setAndGetNodeParameter(parameters_, enable_preprocessing_, "enable_aruco_preprocessing", false);
    setAndGetNodeParameter(parameters_, dictionary_id_, "aruco_dictionary_id", 0);

    // 新增误报过滤参数 (白名单默认只接受 ID=0 和 ID=1)
    std::vector<int64_t> whitelist_default = {0, 1};
    setAndGetNodeParameter(parameters_, whitelist_default, "aruco_id_whitelist", whitelist_default);
    for (auto v : whitelist_default) id_whitelist_.push_back(static_cast<int>(v));

    setAndGetNodeParameter(parameters_, min_consecutive_frames_, "aruco_min_consecutive_frames", 3);
    setAndGetNodeParameter(parameters_, min_marker_area_pixels_, "aruco_min_marker_area_pixels", 100);
    setAndGetNodeParameter(parameters_, process_every_n_frames_, "aruco_process_every_n_frames", 5);

    aruco_detector_ = std::make_unique<cuda::ArUcoDetectorCUDA>(marker_size_);
    aruco_detector_->setDictionary(dictionary_id_);
    aruco_detector_->setDebugMode(enable_debug_image_);
    aruco_detector_->setIdWhitelist(id_whitelist_);
    aruco_detector_->setRequiredConsecutiveFrames(min_consecutive_frames_);
    aruco_detector_->setMinMarkerAreaPixels(min_marker_area_pixels_);

    // 通过 Parameters 系统注册动态参数回调（消除 "can not be changed in runtime" 警告）
    parameters_->setParam("aruco_dictionary_id", rclcpp::ParameterValue(dictionary_id_),
        [this](const rclcpp::Parameter &p) {
            int new_id = p.as_int();
            if (new_id != dictionary_id_) {
                aruco_detector_->setDictionary(new_id);
                dictionary_id_ = new_id;
                RCLCPP_INFO(logger_, "Dictionary switched to %d", new_id);
            }
        });
    parameters_->setParam("aruco_marker_size", rclcpp::ParameterValue(static_cast<double>(marker_size_)),
        [this](const rclcpp::Parameter &p) {
            float new_size = static_cast<float>(p.as_double());
            if (std::fabs(new_size - marker_size_) > 0.001f) {
                marker_size_ = new_size;
                aruco_detector_->setMarkerSize(new_size);
                RCLCPP_INFO(logger_, "Marker size changed to %.3f m", marker_size_);
            }
        });

    RCLCPP_INFO(logger_, "Creating subscribers...");
    rgb_sub_ = std::make_shared<message_filters::Subscriber<Image>>(
        node_, "camera/color/image_raw", rmw_qos_profile_sensor_data);
    camera_info_sub_ = std::make_shared<message_filters::Subscriber<CameraInfo>>(
        node_, "camera/color/camera_info", rmw_qos_profile_sensor_data);
    depth_sub_ = node_->create_subscription<Image>(
        "camera/depth/image_raw", rclcpp::SensorDataQoS(),
        std::bind(&ArUcoDetectorNode::depthCb, this, std::placeholders::_1));
    RCLCPP_INFO(logger_, "Subscribers created");
    
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), *rgb_sub_, *camera_info_sub_);
    sync_->registerCallback(std::bind(&ArUcoDetectorNode::imageCb, this,
                                     std::placeholders::_1,
                                     std::placeholders::_2));
    RCLCPP_INFO(logger_, "Synchronizer created and callback registered");
    
    pose_pub_ = node_->create_publisher<PoseStamped>("qr_code/pose", 10);
    marker_id_pub_ = node_->create_publisher<std_msgs::msg::Int32>("qr_code/marker_id", 10);
    marker_pub_ = node_->create_publisher<Marker>("qr_code/marker", 10);
    debug_image_pub_ = node_->create_publisher<Image>("qr_code/debug_image", 10);
    
    RCLCPP_INFO(logger_, "ArUco Detector Node initialized (Marker size: %.3f m, Dictionary: %d)", marker_size_, dictionary_id_);
    RCLCPP_INFO(logger_, "  - Frame skip: every %d frame (%.1f fps → %.1f fps)", process_every_n_frames_, 30.0f, 30.0f / process_every_n_frames_);
    RCLCPP_INFO(logger_, "  - Anti-false-positive filters:");
    RCLCPP_INFO(logger_, "    errorCorrectionRate: 0.3 (reduced)");
    RCLCPP_INFO(logger_, "    min_consecutive_frames: %d", min_consecutive_frames_);
    RCLCPP_INFO(logger_, "    min_marker_area: %d px", min_marker_area_pixels_);
    if (!id_whitelist_.empty()) {
        std::string ids;
        for (int id : id_whitelist_) ids += std::to_string(id) + " ";
        RCLCPP_INFO(logger_, "    id_whitelist: %s", ids.c_str());
    } else {
        RCLCPP_INFO(logger_, "    id_whitelist: ALL (disabled)");
    }
}

ArUcoDetectorNode::~ArUcoDetectorNode() {
}

void ArUcoDetectorNode::depthCb(const Image::ConstSharedPtr& depth_msg)
{
    try {
        cv_bridge::CvImageConstPtr depth_ptr = cv_bridge::toCvShare(depth_msg, "16UC1");
        std::lock_guard<std::mutex> lock(depth_mutex_);
        latest_depth_ = depth_ptr->image.clone();
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(logger_, "cv_bridge exception in depthCb: %s", e.what());
    }
}

void ArUcoDetectorNode::imageCb(
    const Image::ConstSharedPtr& rgb_msg,
    const CameraInfo::ConstSharedPtr& camera_info_msg)
{
    static int count = 0;
    count++;
    frame_counter_++;
    
    if (count % 3000 == 0) {
        RCLCPP_INFO(logger_, "Image callback called %d times", count);
    }
    
    // 跳帧处理: 仅每 process_every_n_frames_ 帧执行一次检测（默认5，约6fps）
    // 视觉伺服控制频率通常为10-20Hz，6fps完全满足对齐精度要求
    // 可大幅降低CPU占用（79.6% → ~16%）
    if (frame_counter_ % process_every_n_frames_ != 0) {
        return;
    }
    
    try {
        cv_bridge::CvImageConstPtr rgb_ptr = cv_bridge::toCvShare(rgb_msg, "bgr8");
        
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
        
        cv::Mat depth_for_detection;
        {
            std::lock_guard<std::mutex> lock(depth_mutex_);
            depth_for_detection = latest_depth_.clone();
        }
        
        cuda::ArUcoResult result;
        bool detected = aruco_detector_->detectAndDecode(
            rgb_ptr->image,
            depth_for_detection,
            camera_matrix,
            dist_coeffs,
            result);
        
        if (detected) {
            // 发布 marker ID
            auto id_msg = std::make_unique<std_msgs::msg::Int32>();
            id_msg->data = result.marker_id;
            marker_id_pub_->publish(std::move(id_msg));

            RCLCPP_DEBUG(logger_, "ArUco marker detected: ID=%d, Distance=%.3f m, Angle=%.1f deg, Confidence=%.2f",
                       result.marker_id, result.distance, result.angle, result.confidence);
            
            publishPose(result, rgb_msg->header);
            
            if (enable_visualization_) {
                publishMarker(result, rgb_msg->header);
            }
            
            if (enable_debug_image_) {
                publishDebugImage(rgb_ptr->image, result, rgb_msg->header);
            }
        } else {
            if (count % 3000 == 0) {
                RCLCPP_DEBUG(logger_, "No ArUco marker detected (callback %d)", count);
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