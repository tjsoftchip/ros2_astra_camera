#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <cv_bridge/cv_bridge.h>
#include <deque>
#include <unordered_map>
#include <chrono>

namespace astra_camera {

struct MarkerTrack {
    int id;
    std::deque<geometry_msgs::msg::PoseStamped> poses;
    double confidence;
    rclcpp::Time last_seen;
    bool is_active;
    
    MarkerTrack() : confidence(0.0), is_active(false) {}
};

class MultiMarkerTracker : public rclcpp::Node {
public:
    MultiMarkerTracker() : Node("multi_marker_tracker")
    {
        // 订阅单个ArUco位姿
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/qr_code/pose", 10,
            std::bind(&MultiMarkerTracker::pose_callback, this, std::placeholders::_1));
        
        // 发布多标记位姿
        multi_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
            "/multi_marker/poses", 10);
        
        // 发布RViz标记数组
        marker_array_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/multi_marker/markers", 10);
        
        // 参数
        this->declare_parameter("max_track_history", 10);
        this->declare_parameter("marker_timeout", 2.0);
        this->declare_parameter("min_confidence", 0.5);
        
        max_track_history_ = this->get_parameter("max_track_history").as_int();
        marker_timeout_ = this->get_parameter("marker_timeout").as_double();
        min_confidence_ = this->get_parameter("min_confidence").as_double();
        
        // 定时器：发布处理后的数据
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&MultiMarkerTracker::timer_callback, this));
        
        RCLCPP_INFO(this->get_logger(), "多标记跟踪器已启动");
    }
    
private:
    void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        // 从调试图像或单独话题获取标记ID
        // 这里简化处理，假设只有一个标记
        int marker_id = 200; // 默认ID
        
        // 更新或创建轨迹
        auto& track = marker_tracks_[marker_id];
        track.id = marker_id;
        track.poses.push_back(*msg);
        track.last_seen = this->now();
        track.is_active = true;
        
        // 限制历史长度
        while (track.poses.size() > static_cast<size_t>(max_track_history_)) {
            track.poses.pop_front();
        }
        
        // 计算置信度
        update_confidence(track);
    }
    
    void update_confidence(MarkerTrack& track)
    {
        if (track.poses.empty()) {
            track.confidence = 0.0;
            return;
        }
        
        // 基于位置稳定性计算置信度
        double position_variance = 0.0;
        if (track.poses.size() > 1) {
            const auto& last = track.poses.back().pose.position;
            const auto& prev = track.poses[track.poses.size()-2].pose.position;
            
            double dx = last.x - prev.x;
            double dy = last.y - prev.y;
            double dz = last.z - prev.z;
            
            position_variance = sqrt(dx*dx + dy*dy + dz*dz);
        }
        
        // 置信度计算：稳定性越高，置信度越高
        track.confidence = std::exp(-position_variance * 10.0);
        
        // 考虑时间因素
        auto age = (this->now() - track.last_seen).seconds();
        track.confidence *= std::exp(-age);
    }
    
    void timer_callback()
    {
        // 检查并清理超时的标记
        cleanup_old_markers();
        
        // 发布多标记位姿
        publish_multi_marker_poses();
        
        // 发布RViz标记
        publish_marker_array();
    }
    
    void cleanup_old_markers()
    {
        auto now = this->now();
        for (auto it = marker_tracks_.begin(); it != marker_tracks_.end();) {
            if ((now - it->second.last_seen).seconds() > marker_timeout_) {
                it->second.is_active = false;
                RCLCPP_INFO(this->get_logger(), "标记 %d 超时，停止跟踪", it->first);
                // 保留轨迹信息一段时间，以便重新检测时快速恢复
                if (it->second.poses.size() > 20) {
                    it = marker_tracks_.erase(it);
                } else {
                    ++it;
                }
            } else {
                ++it;
            }
        }
    }
    
    void publish_multi_marker_poses()
    {
        auto msg = std::make_unique<geometry_msgs::msg::PoseArray>();
        msg->header.stamp = this->now();
        msg->header.frame_id = "camera_color_optical_frame";
        
        for (const auto& [id, track] : marker_tracks_) {
            if (track.is_active && track.confidence > min_confidence_) {
                geometry_msgs::msg::Pose pose;
                
                if (track.poses.size() > 1) {
                    // 平滑处理
                    const auto& last = track.poses.back().pose;
                    const auto& prev = track.poses[track.poses.size()-2].pose;
                    
                    pose.position.x = 0.7 * last.position.x + 0.3 * prev.position.x;
                    pose.position.y = 0.7 * last.position.y + 0.3 * prev.position.y;
                    pose.position.z = 0.7 * last.position.z + 0.3 * prev.position.z;
                    pose.orientation = last.orientation;
                } else {
                    pose = track.poses.back().pose;
                }
                
                msg->poses.push_back(pose);
            }
        }
        
        multi_pose_pub_->publish(std::move(msg));
    }
    
    void publish_marker_array()
    {
        auto marker_array = std::make_unique<visualization_msgs::msg::MarkerArray>();
        
        int marker_id = 0;
        for (const auto& [id, track] : marker_tracks_) {
            if (track.is_active && track.confidence > min_confidence_) {
                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = "camera_color_optical_frame";
                marker.header.stamp = this->now();
                marker.ns = "aruco_markers";
                marker.id = marker_id++;
                marker.type = visualization_msgs::msg::Marker::CUBE;
                marker.action = visualization_msgs::msg::Marker::ADD;
                
                // 设置位置
                const auto& pose = track.poses.back().pose;
                marker.pose.position = pose.position;
                marker.pose.orientation = pose.orientation;
                
                // 设置尺寸（根据标记大小调整）
                marker.scale.x = 0.1685;
                marker.scale.y = 0.1685;
                marker.scale.z = 0.01;
                
                // 设置颜色（根据置信度）
                marker.color.a = track.confidence;
                marker.color.r = 0.0;
                marker.color.g = 1.0;
                marker.color.b = 0.0;
                
                // 设置其他属性
                marker.lifetime = rclcpp::Duration::from_seconds(1.0);
                
                marker_array->markers.push_back(marker);
                
                // 添加文本标记
                visualization_msgs::msg::Marker text_marker;
                text_marker.header = marker.header;
                text_marker.ns = "aruco_markers_text";
                text_marker.id = marker_id++;
                text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                text_marker.action = visualization_msgs::msg::Marker::ADD;
                
                text_marker.pose = marker.pose;
                text_marker.pose.position.z += 0.1;
                text_marker.scale.x = 0.05;
                text_marker.scale.y = 0.05;
                text_marker.scale.z = 0.05;
                
                text_marker.color.a = 1.0;
                text_marker.color.r = 1.0;
                text_marker.color.g = 1.0;
                text_marker.color.b = 1.0;
                
                text_marker.text = "ID " + std::to_string(id);
                text_marker.lifetime = rclcpp::Duration::from_seconds(1.0);
                
                marker_array->markers.push_back(text_marker);
            }
        }
        
        marker_array_pub_->publish(std::move(marker_array));
    }
    
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr multi_pose_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::unordered_map<int, MarkerTrack> marker_tracks_;
    
    int max_track_history_;
    double marker_timeout_;
    double min_confidence_;
};

class AdaptivePreprocessor : public rclcpp::Node {
public:
    AdaptivePreprocessor(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("adaptive_preprocessor", options)
    {
        // 订阅原始图像
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw", 10,
            std::bind(&AdaptivePreprocessor::image_callback, this, std::placeholders::_1));
        
        // 发布预处理后的图像
        processed_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/camera/processed/image_raw", 10);
        
        // 发布调试信息
        debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/adaptive_preprocessor/debug", 10);
        
        // 参数
        this->declare_parameter("enable_adaptive", true);
        this->declare_parameter("evaluation_interval", 30);
        this->declare_parameter("brightness_threshold", 50);
        this->declare_parameter("contrast_threshold", 30);
        
        enable_adaptive_ = this->get_parameter("enable_adaptive").as_bool();
        evaluation_interval_ = this->get_parameter("evaluation_interval").as_int();
        brightness_threshold_ = this->get_parameter("brightness_threshold").as_int();
        contrast_threshold_ = this->get_parameter("contrast_threshold").as_int();
        
        // 预处理参数
        adaptive_block_size_ = 15;
        adaptive_c_ = 5.0;
        morph_kernel_size_ = 3;
        blur_kernel_size_ = 3;
        
        frame_count_ = 0;
        
        RCLCPP_INFO(this->get_logger(), "自适应预处理器已启动");
    }
    
private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            // 转换图像
            cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            
            // 自适应评估
            if (enable_adaptive_ && frame_count_ % evaluation_interval_ == 0) {
                evaluate_and_adjust(gray);
            }
            
            // 执行预处理
            cv::Mat processed = preprocess_image(gray);
            
            // 发布结果
            auto processed_msg = cv_bridge::CvImage(msg->header, "mono8", processed).toImageMsg();
            processed_pub_->publish(*processed_msg);
            
            // 发布调试图像（每10帧一次）
            if (frame_count_ % 10 == 0) {
                publish_debug_image(gray, processed);
            }
            
            frame_count_++;
            
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge异常: %s", e.what());
        }
    }
    
    void evaluate_and_adjust(const cv::Mat& gray)
    {
        // 计算图像统计信息
        cv::Scalar mean, stddev;
        cv::meanStdDev(gray, mean, stddev);
        
        double brightness = mean[0];
        double contrast = stddev[0];
        
        RCLCPP_INFO(this->get_logger(), 
                   "图像评估 - 亮度: %.1f, 对比度: %.1f", 
                   brightness, contrast);
        
        // 根据亮度调整参数
        if (brightness < 50) {
            adaptive_c_ = std::min(15.0, adaptive_c_ + 2.0);
            RCLCPP_INFO(this->get_logger(), "检测到暗图像，调整C值为: %.1f", adaptive_c_);
        } else if (brightness > 200) {
            adaptive_c_ = std::max(2.0, adaptive_c_ - 2.0);
            RCLCPP_INFO(this->get_logger(), "检测到亮图像，调整C值为: %.1f", adaptive_c_);
        }
        
        // 根据对比度调整块大小
        if (contrast < 30) {
            adaptive_block_size_ = std::min(31, adaptive_block_size_ + 2);
            if (adaptive_block_size_ % 2 == 0) adaptive_block_size_++;
            RCLCPP_INFO(this->get_logger(), "检测到低对比度，调整块大小为: %d", adaptive_block_size_);
        } else if (contrast > 80) {
            adaptive_block_size_ = std::max(7, adaptive_block_size_ - 2);
            if (adaptive_block_size_ % 2 == 0) adaptive_block_size_++;
            RCLCPP_INFO(this->get_logger(), "检测到高对比度，调整块大小为: %d", adaptive_block_size_);
        }
    }
    
    cv::Mat preprocess_image(const cv::Mat& gray)
    {
        cv::Mat result;
        
        // 1. 高斯模糊降噪
        if (blur_kernel_size_ > 1) {
            cv::GaussianBlur(gray, result, cv::Size(blur_kernel_size_, blur_kernel_size_), 0);
        } else {
            result = gray.clone();
        }
        
        // 2. 自适应阈值
        cv::adaptiveThreshold(result, result, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv::THRESH_BINARY, adaptive_block_size_, adaptive_c_);
        
        // 3. 形态学操作
        if (morph_kernel_size_ > 0) {
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                    cv::Size(morph_kernel_size_, morph_kernel_size_));
            cv::morphologyEx(result, result, cv::MORPH_OPEN, kernel);
        }
        
        return result;
    }
    
    void publish_debug_image(const cv::Mat& gray, const cv::Mat& processed)
    {
        cv::Mat debug;
        cv::cvtColor(gray, debug, cv::COLOR_GRAY2BGR);
        
        // 在左侧显示预处理结果
        cv::Mat processed_color;
        cv::cvtColor(processed, processed_color, cv::COLOR_GRAY2BGR);
        
        // 合并图像
        cv::hconcat(debug, processed_color, debug);
        
        // 添加文本信息
        std::string info = cv::format("Block: %d, C: %.1f, Blur: %d", 
                                   adaptive_block_size_, adaptive_c_, blur_kernel_size_);
        cv::putText(debug, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                   0.7, cv::Scalar(0, 255, 0), 2);
        
        // 发布调试图像
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", debug).toImageMsg();
        debug_pub_->publish(*msg);
    }
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
    
    bool enable_adaptive_;
    int evaluation_interval_;
    int brightness_threshold_;
    int contrast_threshold_;
    
    // 动态参数
    int adaptive_block_size_;
    double adaptive_c_;
    int morph_kernel_size_;
    int blur_kernel_size_;
    
    int frame_count_;
};

class PoseOptimizer : public rclcpp::Node {
public:
    PoseOptimizer(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("pose_optimizer", options)
    {
        // 订阅ArUco位姿
        aruco_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/qr_code/pose", 10,
            std::bind(&PoseOptimizer::aruco_callback, this, std::placeholders::_1));
        
        // 发布优化后的位姿
        optimized_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/aruco/optimized_pose", 10);
        
        // 发布里程计
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/aruco/odom", 10);
        
        // 参数
        this->declare_parameter("window_size", 10);
        this->declare_parameter("min_measurements", 3);
        this->declare_parameter("max_velocity", 1.0);
        this->declare_parameter("enable_kalman", true);
        this->declare_parameter("process_noise", 0.1);
        this->declare_parameter("measurement_noise", 0.05);
        
        window_size_ = this->get_parameter("window_size").as_int();
        min_measurements_ = this->get_parameter("min_measurements").as_int();
        max_velocity_ = this->get_parameter("max_velocity").as_double();
        enable_kalman_ = this->get_parameter("enable_kalman").as_bool();
        process_noise_ = this->get_parameter("process_noise").as_double();
        measurement_noise_ = this->get_parameter("measurement_noise").as_double();
        
        // 初始化卡尔曼滤波器
        init_kalman_filter();
        
        // 初始化成员变量
        has_odom_ = false;
        last_pose_ = geometry_msgs::msg::PoseWithCovarianceStamped();
        
        // 定时器：发布优化结果
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&PoseOptimizer::timer_callback, this));
        
        RCLCPP_INFO(this->get_logger(), "位姿优化器已启动");
    }
    
private:
    void aruco_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        // 添加到测量队列
        measurements_.push_back(PoseMeasurement(*msg, 1.0));
        
        // 限制队列长度
        while (measurements_.size() > static_cast<size_t>(window_size_)) {
            measurements_.pop_front();
        }
        
        last_aruco_time_ = this->now();
    }
    
    void init_kalman_filter()
    {
        // 状态向量: [x, y, z, vx, vy, vz, qx, qy, qz, qw]
        kalman_.init(10, 6, 0);
        
        // 转移矩阵
        cv::setIdentity(kalman_.transitionMatrix);
        float dt = 0.05; // 20Hz
        for (int i = 0; i < 3; i++) {
            kalman_.transitionMatrix.at<float>(i, i+3) = dt;
        }
        
        // 测量矩阵
        kalman_.measurementMatrix = cv::Mat::zeros(6, 10, CV_32F);
        for (int i = 0; i < 6; i++) {
            kalman_.measurementMatrix.at<float>(i, i) = 1.0f;
        }
        
        // 过程噪声协方差
        cv::setIdentity(kalman_.processNoiseCov, cv::Scalar::all(process_noise_));
        
        // 测量噪声协方差
        cv::setIdentity(kalman_.measurementNoiseCov, cv::Scalar::all(measurement_noise_));
        
        // 误差协方差
        cv::setIdentity(kalman_.errorCovPost, cv::Scalar::all(1.0));
        
        // 初始状态
        kalman_.statePost = cv::Mat::zeros(10, 1, CV_32F);
    }
    
    void timer_callback()
    {
        if (measurements_.size() < static_cast<size_t>(min_measurements_)) {
            return;
        }
        
        // 执行位姿优化
        geometry_msgs::msg::PoseWithCovarianceStamped optimized_pose;
        if (optimize_pose(optimized_pose)) {
            // 发布优化后的位姿
            optimized_pose_pub_->publish(optimized_pose);
            
            // 发布里程计
            publish_odometry(optimized_pose);
        }
    }
    
    bool optimize_pose(geometry_msgs::msg::PoseWithCovarianceStamped& optimized_pose)
    {
        if (enable_kalman_) {
            return optimize_with_kalman(optimized_pose);
        } else {
            return optimize_with_average(optimized_pose);
        }
    }
    
    bool optimize_with_kalman(geometry_msgs::msg::PoseWithCovarianceStamped& optimized_pose)
    {
        // 使用最新的测量进行预测和更新
        const auto& latest = measurements_.back();
        
        // 预测
        cv::Mat prediction = kalman_.predict();
        
        // 更新
        cv::Mat measurement = cv::Mat::zeros(6, 1, CV_32F);
        measurement.at<float>(0) = latest.pose.pose.position.x;
        measurement.at<float>(1) = latest.pose.pose.position.y;
        measurement.at<float>(2) = latest.pose.pose.position.z;
        measurement.at<float>(3) = latest.pose.pose.orientation.x;
        measurement.at<float>(4) = latest.pose.pose.orientation.y;
        measurement.at<float>(5) = latest.pose.pose.orientation.z;
        
        cv::Mat estimated = kalman_.correct(measurement);
        
        // 提取优化后的位姿
        optimized_pose.header.stamp = this->now();
        optimized_pose.header.frame_id = "camera_color_optical_frame";
        
        optimized_pose.pose.pose.position.x = estimated.at<float>(0);
        optimized_pose.pose.pose.position.y = estimated.at<float>(1);
        optimized_pose.pose.pose.position.z = estimated.at<float>(2);
        
        // 重构四元数（简化处理）
        optimized_pose.pose.pose.orientation = latest.pose.pose.orientation;
        
        // 设置协方差
        optimized_pose.pose.covariance = {
            0.01, 0, 0, 0, 0, 0,
            0, 0.01, 0, 0, 0, 0,
            0, 0, 0.01, 0, 0, 0,
            0, 0, 0, 0, 0.01, 0,
            0, 0, 0, 0, 0, 0.01
        };
        
        return true;
    }
    
    bool optimize_with_average(geometry_msgs::msg::PoseWithCovarianceStamped& optimized_pose)
    {
        // 简单的平均滤波
        double sum_x = 0, sum_y = 0, sum_z = 0;
        double sum_qx = 0, sum_qy = 0, sum_qz = 0, sum_qw = 0;
        double total_weight = 0;
        
        for (const auto& m : measurements_) {
            double weight = m.confidence;
            sum_x += m.pose.pose.position.x * weight;
            sum_y += m.pose.pose.position.y * weight;
            sum_z += m.pose.pose.position.z * weight;
            sum_qx += m.pose.pose.orientation.x * weight;
            sum_qy += m.pose.pose.orientation.y * weight;
            sum_qz += m.pose.pose.orientation.z * weight;
            sum_qw += m.pose.pose.orientation.w * weight;
            total_weight += weight;
        }
        
        if (total_weight == 0) return false;
        
        optimized_pose.header.stamp = this->now();
        optimized_pose.header.frame_id = "camera_color_optical_frame";
        
        optimized_pose.pose.pose.position.x = sum_x / total_weight;
        optimized_pose.pose.pose.position.y = sum_y / total_weight;
        optimized_pose.pose.pose.position.z = sum_z / total_weight;
        
        // 归一化四元数
        double norm = sqrt(sum_qx*sum_qx + sum_qy*sum_qy + sum_qz*sum_qz + sum_qw*sum_qw);
        optimized_pose.pose.pose.orientation.x = sum_qx / norm;
        optimized_pose.pose.pose.orientation.y = sum_qy / norm;
        optimized_pose.pose.pose.orientation.z = sum_qz / norm;
        optimized_pose.pose.pose.orientation.w = sum_qw / norm;
        
        // 计算协方差
        double var_x = 0, var_y = 0, var_z = 0;
        for (const auto& m : measurements_) {
            double dx = m.pose.pose.position.x - optimized_pose.pose.pose.position.x;
            double dy = m.pose.pose.position.y - optimized_pose.pose.pose.position.y;
            double dz = m.pose.pose.position.z - optimized_pose.pose.pose.position.z;
            var_x += dx * dx;
            var_y += dy * dy;
            var_z += dz * dz;
        }
        
        int n = measurements_.size();
        optimized_pose.pose.covariance[0] = var_x / n;
        optimized_pose.pose.covariance[7] = var_y / n;
        optimized_pose.pose.covariance[14] = var_z / n;
        
        return true;
    }
    
    void publish_odometry(const geometry_msgs::msg::PoseWithCovarianceStamped& optimized_pose)
    {
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = optimized_pose.header.stamp;
        odom.header.frame_id = "camera_color_optical_frame";
        odom.child_frame_id = "aruco_base";
        
        odom.pose.pose = optimized_pose.pose.pose;
        odom.pose.covariance = optimized_pose.pose.covariance;
        
        // 如果有里程计数据，计算速度
        if (has_odom_ && last_pose_.header.stamp.sec != 0) {
            double dt = (rclcpp::Time(optimized_pose.header.stamp) - rclcpp::Time(last_pose_.header.stamp)).seconds();
            if (dt > 0) {
                odom.twist.twist.linear.x = (optimized_pose.pose.pose.position.x - last_pose_.pose.pose.position.x) / dt;
                odom.twist.twist.linear.y = (optimized_pose.pose.pose.position.y - last_pose_.pose.pose.position.y) / dt;
                odom.twist.twist.linear.z = (optimized_pose.pose.pose.position.z - last_pose_.pose.pose.position.z) / dt;
            }
        }
        
        odom_pub_->publish(odom);
        last_pose_ = optimized_pose;
        has_odom_ = true;
    }
    
    struct PoseMeasurement {
        geometry_msgs::msg::PoseStamped pose;
        double confidence;
        rclcpp::Time timestamp;
        
        PoseMeasurement(const geometry_msgs::msg::PoseStamped& p, double conf)
            : pose(p), confidence(conf), timestamp(p.header.stamp) {}
    };
    
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr aruco_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr optimized_pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::deque<PoseMeasurement> measurements_;
    geometry_msgs::msg::PoseWithCovarianceStamped last_pose_;
    nav_msgs::msg::Odometry last_odom_;
    
    cv::KalmanFilter kalman_;
    rclcpp::Time last_aruco_time_;
    
    int window_size_;
    int min_measurements_;
    double max_velocity_;
    bool enable_kalman_;
    double process_noise_;
    double measurement_noise_;
    bool has_odom_ = false;
};

} // namespace astra_camera

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    // 创建多标记跟踪器
    auto multi_tracker = std::make_shared<astra_camera::MultiMarkerTracker>();
    
    // 创建自适应预处理器
    auto preprocessor = std::make_shared<astra_camera::AdaptivePreprocessor>();
    
    // 创建位姿优化器
    auto optimizer = std::make_shared<astra_camera::PoseOptimizer>();
    
    // 使用多线程执行器
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(multi_tracker);
    executor.add_node(preprocessor);
    executor.add_node(optimizer);
    
    executor.spin();
    rclcpp::shutdown();
    return 0;
}