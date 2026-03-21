#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <opencv2/opencv.hpp>
#include <deque>
#include <vector>
#include <memory>

namespace astra_camera {

struct PoseMeasurement {
    geometry_msgs::msg::PoseStamped pose;
    double confidence;
    rclcpp::Time timestamp;
    
    PoseMeasurement(const geometry_msgs::msg::PoseStamped& p, double conf)
        : pose(p), confidence(conf), timestamp(p.header.stamp) {}
};

class PoseOptimizer : public rclcpp::Node

{

public:

    PoseOptimizer(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("pose_optimizer", options)
    {
        // 订阅ArUco位姿
        aruco_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/qr_code/pose", 10,
            std::bind(&PoseOptimizer::aruco_callback, this, std::placeholders::_1));
        
        // 订阅里程计（如果有）
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&PoseOptimizer::odom_callback, this, std::placeholders::_1));
        
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
    
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        last_odom_ = *msg;
        has_odom_ = true;
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
            0, 0, 0, 0.01, 0, 0,
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
    
    void publish_odometry(const geometry_msgs::msg::PoseWithCovarianceStamped& pose)
    {
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = pose.header.stamp;
        odom.header.frame_id = "camera_color_optical_frame";
        odom.child_frame_id = "aruco_base";
        
        odom.pose.pose = pose.pose.pose;
        odom.pose.covariance = pose.pose.covariance;
        
        // 如果有里程计数据，计算速度
        if (has_odom_ && !last_pose_.pose.pose.position.x == 0) {
            double dt = (rclcpp::Time(pose.header.stamp) - rclcpp::Time(last_pose_.header.stamp)).seconds();
            if (dt > 0) {
                odom.twist.twist.linear.x = (pose.pose.pose.position.x - last_pose_.pose.pose.position.x) / dt;
                odom.twist.twist.linear.y = (pose.pose.pose.position.y - last_pose_.pose.pose.position.y) / dt;
                odom.twist.twist.linear.z = (pose.pose.pose.position.z - last_pose_.pose.pose.position.z) / dt;
            }
        }
        
        odom_pub_->publish(odom);
        last_pose_ = pose;
    }
    
    
    
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr aruco_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
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

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(astra_camera::PoseOptimizer)