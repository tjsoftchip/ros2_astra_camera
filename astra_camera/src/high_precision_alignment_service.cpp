#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <thread>
#include <chrono>

class HighPrecisionAlignmentService : public rclcpp::Node {
public:
    HighPrecisionAlignmentService() : Node("high_precision_alignment_service")
    {
        // 订阅ArUco位姿
        aruco_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/qr_code/pose", 10,
            std::bind(&HighPrecisionAlignmentService::aruco_callback, this, std::placeholders::_1));
        
        // 创建服务
        align_service_ = this->create_service<std_srvs::srv::Trigger>(
            "start_high_precision_alignment",
            std::bind(&HighPrecisionAlignmentService::align_callback, this, 
                     std::placeholders::_1, std::placeholders::_2));
        
        // 发布调试图像
        debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/alignment/debug_image", 10);
        
        // 参数
        this->declare_parameter("target_distance", 0.3);  // 目标距离30cm
        this->declare_parameter("lateral_tolerance", 0.02);  // 左右容差2cm
        this->declare_parameter("angular_tolerance", 2.0);  // 角度容差2度
        this->declare_parameter("max_alignment_time", 30.0);  // 最大对齐时间30秒
        
        target_distance_ = this->get_parameter("target_distance").as_double();
        lateral_tolerance_ = this->get_parameter("lateral_tolerance").as_double();
        angular_tolerance_ = this->get_parameter("angular_tolerance").as_double();
        max_alignment_time_ = this->get_parameter("max_alignment_time").as_double();
        
        RCLCPP_INFO(this->get_logger(), 
                   "高精度对齐服务已启动: 目标距离=%.2fm, 容差=%.2fm, %.1f°", 
                   target_distance_, lateral_tolerance_, angular_tolerance_);
    }
    
private:
    void aruco_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        latest_pose_ = msg;
        last_pose_time_ = this->now();
    }
    
    void align_callback(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        response->success = false;
        
        if (!latest_pose_) {
            response->message = "未检测到ArUco标记";
            return;
        }
        
        // 检查数据是否新鲜
        auto now = this->now();
        auto age = (now - last_pose_time_).seconds();
        if (age > 1.0) {
            response->message = "ArUco数据过期";
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "开始高精度对齐...");
        
        // 执行对齐
        bool aligned = perform_high_precision_alignment();
        
        if (aligned) {
            response->success = true;
            response->message = "高精度对齐完成！可以开始补给";
        } else {
            response->message = "对齐超时或失败";
        }
    }
    
    bool perform_high_precision_alignment()
    {
        auto start_time = this->now();
        
        while (rclcpp::ok()) {
            // 检查超时
            auto elapsed = (this->now() - start_time).seconds();
            if (elapsed > max_alignment_time_) {
                RCLCPP_WARN(this->get_logger(), "对齐超时（%.1f秒）", elapsed);
                return false;
            }
            
            if (!latest_pose_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            // 获取当前位姿
            double distance = latest_pose_->pose.position.z;
            double lateral_offset = latest_pose_->pose.position.x;
            
            // 计算角度偏差
            auto& q = latest_pose_->pose.orientation;
            double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
            double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
            double yaw = std::atan2(siny_cosp, cosy_cosp) * 180.0 / M_PI;
            
            // 创建调试图像
            create_debug_image(distance, lateral_offset, yaw);
            
            // 检查对齐状态
            bool distance_ok = std::abs(distance - target_distance_) < 0.02;  // 2cm容差
            bool lateral_ok = std::abs(lateral_offset) < lateral_tolerance_;
            bool angular_ok = std::abs(yaw) < angular_tolerance_;
            
            RCLCPP_INFO(this->get_logger(), 
                       "对齐状态: 距离=%.3fm(目标%.2fm), 偏移=%.3fm, 角度=%.1f°",
                       distance, target_distance_, lateral_offset, yaw);
            
            if (distance_ok && lateral_ok && angular_ok) {
                RCLCPP_INFO(this->get_logger(), 
                           "✅ 已精确对齐！距离误差=%.3fm, 偏移=%.3fm, 角度=%.1f°",
                           std::abs(distance - target_distance_), lateral_offset, yaw);
                return true;
            }
            
            // 输出调整建议
            if (!distance_ok) {
                if (distance > target_distance_ + 0.02) {
                    RCLCPP_INFO(this->get_logger(), "⚠️ 需要前进 %.3fm", distance - target_distance_);
                } else {
                    RCLCPP_INFO(this->get_logger(), "⚠️ 需要后退 %.3fm", target_distance_ - distance);
                }
            }
            
            if (!lateral_ok) {
                if (lateral_offset > 0) {
                    RCLCPP_INFO(this->get_logger(), "⚠️ 需要向左 %.3fm", lateral_offset);
                } else {
                    RCLCPP_INFO(this->get_logger(), "⚠️ 需要向右 %.3fm", -lateral_offset);
                }
            }
            
            if (!angular_ok) {
                if (yaw > 0) {
                    RCLCPP_INFO(this->get_logger(), "⚠️ 需要逆时针转 %.1f°", yaw);
                } else {
                    RCLCPP_INFO(this->get_logger(), "⚠️ 需要顺时针转 %.1f°", -yaw);
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        
        return false;
    }
    
    void create_debug_image(double distance, double lateral_offset, double yaw)
    {
        cv::Mat debug_img = cv::Mat::zeros(300, 600, CV_8UC3);
        
        // 绘制对齐状态
        cv::rectangle(debug_img, cv::Point(50, 50), cv::Point(550, 250), cv::Scalar(50, 50, 50), 2);
        
        // 距离指示
        double distance_ratio = distance / 1.0;  // 假设最大1米
        int distance_x = 50 + int(distance_ratio * 500);
        cv::circle(debug_img, cv::Point(distance_x, 100), 10, cv::Scalar(0, 255, 0), -1);
        cv::putText(debug_img, "Distance", cv::Point(50, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // 左右偏移指示
        int lateral_x = 300 + int(lateral_offset * 1000);  // 放大显示
        cv::circle(debug_img, cv::Point(lateral_x, 150), 10, cv::Scalar(0, 0, 255), -1);
        cv::line(debug_img, cv::Point(300, 140), cv::Point(300, 160), cv::Scalar(255, 255, 255), 2);
        cv::putText(debug_img, "Lateral", cv::Point(50, 140), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // 角度指示
        int angle_x = 300 + int(yaw * 10);  // 放大显示
        cv::circle(debug_img, cv::Point(angle_x, 200), 10, cv::Scalar(255, 0, 0), -1);
        cv::line(debug_img, cv::Point(300, 190), cv::Point(300, 210), cv::Scalar(255, 255, 255), 2);
        cv::putText(debug_img, "Angle", cv::Point(50, 190), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // 状态文本
        std::string status = "Aligning...";
        bool distance_ok = std::abs(distance - target_distance_) < 0.02;
        bool lateral_ok = std::abs(lateral_offset) < lateral_tolerance_;
        bool angular_ok = std::abs(yaw) < angular_tolerance_;
        
        if (distance_ok && lateral_ok && angular_ok) {
            status = "ALIGNED!";
        }
        
        cv::putText(debug_img, status, cv::Point(50, 280), cv::FONT_HERSHEY_SIMPLEX, 0.8, 
                   cv::Scalar(0, 255, 0), 2);
        
        // 发布调试图像
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", debug_img).toImageMsg();
        debug_image_pub_->publish(*msg);
    }
    
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr aruco_sub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr align_service_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
    
    geometry_msgs::msg::PoseStamped::SharedPtr latest_pose_;
    rclcpp::Time last_pose_time_;
    
    double target_distance_;
    double lateral_tolerance_;
    double angular_tolerance_;
    double max_alignment_time_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<HighPrecisionAlignmentService>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}