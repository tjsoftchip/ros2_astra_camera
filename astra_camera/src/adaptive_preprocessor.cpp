#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <vector>
#include <numeric>

namespace astra_camera {

class AdaptivePreprocessor : public rclcpp::Node

{

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
        this->declare_parameter("evaluation_interval", 30);  // 每30帧评估一次
        this->declare_parameter("brightness_threshold", 50);   // 亮度阈值
        this->declare_parameter("contrast_threshold", 30);     // 对比度阈值
        
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
            // 暗图像：增加C值
            adaptive_c_ = std::min(15.0, adaptive_c_ + 2.0);
            RCLCPP_INFO(this->get_logger(), "检测到暗图像，调整C值为: %.1f", adaptive_c_);
        } else if (brightness > 200) {
            // 亮图像：减少C值
            adaptive_c_ = std::max(2.0, adaptive_c_ - 2.0);
            RCLCPP_INFO(this->get_logger(), "检测到亮图像，调整C值为: %.1f", adaptive_c_);
        }
        
        // 根据对比度调整块大小
        if (contrast < 30) {
            // 低对比度：增加块大小
            adaptive_block_size_ = std::min(31, adaptive_block_size_ + 2);
            // 确保块大小为奇数
            if (adaptive_block_size_ % 2 == 0) adaptive_block_size_++;
            RCLCPP_INFO(this->get_logger(), "检测到低对比度，调整块大小为: %d", adaptive_block_size_);
        } else if (contrast > 80) {
            // 高对比度：减少块大小
            adaptive_block_size_ = std::max(7, adaptive_block_size_ - 2);
            // 确保块大小为奇数
            if (adaptive_block_size_ % 2 == 0) adaptive_block_size_++;
            RCLCPP_INFO(this->get_logger(), "检测到高对比度，调整块大小为: %d", adaptive_block_size_);
        }
        
        // 根据噪声水平调整模糊核大小
        double noise_level = estimate_noise_level(gray);
        if (noise_level > 10) {
            blur_kernel_size_ = std::min(5, blur_kernel_size_ + 1);
            RCLCPP_INFO(this->get_logger(), "检测到高噪声，调整模糊核大小为: %d", blur_kernel_size_);
        } else if (noise_level < 5) {
            blur_kernel_size_ = std::max(3, blur_kernel_size_ - 1);
            RCLCPP_INFO(this->get_logger(), "检测到低噪声，调整模糊核大小为: %d", blur_kernel_size_);
        }
    }
    
    double estimate_noise_level(const cv::Mat& gray)
    {
        // 使用拉普拉斯算子估计噪声
        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_64F);
        
        cv::Scalar mu, sigma;
        cv::meanStdDev(laplacian, mu, sigma);
        
        return sigma[0];
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

} // namespace astra_camera

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(astra_camera::AdaptivePreprocessor)