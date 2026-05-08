#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <memory>
#include "cuda_utils.h"
#include "aruco_preprocessor.h"

namespace astra_camera {
namespace cuda {

struct ArUcoResult {
    bool detected;
    int marker_id;
    std::string data;
    std::vector<cv::Point2f> corners;
    cv::Mat rvec;
    cv::Mat tvec;
    float distance;
    float angle;
    float confidence;
};

class ArUcoDetectorCUDA {
public:
    ArUcoDetectorCUDA(float marker_size = 0.1f);
    ~ArUcoDetectorCUDA();
    
    bool detectAndDecode(
        const cv::Mat& rgb_image,
        const cv::Mat& depth_image,
        const cv::Mat& camera_matrix,
        const cv::Mat& dist_coeffs,
        ArUcoResult& result);
    
    void setCameraParams(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);
    void setMarkerSize(float size) { marker_size_ = size; }
    void setDictionary(int dict_id);
    void setDebugMode(bool enable) { debug_mode_ = enable; }
    void setIdWhitelist(const std::vector<int>& whitelist) { id_whitelist_ = whitelist; }
    void setRequiredConsecutiveFrames(int frames) { required_consecutive_frames_ = frames; }
    void setMinMarkerAreaPixels(int area) { min_marker_area_pixels_ = area; }
    
    // GPU预处理功能
    void preprocessImage(
        const cv::Mat& input,
        cv::Mat& output,
        bool enable_adaptive_threshold = true,
        bool enable_morphology = true);
    
private:
    float marker_size_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    cv::Ptr<cv::aruco::DetectorParameters> parameters_;
    
    bool debug_mode_;
    int dictionary_id_;

    // 误报过滤参数
    std::vector<int> id_whitelist_;           // ID白名单，空表示接受所有
    int required_consecutive_frames_ = 3;     // 需要连续检测到的帧数
    int consecutive_count_ = 0;               // 当前连续计数
    int last_detected_id_ = -1;               // 上一帧检测到的ID
    int min_marker_area_pixels_ = 100;        // 最小marker面积（像素）

    // GPU内存管理
    uint8_t* d_input_;
    uint8_t* d_output_;
    uint8_t* d_temp1_;
    uint8_t* d_temp2_;
    size_t image_size_;
    cudaStream_t stream_;
    
    bool estimatePoseWithDepth(
        const std::vector<cv::Point2f>& corners,
        const cv::Mat& depth_image,
        cv::Mat& rvec,
        cv::Mat& tvec,
        float& distance,
        float& angle,
        float& confidence);
    
    float getDepthAtPoint(const cv::Mat& depth_image, const cv::Point2f& point);
    
    // GPU内存初始化和释放
    void initGPUMemory(int width, int height);
    void freeGPUMemory();
    
    // GPU预处理函数
    void gpuAdaptiveThreshold(
        const cv::Mat& input,
        cv::Mat& output,
        double max_value,
        int adaptive_method,
        int threshold_type,
        int block_size,
        double C);
    
    void gpuMorphology(
        const cv::Mat& input,
        cv::Mat& output,
        int operation,
        cv::Mat kernel);
};

}
}