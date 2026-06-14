#include "astra_camera/cuda/aruco_detector_cuda.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>

namespace astra_camera {
namespace cuda {

ArUcoDetectorCUDA::ArUcoDetectorCUDA(float marker_size)
    : marker_size_(marker_size),
      debug_mode_(false),
      dictionary_id_(cv::aruco::DICT_6X6_250)
{
    dictionary_ = cv::aruco::getPredefinedDictionary(dictionary_id_);
    parameters_ = cv::aruco::DetectorParameters::create();
    
    // 优化参数以提高检测速度
    parameters_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    parameters_->cornerRefinementWinSize = 3;
    parameters_->cornerRefinementMaxIterations = 30;
    parameters_->cornerRefinementMinAccuracy = 0.01;
    parameters_->errorCorrectionRate = 0.6;
    parameters_->detectInvertedMarker = false;
}

ArUcoDetectorCUDA::~ArUcoDetectorCUDA() {
}

void ArUcoDetectorCUDA::setCameraParams(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs) {
    camera_matrix_ = camera_matrix.clone();
    dist_coeffs_ = dist_coeffs.clone();
}

void ArUcoDetectorCUDA::setDictionary(int dict_id) {
    dictionary_id_ = dict_id;
    dictionary_ = cv::aruco::getPredefinedDictionary(dict_id);
}

float ArUcoDetectorCUDA::getDepthAtPoint(const cv::Mat& depth_image, const cv::Point2f& point) {
    int x = static_cast<int>(point.x + 0.5f);
    int y = static_cast<int>(point.y + 0.5f);
    
    if (x < 0 || x >= depth_image.cols || y < 0 || y >= depth_image.rows) {
        return 0.0f;
    }
    
    int radius = 3;
    float sum = 0.0f;
    int count = 0;
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < depth_image.cols && ny >= 0 && ny < depth_image.rows) {
                uint16_t depth = depth_image.at<uint16_t>(ny, nx);
                if (depth > 0 && depth < 10000) {
                    sum += depth * 0.001f;
                    count++;
                }
            }
        }
    }
    
    return (count > 0) ? (sum / count) : 0.0f;
}

bool ArUcoDetectorCUDA::estimatePoseWithDepth(
    const std::vector<cv::Point2f>& corners,
    const cv::Mat& depth_image,
    cv::Mat& rvec,
    cv::Mat& tvec,
    float& distance,
    float& angle,
    float& confidence)
{
    if (corners.size() != 4) return false;
    
    std::vector<cv::Point3f> object_points;
    object_points.push_back(cv::Point3f(-marker_size_/2, marker_size_/2, 0));
    object_points.push_back(cv::Point3f(marker_size_/2, marker_size_/2, 0));
    object_points.push_back(cv::Point3f(marker_size_/2, -marker_size_/2, 0));
    object_points.push_back(cv::Point3f(-marker_size_/2, -marker_size_/2, 0));
    
    cv::Point2f center(0, 0);
    for (const auto& corner : corners) {
        center.x += corner.x;
        center.y += corner.y;
    }
    center.x /= 4.0f;
    center.y /= 4.0f;
    
    distance = getDepthAtPoint(depth_image, center);
    
    // 深度仅用于置信度打分，不用作PnP的gate:
    // 1) Astra深度相机在>5m时数据不可靠，但PnP(基于RGB角点)在9m内仍有效
    // 2) 去掉硬阈值后PnP正常求解，深度一致性由下面的confidence公式自动降权
    // 3) 下游aruco_pose_estimator的max_valid_distance=9.0m做最终距离过滤
    
    // 使用PnP求解位姿（不依赖深度）
    bool success = cv::solvePnP(
        object_points,
        corners,
        camera_matrix_,
        dist_coeffs_,
        rvec,
        tvec,
        false,
        cv::SOLVEPNP_ITERATIVE);
    
    if (success) {
        // 计算角度
        cv::Mat rotation_matrix;
        cv::Rodrigues(rvec, rotation_matrix);
        
        angle = atan2(rotation_matrix.at<double>(1, 0), 
                     rotation_matrix.at<double>(0, 0)) * 180.0 / CV_PI;
        
        // 计算置信度（基于角点质量和深度一致性）
        confidence = 1.0f;
        
        // 检查深度一致性
        float projected_z = tvec.at<double>(2);
        float depth_diff = std::abs(distance - projected_z);
        confidence *= std::max(0.0f, 1.0f - depth_diff / projected_z);
        
        // 检查重投影误差
        std::vector<cv::Point2f> projected_corners;
        cv::projectPoints(object_points, rvec, tvec, camera_matrix_, dist_coeffs_, projected_corners);
        
        float reprojection_error = 0.0f;
        for (size_t i = 0; i < 4; i++) {
            float dx = corners[i].x - projected_corners[i].x;
            float dy = corners[i].y - projected_corners[i].y;
            reprojection_error += sqrt(dx*dx + dy*dy);
        }
        reprojection_error /= 4.0f;
        
        confidence *= std::max(0.0f, 1.0f - reprojection_error / 10.0f);
    }
    
    return success;
}

void ArUcoDetectorCUDA::preprocessImage(
    const cv::Mat& input,
    cv::Mat& output,
    bool enable_adaptive_threshold,
    bool enable_morphology)
{
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    
    // 自适应阈值（CPU实现，因为CUDA版本可能不可用）
    if (enable_adaptive_threshold) {
        cv::adaptiveThreshold(gray, output, 255, cv::ADAPTIVE_THRESH_MEAN_C, 15, 5);
    } else {
        output = gray.clone();
    }
    
    // 形态学操作
    if (enable_morphology) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(output, output, cv::MORPH_OPEN, kernel);
    } else {
        if (!enable_adaptive_threshold) {
            output = gray.clone();
        }
    }
}

bool ArUcoDetectorCUDA::detectAndDecode(
    const cv::Mat& rgb_image,
    const cv::Mat& depth_image,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    ArUcoResult& result)
{
    result.detected = false;
    
    if (rgb_image.empty()) return false;
    
    setCameraParams(camera_matrix, dist_coeffs);
    
    // 预处理图像
    cv::Mat processed_image;
    preprocessImage(rgb_image, processed_image, true, true);
    
    // 检测ArUco标记
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    
    cv::Ptr<cv::aruco::Dictionary> dict_ptr = cv::makePtr<cv::aruco::Dictionary>(dictionary_);
    cv::aruco::detectMarkers(processed_image, dict_ptr, marker_corners, marker_ids, parameters_);
    
    if (marker_ids.empty()) {
        return false;
    }
    
    // 选择最大的标记（假设只有一个标记）
    int best_idx = 0;
    float max_area = 0.0f;
    
    for (size_t i = 0; i < marker_corners.size(); i++) {
        float area = cv::contourArea(marker_corners[i]);
        if (area > max_area) {
            max_area = area;
            best_idx = i;
        }
    }
    
    result.detected = true;
    result.marker_id = marker_ids[best_idx];
    result.corners = marker_corners[best_idx];
    result.data = std::to_string(marker_ids[best_idx]);
    
    // 姿态估计
    if (!depth_image.empty() && result.corners.size() == 4) {
        estimatePoseWithDepth(
            result.corners,
            depth_image,
            result.rvec,
            result.tvec,
            result.distance,
            result.angle,
            result.confidence);
    }
    
    return true;
}

}
}