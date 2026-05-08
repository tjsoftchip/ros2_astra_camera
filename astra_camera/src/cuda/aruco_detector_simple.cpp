#include "astra_camera/cuda/aruco_detector_cuda.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>

namespace astra_camera {
namespace cuda {

ArUcoDetectorCUDA::ArUcoDetectorCUDA(float marker_size)
    : marker_size_(marker_size),
      debug_mode_(false),
      dictionary_id_(cv::aruco::DICT_6X6_250),
      d_input_(nullptr),
      d_output_(nullptr),
      d_temp1_(nullptr),
      d_temp2_(nullptr),
      image_size_(0)
{
    dictionary_ = cv::makePtr<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(dictionary_id_));
    parameters_ = cv::makePtr<cv::aruco::DetectorParameters>();
    
    // 使用默认参数（不要修改）
    // parameters_->adaptiveThreshWinSizeMin = 3;
    // parameters_->adaptiveThreshWinSizeMax = 23;
    // parameters_->adaptiveThreshWinSizeStep = 10;
    // parameters_->minMarkerPerimeterRate = 0.01;
    // parameters_->maxMarkerPerimeterRate = 10.0;
    // parameters_->polygonalApproxAccuracyRate = 0.2;
    // parameters_->minCornerDistanceRate = 0.01;
    // parameters_->minDistanceToBorder = 1;
    // parameters_->minMarkerDistanceRate = 0.01;
    // parameters_->errorCorrectionRate = 0.8;
    // parameters_->minOtsuStdDev = 5.0;
    
    // 优化参数以提高检测速度
    parameters_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    parameters_->cornerRefinementWinSize = 3;
    parameters_->cornerRefinementMaxIterations = 30;
    parameters_->cornerRefinementMinAccuracy = 0.01;
    parameters_->errorCorrectionRate = 0.3;   // 降低纠错率以减少误报
    parameters_->detectInvertedMarker = false;
    
    // 创建CUDA流
    cudaStreamCreate(&stream_);
}

ArUcoDetectorCUDA::~ArUcoDetectorCUDA() {
    freeGPUMemory();
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void ArUcoDetectorCUDA::setCameraParams(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs) {
    camera_matrix_ = camera_matrix.clone();
    dist_coeffs_ = dist_coeffs.clone();
}

void ArUcoDetectorCUDA::setDictionary(int dict_id) {
    dictionary_id_ = dict_id;
    dictionary_ = cv::makePtr<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(dict_id));
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
    
    // 初始化GPU内存
    initGPUMemory(gray.cols, gray.rows);
    
    // 将输入图像复制到GPU
    CUDA_CHECK(cudaMemcpyAsync(d_input_, gray.data, image_size_, 
                              cudaMemcpyHostToDevice, stream_));
    
    // GPU预处理
    preprocessArUcoImageGPU(
        d_input_, d_output_, d_temp1_, d_temp2_,
        gray.cols, gray.rows,
        enable_adaptive_threshold,
        enable_morphology,
        true,  // 启用高斯模糊降噪
        true,  // 启用边缘增强
        stream_);
    
    // 将结果复制回CPU
    output = cv::Mat(gray.rows, gray.cols, CV_8UC1);
    CUDA_CHECK(cudaMemcpyAsync(output.data, d_output_, image_size_, 
                              cudaMemcpyDeviceToHost, stream_));
    
    // 同步等待完成
    CUDA_CHECK(cudaStreamSynchronize(stream_));
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

void ArUcoDetectorCUDA::initGPUMemory(int width, int height) {
    size_t new_size = width * height;
    
    // 如果尺寸改变，重新分配内存
    if (new_size != image_size_) {
        freeGPUMemory();
        
        image_size_ = new_size;
        
        // 分配GPU内存
        CUDA_CHECK(cudaMalloc(&d_input_, image_size_));
        CUDA_CHECK(cudaMalloc(&d_output_, image_size_));
        CUDA_CHECK(cudaMalloc(&d_temp1_, image_size_));
        CUDA_CHECK(cudaMalloc(&d_temp2_, image_size_));
    }
}

void ArUcoDetectorCUDA::freeGPUMemory() {
    if (d_input_) {
        cudaFree(d_input_);
        d_input_ = nullptr;
    }
    if (d_output_) {
        cudaFree(d_output_);
        d_output_ = nullptr;
    }
    if (d_temp1_) {
        cudaFree(d_temp1_);
        d_temp1_ = nullptr;
    }
    if (d_temp2_) {
        cudaFree(d_temp2_);
        d_temp2_ = nullptr;
    }
    image_size_ = 0;
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

    if (debug_mode_) {
        printf("DEBUG: Center depth = %.3f m\n", distance);
    }

    // 即使深度无效，也尝试使用纯视觉PnP估计
    bool success = cv::solvePnP(
        object_points,
        corners,
        camera_matrix_,
        dist_coeffs_,
        rvec,
        tvec,
        false,
        cv::SOLVEPNP_ITERATIVE);

    if (debug_mode_) {
        printf("DEBUG: solvePnP success = %d\n", success);
        if (success && !tvec.empty()) {
            printf("DEBUG: tvec = [%.3f, %.3f, %.3f]\n",
                   tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
        }
    }

    // 如果深度数据无效，使用PnP估计的距离
    if (distance < 0.1f || distance > 5.0f) {
        if (success && !tvec.empty()) {
            distance = tvec.at<double>(2);
            if (debug_mode_) {
                printf("DEBUG: Using PnP distance = %.3f m\n", distance);
            }
        } else {
            return false;
        }
    }
    
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

    // 禁用预处理，直接使用原始彩色图像检测（避免自适应阈值产生假角点）
    cv::Mat processed_image = rgb_image.clone();

    // 检测ArUco标记
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;

    cv::aruco::detectMarkers(processed_image, dictionary_, marker_corners, marker_ids, parameters_);

    if (marker_ids.empty()) {
        return false;
    }

    // ---- 步骤1: 面积过滤 + ID白名单过滤 ----
    std::vector<int> valid_indices;
    for (size_t i = 0; i < marker_ids.size(); i++) {
        float area = cv::contourArea(marker_corners[i]);

        // 面积过滤: 小于阈值的直接丢弃
        if (area < static_cast<float>(min_marker_area_pixels_)) {
            if (debug_mode_) {
                printf("DEBUG: Marker ID %d rejected by area filter (%.0f < %d px)\n",
                       marker_ids[i], area, min_marker_area_pixels_);
            }
            continue;
        }

        // ID白名单过滤: 如果白名单非空，只接受白名单中的ID
        if (!id_whitelist_.empty()) {
            bool in_whitelist = false;
            for (int allowed : id_whitelist_) {
                if (marker_ids[i] == allowed) {
                    in_whitelist = true;
                    break;
                }
            }
            if (!in_whitelist) {
                if (debug_mode_) {
                    printf("DEBUG: Marker ID %d rejected by whitelist filter\n", marker_ids[i]);
                }
                continue;
            }
        }

        valid_indices.push_back(static_cast<int>(i));
    }

    if (valid_indices.empty()) {
        return false;
    }

    // ---- 步骤2: 在有效marker中选择面积最大的 ----
    int best_idx = valid_indices[0];
    float max_area = cv::contourArea(marker_corners[best_idx]);
    for (size_t j = 1; j < valid_indices.size(); j++) {
        int idx = valid_indices[j];
        float area = cv::contourArea(marker_corners[idx]);
        if (area > max_area) {
            max_area = area;
            best_idx = idx;
        }
    }

    int detected_id = marker_ids[best_idx];

    // ---- 步骤3: 时序一致性过滤 ----
    // 要求连续 N 帧检测到同一个ID才确认
    if (detected_id == last_detected_id_) {
        consecutive_count_++;
    } else {
        consecutive_count_ = 1;
        last_detected_id_ = detected_id;
    }

    if (consecutive_count_ < required_consecutive_frames_) {
        if (debug_mode_) {
            printf("DEBUG: Marker ID %d consecutive_count=%d < %d, holding...\n",
                   detected_id, consecutive_count_, required_consecutive_frames_);
        }
        return false;
    }

    // 通过所有过滤，确认检测
    result.detected = true;
    result.marker_id = detected_id;
    result.corners = marker_corners[best_idx];
    result.data = std::to_string(detected_id);

    if (debug_mode_) {
        printf("DEBUG: Marker ID %d CONFIRMED after %d consecutive frames, area=%.0f px\n",
               detected_id, consecutive_count_, max_area);
    }

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