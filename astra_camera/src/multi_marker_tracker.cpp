#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <astra_camera_msgs/msg/multi_marker_pose.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <unordered_map>
#include <deque>
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
        this->declare_parameter("enable_smoothing", true);
        this->declare_parameter("smoothing_factor", 0.3);
        
        max_track_history_ = this->get_parameter("max_track_history").as_int();
        marker_timeout_ = this->get_parameter("marker_timeout").as_double();
        min_confidence_ = this->get_parameter("min_confidence").as_double();
        enable_smoothing_ = this->get_parameter("enable_smoothing").as_bool();
        smoothing_factor_ = this->get_parameter("smoothing_factor").as_double();
        
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
        if (track.poses.size() > max_track_history_) {
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
                
                if (enable_smoothing_ && track.poses.size() > 1) {
                    // 平滑处理
                    auto smoothed = smooth_pose(track);
                    pose = smoothed.pose;
                } else {
                    pose = track.poses.back().pose;
                }
                
                msg->poses.push_back(pose);
            }
        }
        
        multi_pose_pub_->publish(std::move(msg));
    }
    
    geometry_msgs::msg::PoseStamped smooth_pose(const MarkerTrack& track)
    {
        geometry_msgs::msg::PoseStamped smoothed;
        smoothed.header = track.poses.back().header;
        
        // 简单的指数移动平均
        const auto& last = track.poses.back().pose;
        const auto& prev = track.poses[track.poses.size()-2].pose;
        
        smoothed.pose.position.x = smoothing_factor_ * last.position.x + 
                                 (1.0 - smoothing_factor_) * prev.position.x;
        smoothed.pose.position.y = smoothing_factor_ * last.position.y + 
                                 (1.0 - smoothing_factor_) * prev.position.y;
        smoothed.pose.position.z = smoothing_factor_ * last.position.z + 
                                 (1.0 - smoothing_factor_) * prev.position.z;
        
        // 四元数插值（简化版）
        smoothed.pose.orientation = last.orientation;
        
        return smoothed;
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
    bool enable_smoothing_;
    double smoothing_factor_;
};

} // namespace astra_camera

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(astra_camera::MultiMarkerTracker)