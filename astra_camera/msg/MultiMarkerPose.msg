# 多标记位姿消息
# 用于同时发布多个ArUco标记的位姿信息

std_msgs/Header header

# 标记ID列表
int32[] marker_ids

# 对应的位姿列表
geometry_msgs/Pose[] poses

# 每个标记的置信度 (0.0-1.0)
float64[] confidences

# 检测时间戳
builtin_interfaces/Duration detection_time

# 活跃标记数量
int32 active_marker_count