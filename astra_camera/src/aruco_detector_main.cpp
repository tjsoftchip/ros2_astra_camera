#include <rclcpp/rclcpp.hpp>
#include "astra_camera/aruco_detector_node.h"
#include "astra_camera/dynamic_params.h"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  
  auto node = std::make_shared<rclcpp::Node>("aruco_detector");
  auto parameters = std::make_shared<astra_camera::Parameters>(node.get());
  
  auto aruco_detector = std::make_shared<astra_camera::ArUcoDetectorNode>(node.get(), parameters);
  
  RCLCPP_INFO(node->get_logger(), "ArUco Detector Node started");
  
  rclcpp::spin(node);
  
  rclcpp::shutdown();
  return 0;
}