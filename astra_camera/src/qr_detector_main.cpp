#include <rclcpp/rclcpp.hpp>
#include "astra_camera/qr_code_detector_node.h"
#include "astra_camera/dynamic_params.h"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  
  auto node = std::make_shared<rclcpp::Node>("qr_detector");
  auto parameters = std::make_shared<astra_camera::Parameters>(node.get());
  
  auto qr_detector = std::make_shared<astra_camera::QRCodeDetectorNode>(node.get(), parameters);
  
  RCLCPP_INFO(node->get_logger(), "QR Code Detector Node started");
  
  rclcpp::spin(node);
  
  rclcpp::shutdown();
  return 0;
}