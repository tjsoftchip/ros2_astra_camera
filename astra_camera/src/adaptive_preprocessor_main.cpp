#include <rclcpp/rclcpp_components/register_node_macro.hpp>
#include "rclcpp/rclcpp.hpp"
#include "astra_camera/adaptive_preprocessor.cpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<astra_camera::AdaptivePreprocessor>());
    rclcpp::shutdown();
    return 0;
}