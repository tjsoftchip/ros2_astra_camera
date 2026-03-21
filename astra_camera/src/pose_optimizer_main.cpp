#include <rclcpp/rclcpp_components/register_node_macro.hpp>
#include "rclcpp/rclcpp.hpp"
#include "astra_camera/pose_optimizer.cpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<astra_camera::PoseOptimizer>());
    rclcpp::shutdown();
    return 0;
}