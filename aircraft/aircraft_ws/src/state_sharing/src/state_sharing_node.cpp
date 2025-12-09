#include <rclcpp/rclcpp.hpp>
#include "rclcpp/executors/multi_threaded_executor.hpp"
#include <mutex>

#include <px4_msgs/msg/vehicle_global_position.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <state_sharing/msg/shared_state.hpp>

class StateSharingNode : public rclcpp::Node
{
public:
    StateSharingNode() : Node("state_sharing_node")
    {
        this->declare_parameter<std::string>("autopilot", "px4");
        this->declare_parameter<int>("drone_id", 1);
        autopilot = this->get_parameter("autopilot").as_string();
        drone_id_ = this->get_parameter("drone_id").as_int();

        if (this->get_parameter("use_sim_time").as_bool()) {
            RCLCPP_INFO(this->get_logger(), "Simulation time is enabled.");
        } else {
            RCLCPP_INFO(this->get_logger(), "Simulation time is disabled.");
        }

        publisher_ = this->create_publisher<state_sharing::msg::SharedState>("/state_sharing_drone_" + std::to_string(drone_id_), 10);

        callback_group_timer_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant); // Timed callbacks in parallel
        timer_ = rclcpp::create_timer(this, this->get_clock(), std::chrono::seconds(1), std::bind(&StateSharingNode::publish_timer_callback, this), callback_group_timer_);

        callback_group_subscriber_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant); // Listen to subscribers in parallel
        auto subscriber_options = rclcpp::SubscriptionOptions();
        subscriber_options.callback_group = callback_group_subscriber_;
        rclcpp::QoS qos_profile_sub(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
        qos_profile_sub.keep_last(10);  // History: KEEP_LAST with depth 10
        qos_profile_sub.reliability(rclcpp::ReliabilityPolicy::BestEffort);

        if (autopilot == "px4")
        {
            subscription_px4_ = this->create_subscription<px4_msgs::msg::VehicleGlobalPosition>(
                "/Drone" + std::to_string(drone_id_) + "/fmu/out/vehicle_global_position", 
                qos_profile_sub, std::bind(&StateSharingNode::px4_callback, this, std::placeholders::_1), subscriber_options);
        }
        else
        {
            subscription_apm_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
                "/mavros/global_position/global", 
                qos_profile_sub, std::bind(&StateSharingNode::ardupilot_callback, this, std::placeholders::_1), subscriber_options);
            
        }
        RCLCPP_INFO(this->get_logger(), "state_sharing_node initialized");
    }

private:
    int drone_id_;
    std::string autopilot;

    state_sharing::msg::SharedState latest_position_;
    std::mutex data_mutex_;

    rclcpp::CallbackGroup::SharedPtr callback_group_timer_;
    rclcpp::CallbackGroup::SharedPtr callback_group_subscriber_;

    rclcpp::Publisher<state_sharing::msg::SharedState>::SharedPtr publisher_;
    rclcpp::Subscription<px4_msgs::msg::VehicleGlobalPosition>::SharedPtr subscription_px4_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr subscription_apm_;
    rclcpp::TimerBase::SharedPtr timer_;

    void px4_callback(const px4_msgs::msg::VehicleGlobalPosition::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        latest_position_.latitude_deg = msg->lat;
        latest_position_.longitude_deg = msg->lon;
        latest_position_.altitude_m = msg->alt; // This is AMSL altitude
    }

    void ardupilot_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        latest_position_.latitude_deg = msg->latitude;
        latest_position_.longitude_deg = msg->longitude;
        latest_position_.altitude_m = msg->altitude; // This is ellipsoid altitude
    }

    void publish_timer_callback()
    {
        state_sharing::msg::SharedState message;
        message.header.stamp = this->get_clock()->now();
        message.drone_id = drone_id_;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            message.latitude_deg = latest_position_.latitude_deg;
            message.longitude_deg = latest_position_.longitude_deg;
            message.altitude_m = latest_position_.altitude_m;
        }
        publisher_->publish(message);
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor executor; // Or set num_threads with executor(rclcpp::ExecutorOptions(), 8);
    auto node = std::make_shared<StateSharingNode>();
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
