// serial_f407_node.cpp — ROS2 ↔ STM32F407 USB-TTL 串口桥.
//
// 职责:
//   1. 订阅 /cmd_vel,                           → 0xAA55 type=0x01 下行
//   2. 订阅 /lift/target_height (Float32),      → 0xAA55 type=0x02 下行
//      提供 /set_lift_height srv,               → 0xAA55 type=0x02 下行 + 可选到位等待
//   3. 提供 /set_electromagnet srv,             → 0xAA55 type=0x03 下行
//   4. 提供 /lift_home (std_srvs/Trigger),      → 0xAA55 type=0x04 下行
//   5. 解析上行 type=0x01 → 发 /odom + map→base_link TF
//   6. 解析上行 type=0x02 → 发 /imu (sensor_msgs/Imu) + /lift_status (LiftStatus.msg)
//   7. 心跳 5Hz (type=0xFF), 通知 STM32 ROS2 还活
//
// 设计要点:
//   - 多线程 ROS2 callback groups + 单 std::thread 读串口; 长服务等待不阻塞心跳
//   - mutex 保护 fd 写, 多 publisher 写不冲突
//   - 上行解析用 ring-buffer + 状态机, 容忍丢字节 / 半包

#include <chrono>
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <atomic>
#include <cmath>
#include <string>
#include <vector>
#include <errno.h>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "my_robot_drivers/serial_protocol.hpp"
#include "my_robot_msgs/msg/lift_status.hpp"
#include "my_robot_msgs/srv/set_electromagnet.hpp"
#include "my_robot_msgs/srv/set_lift_height.hpp"

namespace my_robot_drivers {

class SerialF407Node : public rclcpp::Node
{
public:
    SerialF407Node() : Node("serial_f407_node")
    {
        // === 参数 ===
        this->declare_parameter<std::string>("port_name",   "/dev/F407");
        this->declare_parameter<int>("baud_rate",            115200);
        this->declare_parameter<std::string>("odom_frame",  "odom");
        this->declare_parameter<std::string>("base_frame",  "base_footprint");
        this->declare_parameter<std::string>("imu_frame",   "imu_link");
        this->declare_parameter<std::string>("cmd_vel_topic", "cmd_vel");
        this->declare_parameter<bool>("publish_tf",          true);
        this->declare_parameter<double>("heartbeat_hz",      5.0);
        this->declare_parameter<double>("cmd_vel_timeout_s", 0.60);
        this->declare_parameter<int>("ack_timeout_ms",       300);
        this->declare_parameter<int>("write_timeout_ms",     50);
        this->declare_parameter<double>("diagnostics_hz",    1.0);
        this->declare_parameter<double>("rx_stale_timeout_s", 1.0);
        this->declare_parameter<bool>("require_ack_for_services", false);
        this->declare_parameter<double>("max_linear_mps", 0.25);
        this->declare_parameter<double>("max_angular_rps", 1.20);
        this->declare_parameter<double>("min_lift_height_m", 0.0);
        // TEST_MODE=0 firmware currently caps 5000 steps at 25000 steps/m.
        this->declare_parameter<double>("max_lift_height_m", 0.20);
        this->declare_parameter<double>("lift_arrival_tolerance_m", 0.015);
        this->declare_parameter<double>("lift_arrival_default_timeout_s", 30.0);
        this->declare_parameter<bool>("require_firmware_identity", true);
        this->declare_parameter<double>("firmware_identity_stale_s", 3.0);
        this->declare_parameter<bool>("gate_invalid_imu", true);
        this->declare_parameter<double>("imu_accel_norm_min_mps2", 5.0);
        this->declare_parameter<double>("imu_accel_norm_max_mps2", 15.0);
        this->declare_parameter<int>("imu_min_valid_samples", 5);

        port_name_   = this->get_parameter("port_name").as_string();
        base_frame_  = this->get_parameter("base_frame").as_string();
        odom_frame_  = this->get_parameter("odom_frame").as_string();
        imu_frame_   = this->get_parameter("imu_frame").as_string();
        cmd_vel_topic_ = this->get_parameter("cmd_vel_topic").as_string();
        publish_tf_  = this->get_parameter("publish_tf").as_bool();
        const double hb_hz = std::max(0.1, this->get_parameter("heartbeat_hz").as_double());
        cmd_vel_timeout_s_ = std::max(0.05, this->get_parameter("cmd_vel_timeout_s").as_double());
        ack_timeout_ = std::chrono::milliseconds(
            std::max<int64_t>(1, this->get_parameter("ack_timeout_ms").as_int()));
        write_timeout_ = std::chrono::milliseconds(
            std::max<int64_t>(1, this->get_parameter("write_timeout_ms").as_int()));
        diagnostics_hz_ = std::max(0.2, this->get_parameter("diagnostics_hz").as_double());
        rx_stale_timeout_s_ = std::max(0.1, this->get_parameter("rx_stale_timeout_s").as_double());
        require_ack_for_services_ = this->get_parameter("require_ack_for_services").as_bool();
        max_linear_mps_ = std::max(0.01, this->get_parameter("max_linear_mps").as_double());
        max_angular_rps_ = std::max(0.05, this->get_parameter("max_angular_rps").as_double());
        min_lift_height_m_ = this->get_parameter("min_lift_height_m").as_double();
        max_lift_height_m_ = std::max(min_lift_height_m_, this->get_parameter("max_lift_height_m").as_double());
        lift_arrival_tolerance_m_ = std::max(0.001, this->get_parameter("lift_arrival_tolerance_m").as_double());
        lift_arrival_default_timeout_s_ = std::max(1.0, this->get_parameter("lift_arrival_default_timeout_s").as_double());
        require_firmware_identity_ = this->get_parameter("require_firmware_identity").as_bool();
        firmware_identity_stale_s_ = std::max(1.0, this->get_parameter("firmware_identity_stale_s").as_double());
        gate_invalid_imu_ = this->get_parameter("gate_invalid_imu").as_bool();
        imu_accel_norm_min_mps2_ = std::max(
            0.0, this->get_parameter("imu_accel_norm_min_mps2").as_double());
        imu_accel_norm_max_mps2_ = std::max(
            imu_accel_norm_min_mps2_,
            this->get_parameter("imu_accel_norm_max_mps2").as_double());
        imu_min_valid_samples_ = static_cast<uint32_t>(std::max<int64_t>(
            1, this->get_parameter("imu_min_valid_samples").as_int()));

        service_callback_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        timer_callback_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        // === 串口 ===
        if (!open_serial(port_name_, this->get_parameter("baud_rate").as_int())) {
            RCLCPP_FATAL(this->get_logger(), "Failed to open %s, exiting", port_name_.c_str());
            throw std::runtime_error("serial open failed");
        }
        RCLCPP_INFO(this->get_logger(), "Opened serial %s at 115200; cmd_vel input=%s",
            port_name_.c_str(), cmd_vel_topic_.c_str());

        // === Publishers ===
        odom_pub_   = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
        imu_pub_    = this->create_publisher<sensor_msgs::msg::Imu>("imu", 50);
        imu_raw_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("imu/raw", 50);
        imu_valid_pub_ = this->create_publisher<std_msgs::msg::Bool>("f407/imu_valid", 10);
        lift_pub_   = this->create_publisher<my_robot_msgs::msg::LiftStatus>("lift_status", 10);
        diag_pub_   = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("diagnostics", 10);
        estop_state_pub_ = this->create_publisher<std_msgs::msg::Bool>("f407/estop_latched", 10);
        cmd_vel_expired_pub_ = this->create_publisher<std_msgs::msg::Bool>("f407/cmd_vel_expired", 10);
        firmware_identity_pub_ = this->create_publisher<std_msgs::msg::Bool>("f407/firmware_identity_valid", 10);
        firmware_info_pub_ = this->create_publisher<std_msgs::msg::String>("f407/firmware_info", 10);
        if (publish_tf_) {
            tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        }

        // === Subscribers ===
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            cmd_vel_topic_, 10,
            std::bind(&SerialF407Node::on_cmd_vel, this, std::placeholders::_1));

        lift_target_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "lift/target_height", 10,
            std::bind(&SerialF407Node::on_lift_target, this, std::placeholders::_1));

        estop_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "estop", 10,
            std::bind(&SerialF407Node::on_estop_topic, this, std::placeholders::_1));

        // === Services ===
        set_em_srv_ = this->create_service<my_robot_msgs::srv::SetElectromagnet>(
            "set_electromagnet",
            std::bind(&SerialF407Node::on_set_electromagnet, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default,
            service_callback_group_);

        set_lift_height_srv_ = this->create_service<my_robot_msgs::srv::SetLiftHeight>(
            "set_lift_height",
            std::bind(&SerialF407Node::on_set_lift_height, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default,
            service_callback_group_);

        lift_home_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "lift_home",
            std::bind(&SerialF407Node::on_lift_home, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default,
            service_callback_group_);

        estop_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "estop",
            std::bind(&SerialF407Node::on_estop, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default,
            service_callback_group_);

        clear_estop_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "clear_estop",
            std::bind(&SerialF407Node::on_clear_estop, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default,
            service_callback_group_);

        // === Heartbeat 定时器 ===
        const auto hb_period = std::chrono::milliseconds(
            static_cast<int>(1000.0 / hb_hz));
        heartbeat_timer_ = this->create_wall_timer(
            hb_period, std::bind(&SerialF407Node::send_heartbeat, this), timer_callback_group_);

        // === Safety / diagnostics 定时器 ===
        cmd_vel_watchdog_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&SerialF407Node::check_cmd_vel_timeout, this), timer_callback_group_);

        const auto diag_period = std::chrono::milliseconds(
            static_cast<int>(1000.0 / diagnostics_hz_));
        diagnostics_timer_ = this->create_wall_timer(
            diag_period, std::bind(&SerialF407Node::publish_diagnostics, this), timer_callback_group_);

        // === 串口读线程 ===
        rx_thread_ = std::thread(&SerialF407Node::rx_loop, this);
    }

    ~SerialF407Node() override
    {
        rx_thread_running_.store(false);
        if (rx_thread_.joinable()) {
            rx_thread_.join();
        }
        if (fd_ >= 0) close(fd_);
    }

private:
    struct AckEvent {
        uint64_t sequence = 0;
        uint8_t status = 0;
        uint8_t reserved = 0;
        int64_t stamp_ns = 0;
    };

    struct LiftCommandEpoch {
        int64_t start_ns = 0;
        uint64_t telemetry_sequence = 0;
    };

    struct LiftTelemetrySnapshot {
        int64_t stamp_ns = 0;
        uint64_t sequence = 0;
        double height_m = std::numeric_limits<double>::quiet_NaN();
        double velocity_mps = 0.0;
        bool moving = false;
    };

    // ==================== 串口打开 ====================
    bool open_serial(const std::string & port, int baud)
    {
        fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0) {
            RCLCPP_ERROR(this->get_logger(), "open(%s) failed: %s", port.c_str(), strerror(errno));
            return false;
        }

        struct termios tty;
        memset(&tty, 0, sizeof(tty));
        if (tcgetattr(fd_, &tty) != 0) {
            RCLCPP_ERROR(this->get_logger(), "tcgetattr: %s", strerror(errno));
            return false;
        }

        speed_t spd;
        switch (baud) {
            case 9600:    spd = B9600;   break;
            case 115200:  spd = B115200; break;
            case 230400:  spd = B230400; break;
            case 460800:  spd = B460800; break;
            case 921600:  spd = B921600; break;
            default:      spd = B115200; break;
        }
        cfsetispeed(&tty, spd);
        cfsetospeed(&tty, spd);

        // 8N1 raw
        tty.c_cflag &= ~PARENB;
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;
        tty.c_cflag |= (CLOCAL | CREAD);
        tty.c_cflag &= ~CRTSCTS;
        tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        tty.c_iflag &= ~(IXON | IXOFF | IXANY);
        tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
        tty.c_oflag &= ~OPOST;

        if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
            RCLCPP_ERROR(this->get_logger(), "tcsetattr: %s", strerror(errno));
            return false;
        }
        return true;
    }

    // ==================== 写 (带 mutex) ====================
    bool write_frame(uint8_t type, const uint8_t * payload, uint8_t len)
    {
        std::lock_guard<std::mutex> lock(write_mutex_);
        if (fd_ < 0) {
            record_write_error("serial fd is closed");
            return false;
        }

        uint8_t frame[MAX_FRAME_SIZE];
        size_t idx = 0;
        frame[idx++] = HEADER_0;
        frame[idx++] = HEADER_1;
        frame[idx++] = type;
        frame[idx++] = len;
        if (payload && len > 0) {
            std::memcpy(&frame[idx], payload, len);
            idx += len;
        }
        const uint8_t cks = compute_checksum(frame, idx);
        frame[idx++] = cks;

        const uint8_t * p = frame;
        size_t remaining = idx;
        const auto deadline = std::chrono::steady_clock::now() + write_timeout_;
        while (remaining > 0) {
            ssize_t n = ::write(fd_, p, remaining);
            if (n < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    if (std::chrono::steady_clock::now() >= deadline) {
                        record_write_error("write timeout");
                        RCLCPP_ERROR(this->get_logger(), "write timeout after %lld ms",
                            static_cast<long long>(write_timeout_.count()));
                        return false;
                    }
                    // 内核缓冲区满, 短时再试
                    std::this_thread::sleep_for(std::chrono::microseconds(500));
                    continue;
                }
                const std::string err = strerror(errno);
                record_write_error(err);
                RCLCPP_ERROR(this->get_logger(), "write: %s", err.c_str());
                return false;
            }
            if (n == 0) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    record_write_error("write returned 0 until timeout");
                    RCLCPP_ERROR(this->get_logger(), "write returned 0 until timeout after %lld ms",
                        static_cast<long long>(write_timeout_.count()));
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(500));
                continue;
            }
            p += n;
            remaining -= n;
        }
        tx_frame_count_.fetch_add(1);
        return true;
    }

    bool firmware_identity_fields_match() const
    {
        return firmware_protocol_version_.load() == TARGET_FIRMWARE_PROTOCOL_VERSION
            && firmware_build_id_.load() == TARGET_FIRMWARE_BUILD_ID
            && firmware_test_mode_.load() == TARGET_FIRMWARE_TEST_MODE
            && firmware_hw_variant_.load() == TARGET_FIRMWARE_HW_VARIANT
            && (firmware_capabilities_.load() & TARGET_FIRMWARE_CAPABILITIES)
                == TARGET_FIRMWARE_CAPABILITIES;
    }

    bool firmware_identity_valid_at(int64_t now_value) const
    {
        const int64_t stamp = last_firmware_info_ns_.load();
        if (stamp <= 0 || !firmware_identity_fields_match()) {
            return false;
        }
        return age_seconds(stamp, now_value) <= firmware_identity_stale_s_;
    }

    void publish_firmware_identity(int64_t now_value)
    {
        const bool valid = firmware_identity_valid_at(now_value);
        const bool identity_enforcement_enabled = require_firmware_identity_;
        std_msgs::msg::Bool valid_msg;
        valid_msg.data = valid;
        firmware_identity_pub_->publish(valid_msg);

        std::ostringstream oss;
        oss << "{\"protocol_version\":" << firmware_protocol_version_.load()
            << ",\"capabilities\":" << firmware_capabilities_.load()
            << ",\"build_id\":" << firmware_build_id_.load()
            << ",\"test_mode\":" << static_cast<int>(firmware_test_mode_.load())
            << ",\"hw_variant\":" << static_cast<int>(firmware_hw_variant_.load())
            << ",\"identity_valid\":" << (valid ? "true" : "false")
            << ",\"required\":" << (identity_enforcement_enabled ? "true" : "false")
            << ",\"identity_enforcement_enabled\":"
            << (identity_enforcement_enabled ? "true" : "false")
            << ",\"age_s\":";
        const double age = age_seconds(last_firmware_info_ns_.load(), now_value);
        if (age < 0.0) {
            oss << "null";
        } else {
            oss << format_double(age);
        }
        oss << ",\"cmd_vel_authority_when_invalid\":"
            << (identity_enforcement_enabled ? "false" : "true") << "}";
        std_msgs::msg::String info_msg;
        info_msg.data = oss.str();
        firmware_info_pub_->publish(info_msg);
    }

    // ==================== 订阅回调 ====================
    void on_cmd_vel(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        const auto now_ns_value = now_ns();
        last_cmd_vel_rx_ns_.store(now_ns_value);

        PayloadCmdVel pl;
        pl.linear_v  = static_cast<float>(msg->linear.x);
        pl.angular_w = static_cast<float>(msg->angular.z);

        if (!std::isfinite(pl.linear_v) || !std::isfinite(pl.angular_w)) {
            invalid_cmd_vel_count_.fetch_add(1);
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Ignoring invalid cmd_vel linear=%.3f angular=%.3f; sending zero for safety",
                pl.linear_v, pl.angular_w);
            send_zero_cmd_vel("invalid cmd_vel");
            return;
        }

        if (std::fabs(pl.linear_v) > max_linear_mps_ || std::fabs(pl.angular_w) > max_angular_rps_) {
            limited_cmd_vel_count_.fetch_add(1);
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "cmd_vel over hard limit linear=%.3f angular=%.3f (limits %.3f/%.3f); sending zero",
                pl.linear_v, pl.angular_w, max_linear_mps_, max_angular_rps_);
            send_zero_cmd_vel("cmd_vel over hard limit");
            return;
        }

        const bool nonzero = std::fabs(pl.linear_v) > 1e-6f || std::fabs(pl.angular_w) > 1e-6f;
        if (nonzero && require_firmware_identity_ && !firmware_identity_valid_at(now_ns_value)) {
            blocked_cmd_vel_by_firmware_identity_count_.fetch_add(1);
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "nonzero cmd_vel blocked: F407 firmware identity missing, stale, or mismatched");
            send_zero_cmd_vel("firmware identity invalid");
            return;
        }

        if (estop_latched_.load()) {
            blocked_cmd_vel_count_.fetch_add(1);
            if (std::fabs(pl.linear_v) > 1e-6f || std::fabs(pl.angular_w) > 1e-6f) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "cmd_vel blocked while F407 estop is latched");
            }
            send_zero_cmd_vel("estop latched");
            return;
        }

        if (cmd_vel_expired_.exchange(false)) {
            RCLCPP_INFO(this->get_logger(), "cmd_vel stream restored");
            publish_safety_state();
        }

        if (write_frame(static_cast<uint8_t>(DownType::CMD_VEL),
                        reinterpret_cast<uint8_t*>(&pl), sizeof(pl))) {
            last_cmd_vel_forward_ns_.store(now_ns_value);
            last_cmd_linear_.store(pl.linear_v);
            last_cmd_angular_.store(pl.angular_w);
        }
    }

    void on_lift_target(const std_msgs::msg::Float32::SharedPtr msg)
    {
        std::string message;
        if (!send_lift_target_height(msg->data, false, message)) {
            RCLCPP_WARN(this->get_logger(), "lift target topic rejected: %s", message.c_str());
        }
    }

    void on_set_lift_height(
        const std::shared_ptr<my_robot_msgs::srv::SetLiftHeight::Request> req,
        std::shared_ptr<my_robot_msgs::srv::SetLiftHeight::Response> res)
    {
        const float target = req->target_height_m;
        std::string message;
        LiftCommandEpoch command_epoch;
        res->reached_height_m = current_lift_height();
        if (!send_lift_target_height(target, true, message, &command_epoch)) {
            res->success = false;
            res->message = message;
            res->reached_height_m = current_lift_height();
            return;
        }

        if (!req->wait_for_arrival) {
            res->success = true;
            res->message = message + "; wait_for_arrival=false";
            res->reached_height_m = current_lift_height();
            return;
        }

        const double timeout_s = req->timeout_s > 0.0f
            ? std::max(0.1, static_cast<double>(req->timeout_s))
            : lift_arrival_default_timeout_s_;
        res->success = wait_for_lift_arrival(target, timeout_s, command_epoch, message);
        res->reached_height_m = current_lift_height();
        res->message = message;
    }

    bool send_lift_target_height(
        float target_height_m,
        bool wait_ack,
        std::string & message,
        LiftCommandEpoch * command_epoch = nullptr)
    {
        if (estop_latched_.load()) {
            rejected_lift_target_count_.fetch_add(1);
            blocked_actuator_by_estop_count_.fetch_add(1);
            message = "lift target blocked while estop is latched";
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "lift target blocked while F407 estop is latched");
            return false;
        }

        if (require_firmware_identity_ && !firmware_identity_valid_at(now_ns())) {
            rejected_lift_target_count_.fetch_add(1);
            blocked_actuator_by_firmware_identity_count_.fetch_add(1);
            message = "lift target blocked: F407 firmware identity missing, stale, or mismatched";
            return false;
        }

        if (!std::isfinite(target_height_m)) {
            rejected_lift_target_count_.fetch_add(1);
            message = "invalid lift target height";
            return false;
        }
        const bool fixture_command =
            std::fabs(target_height_m - (-1.0f)) <= 0.01f
            || std::fabs(target_height_m - (-2.0f)) <= 0.01f;
        const bool servo_diagnostic_command =
            std::fabs(target_height_m - (-3.0f)) <= 0.01f
            || std::fabs(target_height_m - (-4.0f)) <= 0.01f;
        if (!fixture_command && !servo_diagnostic_command
            && (target_height_m < min_lift_height_m_
                || target_height_m > max_lift_height_m_)) {
            rejected_lift_target_count_.fetch_add(1);
            std::ostringstream oss;
            oss << "lift target " << target_height_m << " outside safe range ["
                << min_lift_height_m_ << ", " << max_lift_height_m_ << "]";
            message = oss.str();
            return false;
        }

        PayloadSetLiftHeight pl;
        pl.target_height_m = target_height_m;
        requested_lift_target_m_.store(static_cast<double>(target_height_m));
        if (wait_ack) {
            return send_command_wait_ack(
                DownType::SET_LIFT_HEIGHT, reinterpret_cast<uint8_t*>(&pl), sizeof(pl),
                message, command_epoch);
        }
        if (!write_frame(static_cast<uint8_t>(DownType::SET_LIFT_HEIGHT),
                         reinterpret_cast<uint8_t*>(&pl), sizeof(pl))) {
            message = "serial write failed for SET_LIFT_HEIGHT";
            return false;
        }
        message = "SET_LIFT_HEIGHT sent without ACK wait";
        return true;
    }

    float current_lift_height() const
    {
        return static_cast<float>(last_lift_height_m_.load());
    }

    LiftTelemetrySnapshot lift_telemetry_snapshot()
    {
        std::lock_guard<std::mutex> lock(lift_telemetry_mutex_);
        LiftTelemetrySnapshot snapshot;
        snapshot.stamp_ns = last_telem_ns_.load();
        snapshot.sequence = telem_frame_count_.load();
        snapshot.height_m = last_lift_height_m_.load();
        snapshot.velocity_mps = last_lift_velocity_mps_.load();
        snapshot.moving = last_lift_moving_.load();
        return snapshot;
    }

    bool wait_for_lift_arrival(
        float target_height_m,
        double timeout_s,
        const LiftCommandEpoch & command_epoch,
        std::string & message)
    {
        const auto timeout_ms = std::chrono::milliseconds(
            static_cast<int>(std::max(0.1, timeout_s) * 1000.0));
        const auto deadline = std::chrono::steady_clock::now() + timeout_ms;
        while (std::chrono::steady_clock::now() < deadline) {
            if (estop_latched_.load()) {
                message = "estop latched while waiting for lift arrival";
                return false;
            }
            const LiftTelemetrySnapshot telemetry = lift_telemetry_snapshot();
            const bool fresh_post_command =
                telemetry.sequence > command_epoch.telemetry_sequence
                && telemetry.stamp_ns > command_epoch.start_ns;
            if (fresh_post_command) {
                const double h = telemetry.height_m;
                const double v = std::fabs(telemetry.velocity_mps);
                const bool moving = telemetry.moving;
                const double err = std::fabs(h - static_cast<double>(target_height_m));
                if (err <= lift_arrival_tolerance_m_ && (!moving || v < 0.003)) {
                    std::ostringstream oss;
                    oss << "lift arrived: reached=" << h
                        << " target=" << target_height_m
                        << " tolerance=" << lift_arrival_tolerance_m_
                        << " telemetry_seq=" << telemetry.sequence;
                    message = oss.str();
                    return true;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        const LiftTelemetrySnapshot telemetry = lift_telemetry_snapshot();
        std::ostringstream oss;
        oss << "lift arrival timeout: reached=" << telemetry.height_m
            << " target=" << target_height_m
            << " timeout_s=" << timeout_s;
        if (telemetry.sequence <= command_epoch.telemetry_sequence
            || telemetry.stamp_ns <= command_epoch.start_ns)
        {
            oss << "; no fresh post-command F407 lift telemetry"
                << " (command_start_seq=" << command_epoch.telemetry_sequence
                << " latest_seq=" << telemetry.sequence << ")";
        }
        message = oss.str();
        return false;
    }

    void on_set_electromagnet(
        const std::shared_ptr<my_robot_msgs::srv::SetElectromagnet::Request> req,
        std::shared_ptr<my_robot_msgs::srv::SetElectromagnet::Response> res)
    {
        if (estop_latched_.load() && req->turn_on) {
            blocked_actuator_by_estop_count_.fetch_add(1);
            res->success = false;
            res->message = "electromagnet ON blocked while estop is latched; OFF is still allowed";
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "electromagnet ON blocked while F407 estop is latched");
            return;
        }
        if (req->turn_on && require_firmware_identity_ && !firmware_identity_valid_at(now_ns())) {
            blocked_actuator_by_firmware_identity_count_.fetch_add(1);
            res->success = false;
            res->message = "electromagnet ON blocked: F407 firmware identity invalid; OFF remains allowed";
            return;
        }

        PayloadSetElectromagnet pl;
        pl.turn_on = req->turn_on ? 1 : 0;
        std::string message;
        res->success = send_command_wait_ack(
            DownType::SET_ELECTROMAGNET,
            reinterpret_cast<uint8_t*>(&pl), sizeof(pl), message);
        res->message = message;
    }

    void on_lift_home(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> res)
    {
        if (estop_latched_.load()) {
            blocked_actuator_by_estop_count_.fetch_add(1);
            res->success = false;
            res->message = "lift_home blocked while estop is latched";
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "lift_home blocked while F407 estop is latched");
            return;
        }
        if (require_firmware_identity_ && !firmware_identity_valid_at(now_ns())) {
            blocked_actuator_by_firmware_identity_count_.fetch_add(1);
            res->success = false;
            res->message = "lift_home blocked: F407 firmware identity missing, stale, or mismatched";
            return;
        }

        std::string message;
        res->success = send_command_wait_ack(DownType::LIFT_HOME, nullptr, 0, message);
        res->message = message;
    }

    void on_estop(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> res)
    {
        std::string message;
        res->success = assert_estop(true, message);
        res->message = message;
    }

    void on_clear_estop(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> res)
    {
        send_zero_cmd_vel("clear_estop");
        if (require_firmware_identity_ && !firmware_identity_valid_at(now_ns())) {
            res->success = false;
            res->message = "local estop remains latched; F407 firmware identity missing, stale, or mismatched";
            return;
        }
        const uint64_t before_seq = ack_sequence_.load();
        if (!write_frame(static_cast<uint8_t>(DownType::CLEAR_ESTOP), nullptr, 0)) {
            res->success = false;
            res->message = "local estop remains latched; serial write failed for CLEAR_ESTOP";
            return;
        }
        std::string message;
        if (!wait_for_ack(DownType::CLEAR_ESTOP, before_seq, message)) {
            res->success = false;
            res->message = "local estop remains latched; " + message;
            return;
        }
        hardware_estop_latched_.store(false);
        hardware_emergency_active_.store(false);
        clear_local_estop();
        res->success = true;
        res->message = "F407 CLEAR_ESTOP ACK ok; zero cmd_vel sent; local latch cleared";
    }

    void on_estop_topic(const std_msgs::msg::Bool::SharedPtr msg)
    {
        if (msg->data) {
            std::string ignored;
            assert_estop(false, ignored);
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                "Ignoring false /estop topic message; clear requires explicit /clear_estop service");
            publish_safety_state();
        }
    }

    void send_heartbeat()
    {
        write_frame(static_cast<uint8_t>(DownType::HEARTBEAT), nullptr, 0);
    }

    bool send_command_wait_ack(
        DownType type,
        const uint8_t * payload,
        uint8_t len,
        std::string & message,
        LiftCommandEpoch * lift_command_epoch = nullptr)
    {
        const uint64_t before_seq = ack_sequence_.load();
        const int64_t command_start_ns = lift_command_epoch != nullptr ? now_ns() : 0;
        if (!write_frame(static_cast<uint8_t>(type), payload, len)) {
            message = "serial write failed for " + down_type_name(type);
            return false;
        }
        if (lift_command_epoch != nullptr) {
            std::lock_guard<std::mutex> lock(lift_telemetry_mutex_);
            lift_command_epoch->start_ns = command_start_ns;
            // Exclude telemetry completed before the command write returned.
            lift_command_epoch->telemetry_sequence = telem_frame_count_.load();
        }
        if (!require_ack_for_services_) {
            message = down_type_name(type) + " sent; ACK wait disabled";
            return true;
        }
        return wait_for_ack(type, before_seq, message);
    }

    bool wait_for_ack(DownType type, uint64_t after_seq, std::string & message)
    {
        const uint8_t idx = static_cast<uint8_t>(type);
        std::unique_lock<std::mutex> lock(ack_mutex_);
        const auto deadline = std::chrono::steady_clock::now() + ack_timeout_;
        while (true) {
            const AckEvent & ev = last_ack_by_type_[idx];
            if (ev.sequence > after_seq) {
                if (ev.status == 0) {
                    message = down_type_name(type) + " ACK ok";
                    return true;
                }
                ack_error_count_.fetch_add(1);
                message = down_type_name(type) + " ACK error status=" + std::to_string(ev.status);
                return false;
            }

            if (ack_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                ack_timeout_count_.fetch_add(1);
                last_ack_timeout_ns_.store(now_ns());
                message = down_type_name(type) + " ACK timeout after "
                    + std::to_string(ack_timeout_.count()) + " ms";
                RCLCPP_ERROR(this->get_logger(), "%s", message.c_str());
                return false;
            }
            if (!rclcpp::ok()) {
                message = down_type_name(type) + " ACK wait interrupted";
                return false;
            }
        }
    }

    bool assert_estop(bool wait_ack, std::string & message)
    {
        estop_latched_.store(true);
        cmd_vel_expired_.store(false);
        publish_safety_state();
        RCLCPP_ERROR(this->get_logger(), "F407 estop asserted");

        if (wait_ack) {
            const uint64_t before_seq = ack_sequence_.load();
            if (!write_frame(static_cast<uint8_t>(DownType::EMERGENCY_STOP), nullptr, 0)) {
                message = "local estop latch asserted; serial write failed for EMERGENCY_STOP";
                return false;
            }
            if (!wait_for_ack(DownType::EMERGENCY_STOP, before_seq, message)) {
                message = "local estop latch asserted; " + message;
                return false;
            }
            return true;
        }

        if (!write_frame(static_cast<uint8_t>(DownType::EMERGENCY_STOP), nullptr, 0)) {
            message = "serial write failed for EMERGENCY_STOP";
            return false;
        }
        message = "estop sent";
        return true;
    }

    void clear_local_estop()
    {
        if (estop_latched_.exchange(false)) {
            RCLCPP_WARN(this->get_logger(), "F407 local estop latch cleared");
        }
        publish_safety_state();
    }

    void send_zero_cmd_vel(const char * reason)
    {
        PayloadCmdVel stop{};
        stop.linear_v = 0.0f;
        stop.angular_w = 0.0f;
        if (write_frame(static_cast<uint8_t>(DownType::CMD_VEL),
                        reinterpret_cast<uint8_t*>(&stop), sizeof(stop))) {
            last_cmd_vel_forward_ns_.store(now_ns());
            last_cmd_linear_.store(0.0f);
            last_cmd_angular_.store(0.0f);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to send zero cmd_vel (%s)", reason);
        }
    }

    void check_cmd_vel_timeout()
    {
        const int64_t last_rx = last_cmd_vel_rx_ns_.load();
        if (last_rx <= 0 || estop_latched_.load()) {
            return;
        }

        const double age_s = age_seconds(last_rx, now_ns());
        if (age_s <= cmd_vel_timeout_s_) {
            return;
        }

        if (!cmd_vel_expired_.exchange(true)) {
            cmd_vel_timeout_count_.fetch_add(1);
            RCLCPP_WARN(this->get_logger(),
                "cmd_vel expired after %.3fs (> %.3fs); sending zero cmd_vel",
                age_s, cmd_vel_timeout_s_);
            send_zero_cmd_vel("cmd_vel timeout");
            publish_safety_state();
        }
    }

    // ==================== 读 (后台线程, 状态机解析) ====================
    void rx_loop()
    {
        rx_thread_running_.store(true);
        std::vector<uint8_t> rx_buf;
        rx_buf.reserve(MAX_FRAME_SIZE * 4);
        uint8_t tmp[256];

        while (rx_thread_running_.load() && rclcpp::ok()) {
            ssize_t n = ::read(fd_, tmp, sizeof(tmp));
            if (n > 0) {
                rx_byte_count_.fetch_add(static_cast<uint64_t>(n));
                rx_buf.insert(rx_buf.end(), tmp, tmp + n);
                // 尝试解析
                while (try_parse_one(rx_buf)) {}
                // 防止脏数据无限堆积
                if (rx_buf.size() > MAX_FRAME_SIZE * 8) {
                    RCLCPP_WARN(this->get_logger(),
                        "rx_buf overflow (%zu B), dropping front half", rx_buf.size());
                    rx_buf.erase(rx_buf.begin(), rx_buf.begin() + rx_buf.size() / 2);
                }
            } else if (n < 0) {
                if (errno != EAGAIN && errno != EWOULDBLOCK) {
                    read_error_count_.fetch_add(1);
                    record_last_error(strerror(errno));
                    RCLCPP_ERROR(this->get_logger(), "read: %s", strerror(errno));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    }

    // 尝试从 buf 头部消费一个完整帧. 返回 true 表示消费了一个帧 (或丢了一个错误字节).
    bool try_parse_one(std::vector<uint8_t> & buf)
    {
        // 找 header
        while (buf.size() >= 2 && (buf[0] != HEADER_0 || buf[1] != HEADER_1)) {
            buf.erase(buf.begin());  // 丢一个字节, 重新找
            return true;             // 让 caller 再来调一次
        }
        if (buf.size() < 4) return false;  // 不够 header+type+len

        const uint8_t type = buf[2];
        const uint8_t len  = buf[3];
        const size_t need = 4 + len + 1;  // 头2 + type + len + payload + checksum

        if (buf.size() < need) return false;  // 等更多字节

        // 校验
        const uint8_t cks_calc = compute_checksum(buf.data(), 4 + len);
        const uint8_t cks_recv = buf[4 + len];
        if (cks_calc != cks_recv) {
            checksum_error_count_.fetch_add(1);
            RCLCPP_WARN(this->get_logger(),
                "checksum mismatch type=0x%02X len=%u, expected 0x%02X got 0x%02X",
                type, len, cks_calc, cks_recv);
            buf.erase(buf.begin());  // 丢同步字节, 重找
            return true;
        }

        // 派发
        const uint8_t * payload = &buf[4];
        rx_frame_count_.fetch_add(1);
        last_rx_ns_.store(now_ns());
        dispatch_up(static_cast<UpType>(type), payload, len);

        // 消费整帧
        buf.erase(buf.begin(), buf.begin() + need);
        return true;
    }

    void dispatch_up(UpType type, const uint8_t * payload, uint8_t len)
    {
        switch (type) {
            case UpType::BASIC_ODOM:
                if (len == sizeof(PayloadBasicOdom)) {
                    PayloadBasicOdom pl;
                    std::memcpy(&pl, payload, sizeof(pl));
                    handle_basic_odom(pl);
                }
                break;
            case UpType::EXT_TELEMETRY:
                if (len == sizeof(PayloadExtTelemetry)) {
                    PayloadExtTelemetry pl;
                    std::memcpy(&pl, payload, sizeof(pl));
                    handle_ext_telemetry(pl);
                }
                break;
            case UpType::SAFETY_STATE:
                if (len == sizeof(PayloadSafetyState)) {
                    PayloadSafetyState pl;
                    std::memcpy(&pl, payload, sizeof(pl));
                    handle_safety_state(pl);
                } else {
                    RCLCPP_WARN(this->get_logger(), "short SAFETY_STATE frame len=%u", len);
                }
                break;
            case UpType::FIRMWARE_INFO:
                if (len == sizeof(PayloadFirmwareInfo)) {
                    PayloadFirmwareInfo pl;
                    std::memcpy(&pl, payload, sizeof(pl));
                    handle_firmware_info(pl);
                } else {
                    RCLCPP_WARN(this->get_logger(), "short FIRMWARE_INFO frame len=%u", len);
                }
                break;
            case UpType::ACK:
                handle_ack(payload, len);
                break;
            case UpType::ERROR: {
                stm_error_count_.fetch_add(1);
                if (len == 0) {
                    record_last_error("STM32 ERROR frame with empty payload");
                    RCLCPP_ERROR(this->get_logger(), "STM32 ERROR frame with empty payload");
                    break;
                }
                std::string msg(reinterpret_cast<const char*>(payload + 1),
                                std::min<size_t>(len - 1, 31));
                record_last_error(msg);
                RCLCPP_ERROR(this->get_logger(),
                    "STM32 error code=%u msg=%s", payload[0], msg.c_str());
                break;
            }
            default:
                RCLCPP_WARN(this->get_logger(), "unknown up type 0x%02X", static_cast<uint8_t>(type));
        }
    }

    void handle_ack(const uint8_t * payload, uint8_t len)
    {
        if (len < 2) {
            ack_error_count_.fetch_add(1);
            RCLCPP_WARN(this->get_logger(), "short ACK frame len=%u", len);
            return;
        }

        PayloadAck pl{};
        pl.ack_for_type = payload[0];
        pl.status = payload[1];
        pl.reserved = len >= 3 ? payload[2] : 0;

        const int64_t stamp = now_ns();
        {
            std::lock_guard<std::mutex> lock(ack_mutex_);
            const uint64_t seq = ack_sequence_.fetch_add(1) + 1;
            AckEvent & ev = last_ack_by_type_[pl.ack_for_type];
            ev.sequence = seq;
            ev.status = pl.status;
            ev.reserved = pl.reserved;
            ev.stamp_ns = stamp;
        }
        last_ack_ns_.store(stamp);
        last_ack_for_type_.store(pl.ack_for_type);
        last_ack_status_.store(pl.status);
        ack_frame_count_.fetch_add(1);
        ack_cv_.notify_all();

        if (pl.status != 0) {
            RCLCPP_WARN(this->get_logger(),
                "ACK error for type=0x%02X status=%u", pl.ack_for_type, pl.status);
        }
    }

    void handle_basic_odom(const PayloadBasicOdom & pl)
    {
        const auto stamp = this->get_clock()->now();
        last_odom_ns_.store(stamp.nanoseconds());
        odom_frame_count_.fetch_add(1);
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = odom_frame_;
        odom.child_frame_id  = base_frame_;

        odom.pose.pose.position.x = pl.x;
        odom.pose.pose.position.y = pl.y;
        odom.pose.pose.position.z = 0.0;

        const double yaw_rad = pl.yaw_deg * M_PI / 180.0;
        tf2::Quaternion q;
        q.setRPY(0, 0, yaw_rad);
        q.normalize();
        odom.pose.pose.orientation = tf2::toMsg(q);

        odom.twist.twist.linear.x  = pl.vx;
        odom.twist.twist.angular.z = pl.wz;

        // 协方差: 简单写, EKF 融合时可调
        odom.pose.covariance = {
            0.05, 0, 0, 0, 0, 0,
            0, 0.05, 0, 0, 0, 0,
            0, 0, 1e6, 0, 0, 0,   // z 方向不可观测, 大方差
            0, 0, 0, 1e6, 0, 0,
            0, 0, 0, 0, 1e6, 0,
            0, 0, 0, 0, 0, 0.1
        };
        odom.twist.covariance = odom.pose.covariance;

        odom_pub_->publish(odom);

        if (publish_tf_) {
            geometry_msgs::msg::TransformStamped tf;
            tf.header.stamp = stamp;
            tf.header.frame_id = odom_frame_;
            tf.child_frame_id  = base_frame_;
            tf.transform.translation.x = pl.x;
            tf.transform.translation.y = pl.y;
            tf.transform.translation.z = 0.0;
            tf.transform.rotation = odom.pose.pose.orientation;
            tf_broadcaster_->sendTransform(tf);
        }
    }

    void handle_ext_telemetry(const PayloadExtTelemetry & pl)
    {
        const auto stamp = this->get_clock()->now();

        {
            std::lock_guard<std::mutex> lock(lift_telemetry_mutex_);
            last_lift_height_m_.store(static_cast<double>(pl.lift_height_m));
            last_lift_velocity_mps_.store(static_cast<double>(pl.lift_velocity_mps));
            last_lift_moving_.store(std::fabs(pl.lift_velocity_mps) > 0.001f);
            last_telem_ns_.store(stamp.nanoseconds());
            telem_frame_count_.fetch_add(1);
        }

        // Preserve raw telemetry for diagnosis, but do not feed an all-zero or
        // otherwise implausible sensor into state estimation.
        sensor_msgs::msg::Imu imu;
        imu.header.stamp = stamp;
        imu.header.frame_id = imu_frame_;
        imu.linear_acceleration.x = pl.accel_x;
        imu.linear_acceleration.y = pl.accel_y;
        imu.linear_acceleration.z = pl.accel_z;
        imu.angular_velocity.x = pl.gyro_x;
        imu.angular_velocity.y = pl.gyro_y;
        imu.angular_velocity.z = pl.gyro_z;
        // orientation 不知, 只发裸 IMU; orientation_covariance[0]=-1 表示无效
        imu.orientation_covariance[0] = -1.0;
        // 默认协方差 (校准前)
        for (int i = 0; i < 9; ++i) {
            imu.linear_acceleration_covariance[i] = (i % 4 == 0) ? 0.04 : 0.0;
            imu.angular_velocity_covariance[i]    = (i % 4 == 0) ? 0.001 : 0.0;
        }
        imu_raw_pub_->publish(imu);
        imu_raw_frame_count_.fetch_add(1);

        const bool finite =
            std::isfinite(pl.accel_x) && std::isfinite(pl.accel_y) &&
            std::isfinite(pl.accel_z) && std::isfinite(pl.gyro_x) &&
            std::isfinite(pl.gyro_y) && std::isfinite(pl.gyro_z);
        const double accel_norm = finite ? std::sqrt(
            static_cast<double>(pl.accel_x) * pl.accel_x +
            static_cast<double>(pl.accel_y) * pl.accel_y +
            static_cast<double>(pl.accel_z) * pl.accel_z) :
            std::numeric_limits<double>::quiet_NaN();
        last_imu_accel_norm_mps2_.store(accel_norm);
        const bool plausible = finite &&
            accel_norm >= imu_accel_norm_min_mps2_ &&
            accel_norm <= imu_accel_norm_max_mps2_;
        uint32_t consecutive = 0;
        if (plausible) {
            consecutive = imu_consecutive_valid_samples_.fetch_add(1) + 1;
        } else {
            imu_consecutive_valid_samples_.store(0);
        }
        const bool imu_valid = !gate_invalid_imu_ ||
            consecutive >= imu_min_valid_samples_;
        imu_valid_.store(imu_valid);

        std_msgs::msg::Bool imu_valid_msg;
        imu_valid_msg.data = imu_valid;
        imu_valid_pub_->publish(imu_valid_msg);
        if (imu_valid) {
            imu_pub_->publish(imu);
            imu_valid_frame_count_.fetch_add(1);
        }

        // /lift_status
        my_robot_msgs::msg::LiftStatus ls;
        ls.header.stamp = stamp;
        ls.header.frame_id = "lift_link";
        ls.height_m   = pl.lift_height_m;
        ls.target_height_m = static_cast<float>(requested_lift_target_m_.load());
        ls.velocity_mps = pl.lift_velocity_mps;
        ls.home_switch_triggered = pl.home_switch != 0;
        ls.top_switch_triggered  = pl.top_switch != 0;
        ls.homed                 = pl.homed != 0;
        ls.moving                = std::fabs(pl.lift_velocity_mps) > 0.001f;
        ls.electromagnet_on      = pl.electromagnet_state != 0;
        lift_pub_->publish(ls);

        // 低电压告警
        if (pl.bus_voltage_v < 10.5f && pl.bus_voltage_v > 0.1f) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Bus voltage low: %.2f V", pl.bus_voltage_v);
        }
    }

    void handle_safety_state(const PayloadSafetyState & pl)
    {
        last_safety_state_ns_.store(now_ns());
        safety_state_frame_count_.fetch_add(1);
        hardware_estop_latched_.store(pl.estop_latched != 0);
        hardware_emergency_active_.store(pl.emergency_active != 0);
        firmware_estop_blocked_command_count_.store(pl.blocked_command_count);
        if (pl.estop_latched != 0) {
            estop_latched_.store(true);
        }
        publish_safety_state();
    }

    void handle_firmware_info(const PayloadFirmwareInfo & pl)
    {
        const bool first = firmware_info_frame_count_.fetch_add(1) == 0;
        firmware_protocol_version_.store(pl.protocol_version);
        firmware_capabilities_.store(pl.capabilities);
        firmware_build_id_.store(pl.build_id);
        firmware_test_mode_.store(pl.test_mode);
        firmware_hw_variant_.store(pl.hw_variant);
        last_firmware_info_ns_.store(now_ns());
        const bool fields_match = firmware_identity_fields_match();
        if (first) {
            RCLCPP_INFO(this->get_logger(),
                "F407 firmware identity: protocol=%u capabilities=0x%04X build=%u test_mode=%u hw=%u valid=%s",
                pl.protocol_version, pl.capabilities, pl.build_id, pl.test_mode,
                pl.hw_variant, fields_match ? "true" : "false");
        }
        if (!fields_match) {
            if (require_firmware_identity_) {
                RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "F407 firmware identity mismatch; all non-safety-direction commands are blocked");
            } else {
                RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "F407 firmware identity mismatch while identity enforcement is disabled; "
                    "nonzero motion remains authorized by configuration");
            }
        }
        publish_firmware_identity(now_ns());
    }

    void publish_safety_state()
    {
        std_msgs::msg::Bool estop_msg;
        estop_msg.data = estop_latched_.load() || hardware_estop_latched_.load();
        estop_state_pub_->publish(estop_msg);

        std_msgs::msg::Bool expired_msg;
        expired_msg.data = cmd_vel_expired_.load();
        cmd_vel_expired_pub_->publish(expired_msg);
    }

    void publish_diagnostics()
    {
        const auto stamp = this->get_clock()->now();
        const int64_t now_value = stamp.nanoseconds();
        publish_safety_state();
        publish_firmware_identity(now_value);

        diagnostic_msgs::msg::DiagnosticArray array;
        array.header.stamp = stamp;

        diagnostic_msgs::msg::DiagnosticStatus link;
        link.name = "serial_f407_node: serial_link";
        link.hardware_id = port_name_;
        link.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
        link.message = "serial link active";

        const int64_t last_rx = last_rx_ns_.load();
        const double rx_age = age_seconds(last_rx, now_value);
        if (fd_ < 0) {
            link.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
            link.message = "serial fd closed";
        } else if (last_rx <= 0) {
            link.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
            link.message = "no F407 frames received yet";
        } else if (rx_age > rx_stale_timeout_s_) {
            link.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
            link.message = "F407 RX stream stale";
        } else if (require_firmware_identity_ && !firmware_identity_valid_at(now_value)) {
            link.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
            link.message = "F407 firmware identity missing, stale, or mismatched; motion blocked";
        }

        add_kv(link, "port", port_name_);
        add_kv(link, "rx_age_s", format_age(rx_age));
        add_kv(link, "odom_age_s", format_age(age_seconds(last_odom_ns_.load(), now_value)));
        add_kv(link, "telem_age_s", format_age(age_seconds(last_telem_ns_.load(), now_value)));
        add_kv(link, "last_ack_age_s", format_age(age_seconds(last_ack_ns_.load(), now_value)));
        add_kv(link, "rx_frames", std::to_string(rx_frame_count_.load()));
        add_kv(link, "rx_bytes", std::to_string(rx_byte_count_.load()));
        add_kv(link, "tx_frames", std::to_string(tx_frame_count_.load()));
        add_kv(link, "odom_frames", std::to_string(odom_frame_count_.load()));
        add_kv(link, "telem_frames", std::to_string(telem_frame_count_.load()));
        add_kv(link, "imu_valid", imu_valid_.load() ? "true" : "false");
        add_kv(link, "imu_accel_norm_mps2", format_double(last_imu_accel_norm_mps2_.load()));
        add_kv(link, "imu_consecutive_valid_samples",
               std::to_string(imu_consecutive_valid_samples_.load()));
        add_kv(link, "imu_raw_frames", std::to_string(imu_raw_frame_count_.load()));
        add_kv(link, "imu_valid_frames", std::to_string(imu_valid_frame_count_.load()));
        add_kv(link, "gate_invalid_imu", gate_invalid_imu_ ? "true" : "false");
        add_kv(link, "safety_state_frames", std::to_string(safety_state_frame_count_.load()));
        add_kv(link, "firmware_info_frames", std::to_string(firmware_info_frame_count_.load()));
        add_kv(link, "firmware_info_age_s",
               format_age(age_seconds(last_firmware_info_ns_.load(), now_value)));
        add_kv(link, "ack_frames", std::to_string(ack_frame_count_.load()));
        add_kv(link, "checksum_errors", std::to_string(checksum_error_count_.load()));
        add_kv(link, "read_errors", std::to_string(read_error_count_.load()));
        add_kv(link, "write_errors", std::to_string(write_error_count_.load()));
        add_kv(link, "stm32_errors", std::to_string(stm_error_count_.load()));
        add_kv(link, "last_ack_type", byte_hex(last_ack_for_type_.load()));
        add_kv(link, "last_ack_status", std::to_string(static_cast<int>(last_ack_status_.load())));
        add_kv(link, "last_ack_timeout_age_s",
               format_age(age_seconds(last_ack_timeout_ns_.load(), now_value)));
        add_kv(link, "ack_timeouts", std::to_string(ack_timeout_count_.load()));
        add_kv(link, "ack_errors", std::to_string(ack_error_count_.load()));
        add_kv(link, "last_error", get_last_error());
        array.status.push_back(link);

        diagnostic_msgs::msg::DiagnosticStatus safety;
        safety.name = "serial_f407_node: safety_bridge";
        safety.hardware_id = port_name_;
        safety.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
        safety.message = "safety bridge nominal";
        if (estop_latched_.load() || hardware_estop_latched_.load()) {
            safety.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
            safety.message = "estop latched";
        } else if (cmd_vel_expired_.load()) {
            safety.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
            safety.message = "cmd_vel expired; zero command sent";
        }

        add_kv(safety, "estop_latched", estop_latched_.load() ? "true" : "false");
        add_kv(safety, "hardware_estop_latched", hardware_estop_latched_.load() ? "true" : "false");
        add_kv(safety, "hardware_emergency_active", hardware_emergency_active_.load() ? "true" : "false");
        add_kv(safety, "f407_estop_blocked_commands",
               std::to_string(firmware_estop_blocked_command_count_.load()));
        add_kv(safety, "f407_safety_blocked_commands",
               std::to_string(firmware_estop_blocked_command_count_.load()));
        add_kv(safety, "last_safety_state_age_s",
               format_age(age_seconds(last_safety_state_ns_.load(), now_value)));
        add_kv(safety, "firmware_identity_valid",
               firmware_identity_valid_at(now_value) ? "true" : "false");
        add_kv(safety, "firmware_protocol_version",
               std::to_string(firmware_protocol_version_.load()));
        add_kv(safety, "firmware_capabilities",
               word_hex(firmware_capabilities_.load()));
        add_kv(safety, "firmware_build_id", std::to_string(firmware_build_id_.load()));
        add_kv(safety, "firmware_test_mode",
               std::to_string(static_cast<int>(firmware_test_mode_.load())));
        add_kv(safety, "firmware_hw_variant",
               std::to_string(static_cast<int>(firmware_hw_variant_.load())));
        add_kv(safety, "firmware_info_age_s",
               format_age(age_seconds(last_firmware_info_ns_.load(), now_value)));
        add_kv(safety, "require_firmware_identity", require_firmware_identity_ ? "true" : "false");
        add_kv(safety, "cmd_vel_expired", cmd_vel_expired_.load() ? "true" : "false");
        add_kv(safety, "cmd_vel_topic", cmd_vel_topic_);
        add_kv(safety, "cmd_vel_timeout_s", format_double(cmd_vel_timeout_s_));
        add_kv(safety, "max_linear_mps", format_double(max_linear_mps_));
        add_kv(safety, "max_angular_rps", format_double(max_angular_rps_));
        add_kv(safety, "lift_range_m",
               format_double(min_lift_height_m_) + ".." + format_double(max_lift_height_m_));
        add_kv(safety, "lift_arrival_tolerance_m", format_double(lift_arrival_tolerance_m_));
        add_kv(safety, "requested_lift_target_m", format_double(requested_lift_target_m_.load()));
        add_kv(safety, "last_lift_height_m", format_double(last_lift_height_m_.load()));
        add_kv(safety, "write_timeout_ms", std::to_string(write_timeout_.count()));
        add_kv(safety, "last_cmd_vel_rx_age_s",
               format_age(age_seconds(last_cmd_vel_rx_ns_.load(), now_value)));
        add_kv(safety, "last_cmd_vel_forward_age_s",
               format_age(age_seconds(last_cmd_vel_forward_ns_.load(), now_value)));
        add_kv(safety, "last_cmd_linear", format_double(last_cmd_linear_.load()));
        add_kv(safety, "last_cmd_angular", format_double(last_cmd_angular_.load()));
        add_kv(safety, "cmd_vel_timeouts", std::to_string(cmd_vel_timeout_count_.load()));
        add_kv(safety, "cmd_vel_blocked_by_estop", std::to_string(blocked_cmd_vel_count_.load()));
        add_kv(safety, "actuator_commands_blocked_by_estop",
               std::to_string(blocked_actuator_by_estop_count_.load()));
        add_kv(safety, "cmd_vel_blocked_by_firmware_identity",
               std::to_string(blocked_cmd_vel_by_firmware_identity_count_.load()));
        add_kv(safety, "actuator_commands_blocked_by_firmware_identity",
               std::to_string(blocked_actuator_by_firmware_identity_count_.load()));
        add_kv(safety, "invalid_cmd_vel", std::to_string(invalid_cmd_vel_count_.load()));
        add_kv(safety, "cmd_vel_over_limit", std::to_string(limited_cmd_vel_count_.load()));
        add_kv(safety, "rejected_lift_target", std::to_string(rejected_lift_target_count_.load()));
        add_kv(safety, "require_ack_for_services", require_ack_for_services_ ? "true" : "false");
        array.status.push_back(safety);

        diag_pub_->publish(array);
    }

    int64_t now_ns()
    {
        return this->get_clock()->now().nanoseconds();
    }

    static double age_seconds(int64_t stamp_ns, int64_t now_ns_value)
    {
        if (stamp_ns <= 0) {
            return -1.0;
        }
        return static_cast<double>(now_ns_value - stamp_ns) / 1e9;
    }

    static std::string format_double(double value, int precision = 3)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(precision) << value;
        return oss.str();
    }

    static std::string format_age(double value, int precision = 3)
    {
        if (value < 0.0) {
            return "n/a";
        }
        return format_double(value, precision);
    }

    static std::string byte_hex(uint8_t value)
    {
        std::ostringstream oss;
        oss << "0x" << std::uppercase << std::hex << std::setw(2)
            << std::setfill('0') << static_cast<int>(value);
        return oss.str();
    }

    static std::string word_hex(uint16_t value)
    {
        std::ostringstream oss;
        oss << "0x" << std::uppercase << std::hex << std::setw(4)
            << std::setfill('0') << static_cast<unsigned int>(value);
        return oss.str();
    }

    static std::string down_type_name(DownType type)
    {
        switch (type) {
            case DownType::CMD_VEL: return "CMD_VEL";
            case DownType::SET_LIFT_HEIGHT: return "SET_LIFT_HEIGHT";
            case DownType::SET_ELECTROMAGNET: return "SET_ELECTROMAGNET";
            case DownType::LIFT_HOME: return "LIFT_HOME";
            case DownType::EMERGENCY_STOP: return "EMERGENCY_STOP";
            case DownType::CLEAR_ESTOP: return "CLEAR_ESTOP";
            case DownType::HEARTBEAT: return "HEARTBEAT";
            default: return "UNKNOWN_DOWN_TYPE";
        }
    }

    static void add_kv(
        diagnostic_msgs::msg::DiagnosticStatus & status,
        const std::string & key,
        const std::string & value)
    {
        diagnostic_msgs::msg::KeyValue kv;
        kv.key = key;
        kv.value = value;
        status.values.push_back(kv);
    }

    void record_write_error(const std::string & error)
    {
        write_error_count_.fetch_add(1);
        record_last_error("write: " + error);
    }

    void record_last_error(const std::string & error)
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        last_error_ = error;
    }

    std::string get_last_error()
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        return last_error_.empty() ? "none" : last_error_;
    }

    // ==================== 成员 ====================
    int fd_ = -1;
    std::string port_name_;
    std::string base_frame_, odom_frame_, imu_frame_, cmd_vel_topic_;
    bool publish_tf_ = true;
    double cmd_vel_timeout_s_ = 0.60;
    double diagnostics_hz_ = 1.0;
    double rx_stale_timeout_s_ = 1.0;
    double max_linear_mps_ = 0.25;
    double max_angular_rps_ = 1.20;
    double min_lift_height_m_ = 0.0;
    double max_lift_height_m_ = 0.20;
    double lift_arrival_tolerance_m_ = 0.015;
    double lift_arrival_default_timeout_s_ = 30.0;
    bool require_ack_for_services_ = false;
    bool require_firmware_identity_ = true;
    double firmware_identity_stale_s_ = 3.0;
    bool gate_invalid_imu_ = true;
    double imu_accel_norm_min_mps2_ = 5.0;
    double imu_accel_norm_max_mps2_ = 15.0;
    uint32_t imu_min_valid_samples_ = 5;
    std::chrono::milliseconds ack_timeout_{300};
    std::chrono::milliseconds write_timeout_{50};

    std::mutex write_mutex_;
    std::mutex ack_mutex_;
    std::mutex lift_telemetry_mutex_;
    std::condition_variable ack_cv_;
    std::array<AckEvent, 256> last_ack_by_type_{};
    std::atomic<uint64_t> ack_sequence_{0};

    std::mutex error_mutex_;
    std::string last_error_;

    std::thread rx_thread_;
    std::atomic<bool> rx_thread_running_{false};

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_raw_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr imu_valid_pub_;
    rclcpp::Publisher<my_robot_msgs::msg::LiftStatus>::SharedPtr lift_pub_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr estop_state_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr cmd_vel_expired_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr firmware_identity_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr firmware_info_pub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr lift_target_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr estop_sub_;

    rclcpp::Service<my_robot_msgs::srv::SetElectromagnet>::SharedPtr set_em_srv_;
    rclcpp::Service<my_robot_msgs::srv::SetLiftHeight>::SharedPtr set_lift_height_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr lift_home_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr estop_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_estop_srv_;

    rclcpp::TimerBase::SharedPtr heartbeat_timer_;
    rclcpp::TimerBase::SharedPtr cmd_vel_watchdog_timer_;
    rclcpp::TimerBase::SharedPtr diagnostics_timer_;
    rclcpp::CallbackGroup::SharedPtr service_callback_group_;
    rclcpp::CallbackGroup::SharedPtr timer_callback_group_;

    std::atomic<bool> estop_latched_{false};
    std::atomic<bool> hardware_estop_latched_{false};
    std::atomic<bool> hardware_emergency_active_{false};
    std::atomic<bool> cmd_vel_expired_{false};
    std::atomic<int64_t> last_cmd_vel_rx_ns_{0};
    std::atomic<int64_t> last_cmd_vel_forward_ns_{0};
    std::atomic<int64_t> last_rx_ns_{0};
    std::atomic<int64_t> last_odom_ns_{0};
    std::atomic<int64_t> last_telem_ns_{0};
    std::atomic<int64_t> last_safety_state_ns_{0};
    std::atomic<int64_t> last_firmware_info_ns_{0};
    std::atomic<int64_t> last_ack_ns_{0};
    std::atomic<int64_t> last_ack_timeout_ns_{0};
    std::atomic<float> last_cmd_linear_{0.0f};
    std::atomic<float> last_cmd_angular_{0.0f};
    std::atomic<double> requested_lift_target_m_{0.0};
    std::atomic<double> last_lift_height_m_{std::numeric_limits<double>::quiet_NaN()};
    std::atomic<double> last_lift_velocity_mps_{0.0};
    std::atomic<bool> last_lift_moving_{false};
    std::atomic<uint8_t> last_ack_for_type_{0};
    std::atomic<uint8_t> last_ack_status_{0};
    std::atomic<uint16_t> firmware_protocol_version_{0};
    std::atomic<uint16_t> firmware_capabilities_{0};
    std::atomic<uint32_t> firmware_build_id_{0};
    std::atomic<uint8_t> firmware_test_mode_{255};
    std::atomic<uint8_t> firmware_hw_variant_{0};
    std::atomic<bool> imu_valid_{false};
    std::atomic<uint32_t> imu_consecutive_valid_samples_{0};
    std::atomic<double> last_imu_accel_norm_mps2_{std::numeric_limits<double>::quiet_NaN()};

    std::atomic<uint64_t> rx_byte_count_{0};
    std::atomic<uint64_t> rx_frame_count_{0};
    std::atomic<uint64_t> tx_frame_count_{0};
    std::atomic<uint64_t> odom_frame_count_{0};
    std::atomic<uint64_t> telem_frame_count_{0};
    std::atomic<uint64_t> imu_raw_frame_count_{0};
    std::atomic<uint64_t> imu_valid_frame_count_{0};
    std::atomic<uint64_t> safety_state_frame_count_{0};
    std::atomic<uint64_t> firmware_info_frame_count_{0};
    std::atomic<uint64_t> ack_frame_count_{0};
    std::atomic<uint64_t> checksum_error_count_{0};
    std::atomic<uint64_t> read_error_count_{0};
    std::atomic<uint64_t> write_error_count_{0};
    std::atomic<uint64_t> stm_error_count_{0};
    std::atomic<uint64_t> ack_timeout_count_{0};
    std::atomic<uint64_t> ack_error_count_{0};
    std::atomic<uint64_t> cmd_vel_timeout_count_{0};
    std::atomic<uint64_t> blocked_cmd_vel_count_{0};
    std::atomic<uint64_t> invalid_cmd_vel_count_{0};
    std::atomic<uint64_t> limited_cmd_vel_count_{0};
    std::atomic<uint64_t> rejected_lift_target_count_{0};
    std::atomic<uint64_t> blocked_actuator_by_estop_count_{0};
    std::atomic<uint64_t> blocked_cmd_vel_by_firmware_identity_count_{0};
    std::atomic<uint64_t> blocked_actuator_by_firmware_identity_count_{0};
    std::atomic<uint16_t> firmware_estop_blocked_command_count_{0};
};

}  // namespace my_robot_drivers

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<my_robot_drivers::SerialF407Node>();
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
        executor.add_node(node);
        executor.spin();
    } catch (const std::exception & e) {
        std::cerr << "FATAL: " << e.what() << std::endl;
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
