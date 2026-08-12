// serial_protocol.hpp — 0xAA55 多类型帧协议定义.
//
// 跟 my_robot_drivers/CLAUDE.md 中的协议规约保持同步, 改这里也要改那里.
//
// 帧格式:
//   [0xAA] [0x55] [type u8] [len u8] [payload ...] [checksum u8]
//
//   - 头 2 字节 0xAA 0x55 同步
//   - type:    指明 payload 含义 (上行/下行各自定义)
//   - len:     payload 字节数 (0..251)
//   - payload: 按 type 解析
//   - checksum: 简单 sum 8-bit (从 0xAA 起到 payload 末)
//
// payload 中 float 用小端 (STM32 + ARM64 都默认小端, 不用转).
//
// 兼容 v1 demo 协议: type=0x01 上行/下行 payload 跟 v1 完全一致.

#ifndef MY_ROBOT_DRIVERS__SERIAL_PROTOCOL_HPP_
#define MY_ROBOT_DRIVERS__SERIAL_PROTOCOL_HPP_

#include <cstdint>

namespace my_robot_drivers {

constexpr uint8_t HEADER_0 = 0xAA;
constexpr uint8_t HEADER_1 = 0x55;

// 帧的最大尺寸 (含头/type/len/checksum), 防止 buffer 溢出
constexpr size_t MAX_FRAME_SIZE = 256;

constexpr uint16_t TARGET_FIRMWARE_PROTOCOL_VERSION = 2;
constexpr uint16_t TARGET_FIRMWARE_CAPABILITIES = 0x003F;
constexpr uint32_t TARGET_FIRMWARE_BUILD_ID = 2026071907u;
constexpr uint8_t TARGET_FIRMWARE_TEST_MODE = 0;
constexpr uint8_t TARGET_FIRMWARE_HW_VARIANT = 1;

// ============== 上行 (STM32 → ROS2) ==============

enum class UpType : uint8_t {
    BASIC_ODOM      = 0x01,  // 兼容 v1: x/y/vx/wz/yaw_deg 5 个 float
    EXT_TELEMETRY   = 0x02,  // 升降台 + 电磁铁 + 限位 + IMU
    SAFETY_STATE    = 0x03,  // F407 固件级急停锁存和拒绝计数
    FIRMWARE_INFO   = 0x04,  // 只读协议版本/能力/构建身份
    ACK             = 0x10,  // 对下行命令的 ACK
    ERROR           = 0x1F,  // STM32 报错 (固件 bug / 越限 / 通信丢失)
};

#pragma pack(push, 1)

// type=0x01 上行 payload (20 字节, 跟 v1 兼容)
struct PayloadBasicOdom {
    float x;          // m
    float y;          // m
    float vx;         // m/s, 线速度
    float wz;         // rad/s, 角速度 (注意: v1 是 deg/s, 这里改 rad/s, 跟 ROS2 习惯一致)
                      //                如果 STM32 端来不及改, ROS2 端这里 *=π/180 转换
    float yaw_deg;    // °, 当前 yaw 角 (历史 v1 是 deg, 沿用)
};

// type=0x02 上行 payload (44 字节)
struct PayloadExtTelemetry {
    float    lift_height_m;       // 升降台当前高度 (m)
    float    lift_velocity_mps;   // 升降速度 (m/s)
    uint8_t  home_switch;         // 0/1
    uint8_t  top_switch;          // 0/1
    uint8_t  electromagnet_state; // 0/1
    uint8_t  homed;               // 0/1, 已 home 过则高度可信
    float    accel_x;             // m/s^2 (IMU)
    float    accel_y;
    float    accel_z;
    float    gyro_x;              // rad/s
    float    gyro_y;
    float    gyro_z;
    float    cpu_temp_c;          // STM32 内部温度
    float    bus_voltage_v;       // 总线电压, 监测掉电
};

// type=0x03 上行 payload (4 字节)
struct PayloadSafetyState {
    uint8_t  estop_latched;          // 固件级锁存, 仅 CLEAR_ESTOP 可清除
    uint8_t  emergency_active;       // 显式急停或心跳超时导致的总停机状态
    uint16_t blocked_command_count;  // 急停期间被固件拒绝的命令累计数
};

// type=0x04 上行 payload (12 字节)
struct PayloadFirmwareInfo {
    uint16_t protocol_version;
    uint16_t capabilities;
    uint32_t build_id;
    uint8_t  test_mode;
    uint8_t  hw_variant;
    uint16_t reserved;
};

// type=0x10 上行 payload (3 字节)
struct PayloadAck {
    uint8_t  ack_for_type;   // 这个 ACK 对应哪个下行 type
    uint8_t  status;         // 0=ok, 非 0 = error code
    uint8_t  reserved;
};

// type=0x1F 上行 payload (变长)
struct PayloadError {
    uint8_t  error_code;
    char     message[32];    // 不一定填满, len 给出实际长度
};

// ============== 下行 (ROS2 → STM32) ==============

enum class DownType : uint8_t {
    CMD_VEL          = 0x01,  // 兼容 v1: linear_v + angular_w
    SET_LIFT_HEIGHT  = 0x02,  // 升降台目标高度
    SET_ELECTROMAGNET = 0x03, // 电磁铁 on/off
    LIFT_HOME        = 0x04,  // 让升降台 home (向下到限位)
    EMERGENCY_STOP   = 0x10,  // 急停 (轮子 + 升降台 + 电磁铁断电)
    CLEAR_ESTOP      = 0x11,  // 显式清除 F407 固件级急停锁存
    HEARTBEAT        = 0xFF,  // 心跳, 让 STM32 知道 ROS2 还活
};

// type=0x01 下行 payload (8 字节, 跟 v1 兼容)
struct PayloadCmdVel {
    float linear_v;    // m/s
    float angular_w;   // rad/s
};

// type=0x02 下行 payload (4 字节)
struct PayloadSetLiftHeight {
    float target_height_m;
};

// type=0x03 下行 payload (1 字节)
struct PayloadSetElectromagnet {
    uint8_t turn_on;   // 0/1
};

#pragma pack(pop)

// ============== 校验工具 ==============

// 计算简单 sum 8-bit checksum.
//   data[0..len-1] 包含: 0xAA 0x55 type len payload
//   checksum 是从 data[0] 累加到 data[len-1] (含 payload), 不包含 checksum 字节本身
inline uint8_t compute_checksum(const uint8_t * data, size_t len)
{
    uint8_t sum = 0;
    for (size_t i = 0; i < len; ++i) {
        sum = static_cast<uint8_t>(sum + data[i]);
    }
    return sum;
}

}  // namespace my_robot_drivers

#endif  // MY_ROBOT_DRIVERS__SERIAL_PROTOCOL_HPP_
