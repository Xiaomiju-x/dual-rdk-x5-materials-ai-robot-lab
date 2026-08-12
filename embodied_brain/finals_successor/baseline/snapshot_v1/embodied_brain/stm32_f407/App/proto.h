#ifndef PROTO_H
#define PROTO_H

#include <stdint.h>

/* ============================================================
 * 0xAA55 帧协议, 字节级对齐
 *   embodied_brain/ros2_ws/src/my_robot_drivers/include/my_robot_drivers/serial_protocol.hpp
 * 改这边一定要同步改那边.
 * ============================================================ */

/* 帧头 */
#define PROTO_HDR0   0xAAu
#define PROTO_HDR1   0x55u
#define PROTO_FRAME_MAX  256

/* 上行 type */
#define UP_BASIC_ODOM      0x01u
#define UP_EXT_TELEMETRY   0x02u
#define UP_SAFETY_STATE    0x03u
#define UP_FIRMWARE_INFO   0x04u
#define UP_ACK             0x10u
#define UP_ERROR           0x1Fu

/* 只读固件身份和能力. X5 在发送任何非安全方向命令前必须精确匹配. */
#define PROTO_PROTOCOL_VERSION       2u
#define PROTO_CAP_ESTOP_LATCH        0x0001u
#define PROTO_CAP_SAFETY_STATE       0x0002u
#define PROTO_CAP_ACK_STATUS         0x0004u
#define PROTO_CAP_EXT_TELEMETRY      0x0008u
#define PROTO_CAP_LINK_STALE_GUARD   0x0010u
#define PROTO_CAP_FIRMWARE_IDENTITY  0x0020u
#define PROTO_CAPABILITIES           0x003Fu
#define PROTO_FIRMWARE_BUILD_ID      2026071907UL
#define PROTO_REQUIRED_TEST_MODE     0u
#define PROTO_HW_VARIANT             1u

/* 下行 type */
#define DN_CMD_VEL          0x01u
#define DN_SET_LIFT_HEIGHT  0x02u
#define DN_SET_ELECTROMAGNET 0x03u
#define DN_LIFT_HOME        0x04u
#define DN_EMERGENCY_STOP   0x10u
#define DN_CLEAR_ESTOP      0x11u
#define DN_HEARTBEAT        0xFFu

#define PROTO_ACK_OK             0u
#define PROTO_ACK_UNSUPPORTED    1u
#define PROTO_ACK_BAD_LENGTH     2u
#define PROTO_ACK_ESTOP_LATCHED  3u
#define PROTO_ACK_LINK_STALE     4u
#define PROTO_COMMAND_LINK_TIMEOUT_MS 1000u

/* 全局协议状态 (主循环用) */
typedef struct {
    /* 来自 CMD_VEL 的最新值 */
    float    target_linear_v;
    float    target_angular_w;
    uint32_t last_cmd_vel_ms;

    /* 来自 HEARTBEAT 的最新时间. main 用它做超时急停判断 */
    uint32_t last_heartbeat_ms;

    /* 来自 EMERGENCY_STOP 的标志 (主循环捕获后清零) */
    uint8_t  emergency_stop_request;

    /* 固件级急停锁存, 只能由显式 DN_CLEAR_ESTOP 清除. */
    uint8_t  estop_latched;

    /* 急停或心跳过期时被固件拒绝的非安全方向命令计数. */
    uint16_t estop_blocked_command_count;
} ProtoState;

void          proto_init(void);
ProtoState*   proto_state(void);
uint8_t       proto_fixture_busy(void);
float         proto_fixture_height_m(void);

/* 喂收到的串口字节流, 内部状态机驱动, 完整帧后调用相应 handler */
void          proto_feed_rx(const uint8_t *data, uint16_t len, uint32_t now_ms);

/* 主动发上行 */
void          proto_send_basic_odom(float x, float y, float vx, float wz, float yaw_deg);
void          proto_send_ext_telemetry(float lift_h, float lift_v,
                                       uint8_t home_sw, uint8_t top_sw,
                                       uint8_t em_state, uint8_t homed,
                                       float ax, float ay, float az,
                                       float gx, float gy, float gz,
                                       float cpu_temp_c, float bus_v);
void          proto_send_safety_state(uint8_t estop_latched,
                                      uint8_t emergency_active,
                                      uint16_t blocked_command_count);
void          proto_send_firmware_info(uint16_t protocol_version,
                                       uint16_t capabilities,
                                       uint32_t build_id,
                                       uint8_t test_mode,
                                       uint8_t hw_variant);
void          proto_send_ack(uint8_t ack_for_type, uint8_t status);
void          proto_send_error(uint8_t code, const char *msg);

#endif
