#ifndef ODOM_H
#define ODOM_H

#include <stdint.h>

/* TODO: 用户改这两个机械参数 */
#define WHEEL_RADIUS_M   0.0325f      /* 65mm 轮 → 0.0325; 改成你实际轮半径 (m) */
#define WHEEL_BASE_M     0.30f        /* 左右轮中心距 (m), 跟车实际量 */

/* ★ 固定巡航速度 (2026-06-11 用户定): 任意非零 cmd_vel 都把"主轮"归一化到这个
 * 脉冲率, 速度恒定, 方向/转向比例仍由 cmd_vel 决定 (前进=直走, 纯转=原地转).
 *   500 pps ≈ 0.064 m/s (6.4 cm/s) — 2026-07-18 真车调速确认值.
 * 用户在 Keil TEST_MODE 2 调出来的"最好的速度", 现在让正常 ROS 模式也按它跑.
 * 设 0 = 关闭固定巡航, 回到 cmd_vel 比例变速 (Nav2 全速度域).
 * 改速度只改这一个数 + 重烧 (200=慢稳, 800≈0.1m/s, 1600≈0.2m/s). */
#define CRUISE_PPS       500

/* 差速 cmd_vel → 4 轮速度反解算 (pps).
 *   M1+M3 = 左侧两轮 (并联同速)
 *   M2+M4 = 右侧两轮 (并联同速)
 * 输出: pps[0..3] = M1..M4 的 pulse/sec, 带符号 (正 = "前进方向"; M2/M4 的物理反向另由 bsp_motor MOTOR_INVERT 处理)
 */
void odom_cmdvel_to_wheels(float linear_v, float angular_w, int32_t pps_out[4]);

/* 用 4 路 step delta 推算 odom. 在主循环中以固定 dt 调用 (50Hz 推荐).
 *   step_delta_m1..m4: 自上次调用以来的步数变化 (来自 bsp_motor_take_step_delta)
 *   dt_s: 经过的秒数
 * 内部更新 x, y, yaw (基于 IMU yaw 优先 / 没 IMU 则积分 gyro).
 * 如果 use_imu_yaw=1 且 imu_yaw_valid=1, 直接用 imu_yaw_rad 覆盖. 否则推算积分.
 */
void odom_update(int32_t d_m1, int32_t d_m2, int32_t d_m3, int32_t d_m4, float dt_s,
                 uint8_t use_imu_yaw, uint8_t imu_yaw_valid, float imu_yaw_rad);

/* 取当前状态. yaw 单位是 ° (ROS 端协议 PayloadBasicOdom 用 deg). */
void odom_get(float *x_m, float *y_m, float *vx_mps, float *wz_radps, float *yaw_deg);

/* 急停时清速度, 不动 x/y/yaw */
void odom_zero_velocity(void);

#endif
