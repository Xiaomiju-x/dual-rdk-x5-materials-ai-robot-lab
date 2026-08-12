#ifndef BSP_IMU_H
#define BSP_IMU_H

#include <stdint.h>

typedef struct {
    /* 加速度, m/s^2 (转过的, 不是 raw int16) */
    float accel_x, accel_y, accel_z;
    /* 角速度, rad/s */
    float gyro_x,  gyro_y,  gyro_z;
    /* 欧拉角, ° (维特智能原始就是 °, 转给 ROS2 时用 yaw_deg 字段直接发) */
    float roll_deg, pitch_deg, yaw_deg;
    /* 模块自检温度 (°C, 仅诊断) */
    float temp_c;

    /* 最近一次收到 0x53 角度帧的 millis() 时间. 用来判 IMU 通讯是不是活着 */
    uint32_t last_angle_ms;
} ImuData;

void  bsp_imu_init(void);
/* 在主循环里 (例如每 2ms) 调一次, 内部从 USART2 ring buffer 取数据状态机解析.
 * `now_ms` 传 millis() 用于打 last_angle_ms 时间戳. */
void  bsp_imu_poll(uint32_t now_ms);
const ImuData* bsp_imu_data(void);

#endif
