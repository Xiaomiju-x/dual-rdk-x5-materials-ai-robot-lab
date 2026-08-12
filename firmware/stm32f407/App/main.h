#ifndef MAIN_H
#define MAIN_H

/* 任务节拍 */
#define LOOP_DT_MS              2           /* 主循环每 2ms 一次. UART 解析 + IMU poll. */
#define ODOM_PUBLISH_HZ         50
#define EXT_TELEMETRY_HZ        20

/* 超时 */
#define HEARTBEAT_TIMEOUT_MS    1000        /* 1s 无心跳 → 急停 */
#define CMD_VEL_FRESH_MS        500         /* 0.5s 无 cmd_vel → 速度衰减到 0 */

/* 启动后多少毫秒才打开"超时急停" (上电时 ROS2 还没起来) */
#define BOOT_GRACE_MS           3000

#endif
