#ifndef BSP_MOTOR_H
#define BSP_MOTOR_H

#include <stdint.h>

/* ============================================================
 * 4 个 X57S 闭环步进电机, 全部接在 TIM1 的 4 个通道上.
 * 引脚 (用户实接):
 *   M1: PUL=PE9  (TIM1_CH1 AF1)  DIR=PE1  EN=PE2  UART=USART2 (PA2/PA3)
 *   M2: PUL=PE11 (TIM1_CH2 AF1)  DIR=PE3  EN=PE4  UART=UART4  (PA0/PA1)
 *   M3: PUL=PE13 (TIM1_CH3 AF1)  DIR=PE5  EN=PE6  UART=USART3 (PD8/PD9)
 *   M4: PUL=PE14 (TIM1_CH4 AF1)  DIR=PE7  EN=PE8  UART=USART6 (PC6/PC7)
 *
 * EN 电平约定: 输出 LOW = 使能 (X57S 默认), 输出 HIGH = 失能 (脱机).
 * 如果实测反了, 改 EN_ACTIVE_LOW = 0.
 * ============================================================ */

/* X57S OLED 上的"细分"参数, 默认 8 细分 → 1600 步/圈. 改这里要跟驱动器对齐. */
#define STEPS_PER_REV     1600

/* 物理安装反向: 1 = 反转. 实测哪个轮转反就改 0/1.
 * 2026-06-11 实测: 发 +linear.x (ROS 前进) 车实际往后 → 整车前后反了 (base_link
 * 朝向装反 180°). 四个全翻 (0↔1) 把前后+转向一起纠正回 REP-103 (+x 前进/+z 左转 CCW). */
#define MOTOR_INVERT_M1   1
#define MOTOR_INVERT_M2   0
#define MOTOR_INVERT_M3   1
#define MOTOR_INVERT_M4   0

/* X57S EN 极性. 1=低电平有效 (拉低使能, 默认), 0=高电平有效 */
#define EN_ACTIVE_LOW     1

#define MOTOR_COUNT       4
#define MOTOR_M1          0
#define MOTOR_M2          1
#define MOTOR_M3          2
#define MOTOR_M4          3

/* 速度上限保护 (pps). 1m/s @ 65mm 轮 / 1600 步/圈 ≈ 7826 pps.
 * 2026-06-11 OC toggle 重写后, 每个脉冲 = 2 次 CC 中断:
 *   16000 pps × 4 路 × 2 = 128k IRQ/s 上限 (~4% CPU @168MHz), 够 2 m/s.
 * 别再调回 60000 — 那是 480k IRQ/s, 会饿死主循环.
 */
#define MOTOR_MAX_PPS     16000
/* 死区: |pps| < 这个值就停 */
#define MOTOR_MIN_PPS     10

void    bsp_motor_init(void);

/* 4 路 EN 控制 */
void    bsp_motor_enable (uint8_t idx);   /* 单路使能 */
void    bsp_motor_disable(uint8_t idx);
void    bsp_motor_enable_all (void);
void    bsp_motor_disable_all(void);

/* 设速 (pulse/sec, 带符号; 物理反向另由 MOTOR_INVERT_Mx 处理).
 * 2026-06-11 起为 OC Toggle 实现: 4 路频率完全独立 (Nav2 弧线 OK);
 * 同值重复调用被缓存吸收 (主循环每圈调也无害).
 */
void    bsp_motor_set_speed_pps(uint8_t idx, int32_t pps);

/* 紧急停止: 所有 CCR=0 + EN 失能 */
void    bsp_motor_emergency_stop(void);

/* 取步数 delta. 第一次调用 ≈ 上电至今. 调用后内部清零. */
int32_t bsp_motor_take_step_delta(uint8_t idx);

#endif
