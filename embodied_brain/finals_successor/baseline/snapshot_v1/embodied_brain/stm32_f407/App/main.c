/* ============================================================
 * 底盘下位机主程序
 *
 * TEST_MODE 选择 (改下面这个数字):
 *   0 = 正常模式 (0xAA55 协议 + IMU + 50Hz odom 上行)
 *   1 = 只 M1 满速转 (空载电流测试)
 *   2 = 4 路同速前进 (带载真实场景 — 车会动!)
 *   3 = 升降台串口诊断 (X5 通过 USART2 发 ASCII 命令)
 *   4 = 升降台 42 电机上电自转测试 (不需要 X5)
 *   5 = 初赛升降台降级演示流程 (不需要 X5)
 *   6 = X42S UART PREP 撞顶清堵转/顶端保持测试 (不需要 X5)
 * ============================================================ */
#define TEST_MODE       0
/* TEST_MODE 3 = lift-stage bench firmware:
 * USART2 PD5/PD6 ASCII commands, TIM8_CH4 PC9 pulse, PD7 direction, PD10 enable.
 * Set this back to 0 when rebuilding the normal chassis firmware.
 */

/* mode=4 用: 升降台 42 步进电机独立持续转动测试.
 * 不走 X5 串口控制; 上电后 F407 直接输出 EN/DIR/PUL.
 * 当前现场实测 en_hi=0 有力矩, en_hi=1 会导致 F407 复位, 所以默认 0.
 * 这里改成和 57 电机相同的 OC Toggle 连续转方式, 带简单加速斜坡.
 */
#define LIFT_SPIN_TARGET_PPS        6400
#define LIFT_SPIN_RAMP_START_PPS    200
#define LIFT_SPIN_RAMP_STEP_PPS     200
#define LIFT_SPIN_RAMP_HOLD_MS      250U

/* mode=6 用: 复用已调通 ZDT X42S PREP 参数。
 * UART5 是升降电机专用总线，所以用广播地址 0 避免依赖现场电机 ID。
 * 撞顶后清保护并锁轴保持，再回到离机械底端约 1-2cm 的软底位置。
 */
#define LIFT_ZDT_ADDR                 0U
#define LIFT_ZDT_UP_DIR               1U
#define LIFT_ZDT_UP_RPM               150U
#define LIFT_ZDT_UP_ACC               50U
#define LIFT_ZDT_UP_PULSES            15000UL
#define LIFT_ZDT_PREP_DURATION_MS     2800U
#define LIFT_ZDT_CLEAR_PERIOD_MS      10U
#define LIFT_ZDT_TOP_HOLD_MS          3000U
#define LIFT_ZDT_DOWN_DIR             0U
#define LIFT_ZDT_DOWN_PULSES          10000UL
#define LIFT_ZDT_DOWN_WAIT_MS         4500U
#define LIFT_DEMO_TOP_WAIT_MS         10000U
#define LIFT_DEMO_FINAL_RETRACT_MS    (2U * LIFT_DEMO_ACT_RETRACT_MS)

/* mode=1 用: M1 单独测试速度 (pps) */
#define TEST_M1_PPS     6400

/* mode=2 用: 4 路同步速度 (pps). @ 65mm 轮 / 1600 步/圈:
 *    100  ≈ 0.0125 m/s (蹭着走 — 极限慢, 可能抖)
 *    200  ≈ 0.025 m/s  (★ 实验室慢速 — "刚能动起来" 的稳定值)
 *    400  ≈ 0.05 m/s   (很慢)
 *    800  ≈ 0.10 m/s   (慢走)
 *   1600  ≈ 0.20 m/s   (慢走 — 带载首测最安全)
 *   3200  ≈ 0.41 m/s   (轻快走)
 *   6400  ≈ 0.82 m/s   (慢跑)
 *  16000  ≈ 2.0 m/s    (快, 留刹车空间)
 * 注意 TIM1 共享 ARR, 4 路必须同频率. 此模式都同速所以 OK.
 * 哪个轮转反就改 bsp_motor.h 的 MOTOR_INVERT_Mx (默认 M2/M4=1 右侧镜像)
 *
 * 实验室狭窄空间用 200 pps (2.5 cm/s) — 这个速度下闭环驱动器仍稳, 不会失步抖动.
 * 如果还嫌快, 降到 100; 如果发现走得太一卡一卡, 升回 400.
 */
#define TEST_4WHEEL_PPS 200

/* ============================================================ */

#include "stm32f4xx.h"
#include "main.h"
#include "bsp_clock.h"
#include "bsp_systick.h"
#include "bsp_led.h"
#include "bsp_uart.h"
#include "bsp_motor.h"
#include "bsp_lift.h"
#include "bsp_imu.h"
#include "odom.h"
#include "proto.h"
#include <stdlib.h>
#include <string.h>

/* Cooperative fixture sequencer implemented by proto.c. */
void proto_service(uint32_t now_ms);

#if TEST_MODE == 1

/* ============================================================
 * mode=1: 只跑 M1, 其他失能 (空载电流测试)
 * ============================================================ */
int main(void)
{
    bsp_clock_init();
    bsp_systick_init();
    bsp_led_init();
    bsp_motor_init();

    /* 只使能 M1, 其他保持失能 */
    bsp_motor_disable_all();
    bsp_motor_enable(MOTOR_M1);

    /* 给 X57S 内部一些时间识别 EN 信号 */
    delay_ms(100);

    /* M1 满速持续转, 正方向 */
    bsp_motor_set_speed_pps(MOTOR_M1, TEST_M1_PPS);

    bsp_led_set_mode(LED_MODE_FAST_BLINK);

    for (;;) {
        delay_ms(1);
        bsp_led_tick_1ms();
    }
}

#elif TEST_MODE == 2

/* ============================================================
 * mode=2: 4 路同速前进, 带载真实场景 (车会动!)
 *
 * 测什么:
 *   1. 4 路同时跑能不能稳, 24V PSU 够不够拉 (4× 静态电流 + 启动尖峰)
 *   2. 哪个轮转反 → 改 bsp_motor.h MOTOR_INVERT_Mx (0/1)
 *   3. 长时间运行驱动器/电机温升
 *
 * 提速梯度: TEST_4WHEEL_PPS 默认 3200 (0.4m/s), 安全后改 6400/16000.
 * ============================================================ */
int main(void)
{
    bsp_clock_init();
    bsp_systick_init();
    bsp_led_init();
    bsp_motor_init();

    bsp_motor_enable_all();
    delay_ms(100);                   /* X57S 内部识别 EN 信号 */

    /* 4 路同正向. 物理反向由 MOTOR_INVERT_Mx 处理 */
    bsp_motor_set_speed_pps(MOTOR_M1, TEST_4WHEEL_PPS);
    bsp_motor_set_speed_pps(MOTOR_M2, TEST_4WHEEL_PPS);
    bsp_motor_set_speed_pps(MOTOR_M3, TEST_4WHEEL_PPS);
    bsp_motor_set_speed_pps(MOTOR_M4, TEST_4WHEEL_PPS);

    bsp_led_set_mode(LED_MODE_FAST_BLINK);

    for (;;) {
        delay_ms(1);
        bsp_led_tick_1ms();
    }
}

#elif TEST_MODE == 3

/* ============================================================
 * mode=3: lift-stage bench firmware.
 *
 * Host UART: USART2 PD5/PD6, 115200 8N1, ASCII line protocol.
 * Lift driver: EN=PD10, DIR=PD7, PUL=PC9/TIM8_CH4.
 *
 * Commands:
 *   HELP
 *   PING
 *   STATUS
 *   EN 0|1
 *   ZERO
 *   SAFEZERO [bottom_margin_steps] [pps]
 *   STOP
 *   JOGUP [steps] [pps]
 *   JOGDOWN [steps] [pps]
 *   UP [steps] [pps]
 *   DOWN [steps] [pps]
 *   GOTO <target_steps> [pps]
 *   SPEED <pps>
 *   CYCLE [up_steps] [pps]
 *   RAW <en_hi> <dir_hi> <pul_hi>
 *   BITPULSE <en_hi> <dir_hi> <pulses> <half_ms>
 *
 * No limit switches are installed in this first pass. Use small step counts
 * first, keep one hand on power, and tune DEFAULT_UP_STEPS after observing
 * the real travel.
 * ============================================================ */

static void lift_puts(const char *s)
{
    host_uart_send((const uint8_t *)s, (uint16_t)strlen(s));
}

static long lift_arg_or_default(char **p, long def)
{
    char *s = *p;
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '\0') {
        *p = s;
        return def;
    }
    {
        char *endp = s;
        long v = strtol(s, &endp, 10);
        if (endp == s) {
            *p = s;
            return def;
        }
        *p = endp;
        return v;
    }
}

static int lift_starts_with(const char *s, const char *prefix)
{
    while (*prefix) {
        char a = *s++;
        char b = *prefix++;
        if (a >= 'a' && a <= 'z') a = (char)(a - 'a' + 'A');
        if (a != b) return 0;
    }
    return 1;
}

static char *lift_skip_ws(char *p)
{
    while (*p == ' ' || *p == '\t') p++;
    return p;
}

static char *lift_append_str(char *p, const char *end, const char *s)
{
    while (*s && p < end) *p++ = *s++;
    return p;
}

static char *lift_append_u32(char *p, const char *end, uint32_t v)
{
    char tmp[10];
    uint8_t n = 0;
    if (v == 0) {
        if (p < end) *p++ = '0';
        return p;
    }
    while (v != 0 && n < sizeof(tmp)) {
        tmp[n++] = (char)('0' + (v % 10U));
        v /= 10U;
    }
    while (n != 0 && p < end) *p++ = tmp[--n];
    return p;
}

static char *lift_append_i32(char *p, const char *end, int32_t v)
{
    if (v < 0) {
        if (p < end) *p++ = '-';
        return lift_append_u32(p, end, (uint32_t)(-v));
    }
    return lift_append_u32(p, end, (uint32_t)v);
}

typedef enum {
    LIFT_AUTO_IDLE = 0,
    LIFT_AUTO_START,
    LIFT_AUTO_GRIP_WAIT,
    LIFT_AUTO_MOVE_UP,
    LIFT_AUTO_WAIT_UP,
    LIFT_AUTO_EXTEND_WAIT,
    LIFT_AUTO_SERVO_WAIT,
    LIFT_AUTO_RELEASE_WAIT,
    LIFT_AUTO_RETRACT_WAIT,
    LIFT_AUTO_HOME_WAIT,
    LIFT_AUTO_MOVE_DOWN,
    LIFT_AUTO_WAIT_DOWN
} LiftAutoState;

typedef struct {
    uint8_t active;
    LiftAutoState state;
    uint32_t deadline_ms;
    int32_t travel_steps;
    uint32_t pps;
    uint32_t extend_ms;
    uint32_t retract_ms;
    uint32_t hold_pps;
} LiftAutoCtx;

static LiftAutoCtx s_auto = {
    0,
    LIFT_AUTO_IDLE,
    0,
    LIFT_AUTO_TRAVEL_STEPS,
    LIFT_AUTO_SPEED_PPS,
    LIFT_AUTO_EXTEND_MS,
    LIFT_AUTO_RETRACT_MS,
    LIFT_AUTO_TOP_HOLD_PPS
};

typedef struct {
    uint8_t active;
    uint32_t deadline_ms;
    int32_t steps;
    uint32_t pps;
} LiftCycleCtx;

typedef enum {
    LIFT_DEMO_IDLE = 0,
    LIFT_DEMO_INIT_WAIT,
    LIFT_DEMO_PICK_WAIT,
    LIFT_DEMO_GRIP_WAIT,
    LIFT_DEMO_CARRY_WAIT,
    LIFT_DEMO_NAV_WAIT,
    LIFT_DEMO_RELEASE_SIDE_WAIT,
    LIFT_DEMO_MOVE_UP_WAIT,
    LIFT_DEMO_UP_DWELL_WAIT,
    LIFT_DEMO_MOVE_DOWN_WAIT,
    LIFT_DEMO_DOWN_DWELL_WAIT,
    LIFT_DEMO_RELEASE_WAIT
} LiftDemoState;

typedef struct {
    uint8_t active;
    LiftDemoState state;
    uint32_t deadline_ms;
    int32_t lift_steps;
    uint32_t pps;
    uint32_t nav_ms;
} LiftDemoCtx;

static uint8_t s_safezero_pending = 0;
static LiftCycleCtx s_cycle = { 0, 0, 0, 0 };
static LiftDemoCtx s_demo = { 0, LIFT_DEMO_IDLE, 0, 0, 0, 0 };

static int lift_reject_motion_if_busy(void);
static uint32_t lift_debug_pps(long pps, uint32_t def);

static int lift_time_reached(uint32_t now, uint32_t deadline)
{
    return ((int32_t)(now - deadline) >= 0);
}

static const char *lift_auto_name(void)
{
    if (s_safezero_pending) return "safezero";
    if (s_cycle.active) return "cycle";
    if (s_demo.active) return "demo";
    switch (s_auto.state) {
        case LIFT_AUTO_IDLE:         return "idle";
        case LIFT_AUTO_START:        return "start";
        case LIFT_AUTO_GRIP_WAIT:    return "grip";
        case LIFT_AUTO_MOVE_UP:      return "move_up";
        case LIFT_AUTO_WAIT_UP:      return "wait_up";
        case LIFT_AUTO_EXTEND_WAIT:  return "extend";
        case LIFT_AUTO_SERVO_WAIT:   return "servo";
        case LIFT_AUTO_RELEASE_WAIT: return "release";
        case LIFT_AUTO_RETRACT_WAIT: return "retract";
        case LIFT_AUTO_HOME_WAIT:    return "home";
        case LIFT_AUTO_MOVE_DOWN:    return "move_down";
        case LIFT_AUTO_WAIT_DOWN:    return "wait_down";
        default:                     return "unknown";
    }
}

static void lift_auto_abort(void)
{
    s_safezero_pending = 0;
    s_cycle.active = 0;
    s_demo.active = 0;
    s_demo.state = LIFT_DEMO_IDLE;
    s_auto.active = 0;
    s_auto.state = LIFT_AUTO_IDLE;
    bsp_lift_stop();
    bsp_lift_actuator_stop();
}

static void lift_auto_start(long steps, long pps, long extend_ms, long retract_ms, long hold_pps)
{
    if (steps < 0) steps = -steps;
    if (steps == 0) steps = LIFT_AUTO_TRAVEL_STEPS;
    if (pps < 0) pps = -pps;
    if (pps == 0) pps = LIFT_AUTO_SPEED_PPS;
    pps = (long)lift_debug_pps(pps, LIFT_AUTO_SPEED_PPS);
    if (extend_ms < 500) extend_ms = 500;
    if (extend_ms > 30000) extend_ms = 30000;
    if (retract_ms < 500) retract_ms = 500;
    if (retract_ms > 30000) retract_ms = 30000;
    (void)hold_pps;
    hold_pps = 0;  /* no continuous upward top-hold until limit inputs exist */

    s_auto.travel_steps = (int32_t)steps;
    s_auto.pps = (uint32_t)pps;
    s_auto.extend_ms = (uint32_t)extend_ms;
    s_auto.retract_ms = (uint32_t)retract_ms;
    s_auto.hold_pps = (uint32_t)hold_pps;
    s_auto.deadline_ms = 0;
    s_auto.state = LIFT_AUTO_START;
    s_auto.active = 1;
}

static void lift_auto_tick(void)
{
    if (!s_auto.active) return;

    uint32_t now = millis();
    switch (s_auto.state) {
        case LIFT_AUTO_START:
            bsp_lift_stop();
            bsp_lift_actuator_stop();
            bsp_lift_servo_home();
            bsp_lift_magnet_set(1);
            bsp_lift_zero_position();
            s_auto.deadline_ms = now + LIFT_AUTO_GRIP_MS;
            s_auto.state = LIFT_AUTO_GRIP_WAIT;
            lift_puts("AUTO grip\r\n");
            break;

        case LIFT_AUTO_GRIP_WAIT:
            if (lift_time_reached(now, s_auto.deadline_ms)) {
                s_auto.state = LIFT_AUTO_MOVE_UP;
            }
            break;

        case LIFT_AUTO_MOVE_UP:
            bsp_lift_move_steps(s_auto.travel_steps, s_auto.pps);
            s_auto.state = LIFT_AUTO_WAIT_UP;
            lift_puts("AUTO move-up\r\n");
            break;

        case LIFT_AUTO_WAIT_UP:
            if (!bsp_lift_busy()) {
                if (s_auto.hold_pps > 0U) {
                    bsp_lift_set_speed_pps((int32_t)s_auto.hold_pps);
                    lift_puts("AUTO top-hold\r\n");
                }
                bsp_lift_actuator_extend();
                s_auto.deadline_ms = now + s_auto.extend_ms;
                s_auto.state = LIFT_AUTO_EXTEND_WAIT;
                lift_puts("AUTO extend\r\n");
            }
            break;

        case LIFT_AUTO_EXTEND_WAIT:
            if (lift_time_reached(now, s_auto.deadline_ms)) {
                bsp_lift_actuator_stop();
                bsp_lift_servo_left();
                s_auto.deadline_ms = now + LIFT_AUTO_SERVO_MS;
                s_auto.state = LIFT_AUTO_SERVO_WAIT;
                lift_puts("AUTO servo-left\r\n");
            }
            break;

        case LIFT_AUTO_SERVO_WAIT:
            if (lift_time_reached(now, s_auto.deadline_ms)) {
                bsp_lift_magnet_set(0);
                s_auto.deadline_ms = now + LIFT_AUTO_RELEASE_MS;
                s_auto.state = LIFT_AUTO_RELEASE_WAIT;
                lift_puts("AUTO release\r\n");
            }
            break;

        case LIFT_AUTO_RELEASE_WAIT:
            if (lift_time_reached(now, s_auto.deadline_ms)) {
                bsp_lift_actuator_retract();
                s_auto.deadline_ms = now + s_auto.retract_ms;
                s_auto.state = LIFT_AUTO_RETRACT_WAIT;
                lift_puts("AUTO retract\r\n");
            }
            break;

        case LIFT_AUTO_RETRACT_WAIT:
            if (lift_time_reached(now, s_auto.deadline_ms)) {
                bsp_lift_actuator_stop();
                bsp_lift_servo_home();
                s_auto.deadline_ms = now + LIFT_AUTO_SERVO_MS;
                s_auto.state = LIFT_AUTO_HOME_WAIT;
                lift_puts("AUTO servo-home\r\n");
            }
            break;

        case LIFT_AUTO_HOME_WAIT:
            if (lift_time_reached(now, s_auto.deadline_ms)) {
                s_auto.state = LIFT_AUTO_MOVE_DOWN;
            }
            break;

        case LIFT_AUTO_MOVE_DOWN:
            bsp_lift_stop();          /* stop top-hold pulses before descending */
            bsp_lift_move_steps(-s_auto.travel_steps, s_auto.pps);
            s_auto.state = LIFT_AUTO_WAIT_DOWN;
            lift_puts("AUTO move-down\r\n");
            break;

        case LIFT_AUTO_WAIT_DOWN:
            if (!bsp_lift_busy()) {
                bsp_lift_zero_position();
                s_auto.active = 0;
                s_auto.state = LIFT_AUTO_IDLE;
                lift_puts("AUTO done\r\n");
            }
            break;

        default:
            lift_auto_abort();
            lift_puts("AUTO abort bad-state\r\n");
            break;
    }
}

static void lift_send_status(void)
{
    char buf[192];
    char *p = buf;
    const char *end = buf + sizeof(buf);
    p = lift_append_str(p, end, "STATUS en=");
    p = lift_append_u32(p, end, (uint32_t)bsp_lift_enabled());
    p = lift_append_str(p, end, " busy=");
    p = lift_append_u32(p, end, (uint32_t)bsp_lift_busy());
    p = lift_append_str(p, end, " pos=");
    p = lift_append_i32(p, end, bsp_lift_position_steps());
    p = lift_append_str(p, end, " rem=");
    p = lift_append_i32(p, end, bsp_lift_remaining_steps());
    p = lift_append_str(p, end, " mag=");
    p = lift_append_u32(p, end, (uint32_t)bsp_lift_magnet_on());
    p = lift_append_str(p, end, " act=");
    p = lift_append_u32(p, end, (uint32_t)bsp_lift_actuator_state());
    p = lift_append_str(p, end, " servo_us=");
    p = lift_append_u32(p, end, (uint32_t)bsp_lift_servo_pulse_us());
    p = lift_append_str(p, end, " auto=");
    p = lift_append_str(p, end, lift_auto_name());
    p = lift_append_str(p, end, "\r\n");
    host_uart_send((const uint8_t *)buf, (uint16_t)(p - buf));
}

static int lift_reject_motion_if_busy(void)
{
    if (bsp_lift_busy() || s_auto.active || s_safezero_pending ||
        s_cycle.active || s_demo.active) {
        lift_puts("ERR busy; send STOP or wait\r\n");
        lift_send_status();
        return 1;
    }
    return 0;
}

static uint32_t lift_debug_pps(long pps, uint32_t def)
{
    if (pps < 0) pps = -pps;
    if (pps == 0) pps = (long)def;
    if (pps < (long)LIFT_MIN_PPS) pps = LIFT_MIN_PPS;
    if (pps > (long)LIFT_DEBUG_MAX_PPS) pps = LIFT_DEBUG_MAX_PPS;
    return (uint32_t)pps;
}

static void lift_deferred_tick(uint32_t now)
{
    if (s_safezero_pending && !bsp_lift_busy()) {
        s_safezero_pending = 0;
        bsp_lift_zero_position();
        lift_puts("OK safezero zero\r\n");
    }

    if (s_cycle.active && lift_time_reached(now, s_cycle.deadline_ms)) {
        s_cycle.active = 0;
        bsp_lift_move_steps(s_cycle.steps, s_cycle.pps);
        lift_puts("OK cycle moving-up\r\n");
    }
}

static void lift_demo_start(long lift_steps, long pps, long nav_ms)
{
    if (lift_steps < 0) lift_steps = -lift_steps;
    if (lift_steps == 0) lift_steps = LIFT_DEMO_LIFT_STEPS;
    if (lift_steps > LIFT_SAFE_ZERO_MARGIN_STEPS) lift_steps = LIFT_SAFE_ZERO_MARGIN_STEPS;
    pps = (long)lift_debug_pps(pps, LIFT_DEMO_SPEED_PPS);
    if (nav_ms < 0) nav_ms = LIFT_DEMO_NAV_DELAY_MS;
    if (nav_ms > 30000) nav_ms = 30000;

    lift_auto_abort();
    bsp_lift_enable(1);
    bsp_lift_actuator_stop();
    bsp_lift_magnet_set(0);
    bsp_lift_servo_home();         /* initial/transport position: left */
    lift_puts("DEMO init-left mag-off\r\n");
    s_demo.active = 1;
    s_demo.state = LIFT_DEMO_INIT_WAIT;
    s_demo.deadline_ms = millis() + LIFT_DEMO_SERVO_MS;
    s_demo.lift_steps = (int32_t)lift_steps;
    s_demo.pps = (uint32_t)pps;
    s_demo.nav_ms = (uint32_t)nav_ms;
}

static void lift_demo_tick(uint32_t now)
{
    if (!s_demo.active) return;

    switch (s_demo.state) {
        case LIFT_DEMO_INIT_WAIT:
            if (!lift_time_reached(now, s_demo.deadline_ms)) break;
            bsp_lift_servo_right();
            lift_puts("DEMO pick-right\r\n");
            s_demo.deadline_ms = now + LIFT_DEMO_SERVO_MS;
            s_demo.state = LIFT_DEMO_PICK_WAIT;
            break;
        case LIFT_DEMO_PICK_WAIT:
            if (!lift_time_reached(now, s_demo.deadline_ms)) break;
            bsp_lift_magnet_set(1);
            lift_puts("DEMO mag-on grip\r\n");
            s_demo.deadline_ms = now + LIFT_DEMO_GRIP_MS;
            s_demo.state = LIFT_DEMO_GRIP_WAIT;
            break;
        case LIFT_DEMO_GRIP_WAIT:
            if (!lift_time_reached(now, s_demo.deadline_ms)) break;
            bsp_lift_servo_home();
            lift_puts("DEMO carry-left\r\n");
            s_demo.deadline_ms = now + LIFT_DEMO_SERVO_MS;
            s_demo.state = LIFT_DEMO_CARRY_WAIT;
            break;
        case LIFT_DEMO_CARRY_WAIT:
            if (!lift_time_reached(now, s_demo.deadline_ms)) break;
            lift_puts("DEMO nav-delay\r\n");
            s_demo.deadline_ms = now + s_demo.nav_ms;
            s_demo.state = LIFT_DEMO_NAV_WAIT;
            break;
        case LIFT_DEMO_NAV_WAIT:
            if (!lift_time_reached(now, s_demo.deadline_ms)) break;
            bsp_lift_servo_right();
            lift_puts("DEMO release-right\r\n");
            s_demo.deadline_ms = now + LIFT_DEMO_SERVO_MS;
            s_demo.state = LIFT_DEMO_RELEASE_SIDE_WAIT;
            break;
        case LIFT_DEMO_RELEASE_SIDE_WAIT:
            if (!lift_time_reached(now, s_demo.deadline_ms)) break;
            lift_puts("DEMO lift-up\r\n");
            bsp_lift_move_steps(s_demo.lift_steps, s_demo.pps);
            s_demo.state = LIFT_DEMO_MOVE_UP_WAIT;
            break;
        case LIFT_DEMO_MOVE_UP_WAIT:
            if (bsp_lift_busy()) break;
            s_demo.deadline_ms = now + LIFT_SEGMENT_DWELL_MS;
            s_demo.state = LIFT_DEMO_UP_DWELL_WAIT;
            break;
        case LIFT_DEMO_UP_DWELL_WAIT:
            if (!lift_time_reached(now, s_demo.deadline_ms)) break;
            lift_puts("DEMO lift-down\r\n");
            bsp_lift_move_steps(-s_demo.lift_steps, s_demo.pps);
            s_demo.state = LIFT_DEMO_MOVE_DOWN_WAIT;
            break;
        case LIFT_DEMO_MOVE_DOWN_WAIT:
            if (bsp_lift_busy()) break;
            s_demo.deadline_ms = now + LIFT_SEGMENT_DWELL_MS;
            s_demo.state = LIFT_DEMO_DOWN_DWELL_WAIT;
            break;
        case LIFT_DEMO_DOWN_DWELL_WAIT:
            if (!lift_time_reached(now, s_demo.deadline_ms)) break;
            bsp_lift_magnet_set(0);
            lift_puts("DEMO mag-off drop\r\n");
            s_demo.deadline_ms = now + LIFT_DEMO_RELEASE_MS;
            s_demo.state = LIFT_DEMO_RELEASE_WAIT;
            break;
        case LIFT_DEMO_RELEASE_WAIT:
            if (!lift_time_reached(now, s_demo.deadline_ms)) break;
            bsp_lift_servo_home();
            s_demo.active = 0;
            s_demo.state = LIFT_DEMO_IDLE;
            lift_puts("DEMO done home-left\r\n");
            break;
        default:
            lift_auto_abort();
            lift_puts("DEMO abort bad-state\r\n");
            break;
    }
}

static void lift_help(void)
{
    lift_puts(
        "OK lift-stage mode\r\n"
        "Commands: PING STATUS HELP EN 0|1 ZERO SAFEZERO [steps] [pps] STOP "
        "JOGUP [steps] [pps] JOGDOWN [steps] [pps] "
        "STEPUP [steps] [pps] STEPDOWN [steps] [pps] "
        "UP [steps] [pps] DOWN [steps] [pps] GOTO <pos> [pps] SPEED <pps> CYCLE [steps] [pps] "
        "MAG ON|OFF ACT EXT|RET|STOP SERVO LEFT|HOME|RIGHT|US <us> "
        "AUTO [steps] [pps] [extend_ms] [retract_ms] [hold_pps] "
        "DEMO [lift_steps] [pps] [nav_ms] "
        "RAW <en_hi> <dir_hi> <pul_hi> BITPULSE <en_hi> <dir_hi> <pulses> <half_ms>\r\n"
        "Pins: EN=PD10 DIR=PD7 PUL=PC9 ACT_IN1=PC13 ACT_IN2=PC0 SERVO=PB8 MAG=PE0\r\n");
}

static void lift_process_line(char *line)
{
    char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0') return;

    if (lift_starts_with(p, "PING")) {
        lift_puts("PONG\r\n");
    } else if (lift_starts_with(p, "HELP")) {
        lift_help();
    } else if (lift_starts_with(p, "STATUS")) {
        lift_send_status();
    } else if (lift_starts_with(p, "ZERO")) {
        if (lift_reject_motion_if_busy()) return;
        lift_auto_abort();
        bsp_lift_zero_position();
        lift_puts("OK zero\r\n");
    } else if (lift_starts_with(p, "SAFEZERO")) {
        if (lift_reject_motion_if_busy()) return;
        lift_auto_abort();
        p += 8;
        long steps = lift_arg_or_default(&p, LIFT_SAFE_ZERO_MARGIN_STEPS);
        long pps = lift_arg_or_default(&p, LIFT_SAFE_ZERO_PPS);
        if (steps < 0) steps = -steps;
        if (steps > LIFT_SAFE_ZERO_MAX_STEPS) steps = LIFT_SAFE_ZERO_MAX_STEPS;
        pps = (long)lift_debug_pps(pps, LIFT_SAFE_ZERO_PPS);
        lift_puts("OK safezero move-up\r\n");
        bsp_lift_zero_position();
        bsp_lift_move_steps((int32_t)steps, (uint32_t)pps);
        s_safezero_pending = 1;
    } else if (lift_starts_with(p, "STOP")) {
        lift_auto_abort();
        lift_puts("OK stop\r\n");
    } else if (lift_starts_with(p, "EN")) {
        lift_auto_abort();
        p += 2;
        bsp_lift_enable((uint8_t)(lift_arg_or_default(&p, 1) != 0));
        lift_puts("OK en\r\n");
    } else if (lift_starts_with(p, "JOGUP")) {
        if (lift_reject_motion_if_busy()) return;
        lift_auto_abort();
        p += 5;
        long steps = lift_arg_or_default(&p, LIFT_JOG_STEPS);
        long pps = lift_arg_or_default(&p, LIFT_JOG_PPS);
        if (steps < 0) steps = -steps;
        if (steps > LIFT_SAFE_ZERO_MARGIN_STEPS) steps = LIFT_SAFE_ZERO_MARGIN_STEPS;
        pps = (long)lift_debug_pps(pps, LIFT_JOG_PPS);
        bsp_lift_move_steps((int32_t)steps, (uint32_t)pps);
        lift_puts("OK jogup\r\n");
    } else if (lift_starts_with(p, "JOGDOWN")) {
        if (lift_reject_motion_if_busy()) return;
        lift_auto_abort();
        p += 7;
        long steps = lift_arg_or_default(&p, LIFT_JOG_STEPS);
        long pps = lift_arg_or_default(&p, LIFT_JOG_PPS);
        if (steps < 0) steps = -steps;
        if (steps > LIFT_SAFE_ZERO_MARGIN_STEPS) steps = LIFT_SAFE_ZERO_MARGIN_STEPS;
        pps = (long)lift_debug_pps(pps, LIFT_JOG_PPS);
        bsp_lift_move_steps((int32_t)(-steps), (uint32_t)pps);
        lift_puts("OK jogdown\r\n");
    } else if (lift_starts_with(p, "STEPUP")) {
        if (lift_reject_motion_if_busy()) return;
        lift_auto_abort();
        p += 6;
        long steps = lift_arg_or_default(&p, LIFT_STEP_TEST_STEPS);
        long pps = lift_arg_or_default(&p, LIFT_STEP_TEST_PPS);
        if (steps < 0) steps = -steps;
        if (steps > LIFT_SAFE_ZERO_MARGIN_STEPS) steps = LIFT_SAFE_ZERO_MARGIN_STEPS;
        pps = (long)lift_debug_pps(pps, LIFT_STEP_TEST_PPS);
        bsp_lift_move_steps((int32_t)steps, (uint32_t)pps);
        lift_puts("OK stepup\r\n");
    } else if (lift_starts_with(p, "STEPDOWN")) {
        if (lift_reject_motion_if_busy()) return;
        lift_auto_abort();
        p += 8;
        long steps = lift_arg_or_default(&p, LIFT_STEP_TEST_STEPS);
        long pps = lift_arg_or_default(&p, LIFT_STEP_TEST_PPS);
        if (steps < 0) steps = -steps;
        if (steps > LIFT_SAFE_ZERO_MARGIN_STEPS) steps = LIFT_SAFE_ZERO_MARGIN_STEPS;
        pps = (long)lift_debug_pps(pps, LIFT_STEP_TEST_PPS);
        bsp_lift_move_steps((int32_t)(-steps), (uint32_t)pps);
        lift_puts("OK stepdown\r\n");
    } else if (lift_starts_with(p, "UP")) {
        if (lift_reject_motion_if_busy()) return;
        lift_auto_abort();
        p += 2;
        long steps = lift_arg_or_default(&p, LIFT_DEFAULT_UP_STEPS);
        long pps = lift_arg_or_default(&p, LIFT_DEFAULT_PPS);
        if (steps < 0) steps = -steps;
        pps = (long)lift_debug_pps(pps, LIFT_DEFAULT_PPS);
        bsp_lift_move_steps((int32_t)steps, (uint32_t)pps);
        lift_puts("OK up\r\n");
    } else if (lift_starts_with(p, "DOWN")) {
        if (lift_reject_motion_if_busy()) return;
        lift_auto_abort();
        p += 4;
        long steps = lift_arg_or_default(&p, LIFT_DEFAULT_UP_STEPS);
        long pps = lift_arg_or_default(&p, LIFT_DEFAULT_PPS);
        if (steps < 0) steps = -steps;
        pps = (long)lift_debug_pps(pps, LIFT_DEFAULT_PPS);
        bsp_lift_move_steps((int32_t)(-steps), (uint32_t)pps);
        lift_puts("OK down\r\n");
    } else if (lift_starts_with(p, "GOTO")) {
        if (lift_reject_motion_if_busy()) return;
        lift_auto_abort();
        p += 4;
        long target = lift_arg_or_default(&p, 0);
        long pps = lift_arg_or_default(&p, LIFT_DEFAULT_PPS);
        pps = (long)lift_debug_pps(pps, LIFT_DEFAULT_PPS);
        bsp_lift_move_to((int32_t)target, (uint32_t)pps);
        lift_puts("OK goto\r\n");
    } else if (lift_starts_with(p, "SPEED")) {
        p += 5;
        long pps = lift_arg_or_default(&p, 0);
        if (pps == 0) {
            lift_auto_abort();
            lift_puts("OK speed stop\r\n");
            return;
        }
        (void)pps;
        lift_puts("ERR speed disabled in gpio-safe mode; use STEPUP/STEPDOWN\r\n");
    } else if (lift_starts_with(p, "CYCLE")) {
        if (lift_reject_motion_if_busy()) return;
        lift_auto_abort();
        p += 5;
        long steps = lift_arg_or_default(&p, LIFT_DEFAULT_UP_STEPS);
        long pps = lift_arg_or_default(&p, LIFT_DEFAULT_PPS);
        if (steps < 0) steps = -steps;
        pps = (long)lift_debug_pps(pps, LIFT_DEFAULT_PPS);
        bsp_lift_enable(1);
        lift_puts("OK cycle grip-delay\r\n");
        s_cycle.active = 1;
        s_cycle.deadline_ms = millis() + LIFT_PICK_DELAY_MS;
        s_cycle.steps = (int32_t)steps;
        s_cycle.pps = (uint32_t)pps;
    } else if (lift_starts_with(p, "MAG")) {
        if (lift_reject_motion_if_busy()) return;
        lift_auto_abort();
        p = lift_skip_ws(p + 3);
        if (lift_starts_with(p, "ON") || lift_starts_with(p, "1")) {
            bsp_lift_magnet_set(1);
            lift_puts("OK mag on\r\n");
        } else if (lift_starts_with(p, "OFF") || lift_starts_with(p, "0")) {
            bsp_lift_magnet_set(0);
            lift_puts("OK mag off\r\n");
        } else {
            lift_puts("ERR mag expects ON|OFF\r\n");
        }
    } else if (lift_starts_with(p, "ACT")) {
        if (lift_reject_motion_if_busy()) return;
        lift_auto_abort();
        p = lift_skip_ws(p + 3);
        if (lift_starts_with(p, "EXT")) {
            bsp_lift_actuator_extend();
            lift_puts("OK act extend\r\n");
        } else if (lift_starts_with(p, "RET")) {
            bsp_lift_actuator_retract();
            lift_puts("OK act retract\r\n");
        } else if (lift_starts_with(p, "STOP")) {
            bsp_lift_actuator_stop();
            lift_puts("OK act stop\r\n");
        } else {
            lift_puts("ERR act expects EXT|RET|STOP\r\n");
        }
    } else if (lift_starts_with(p, "SERVO")) {
        if (lift_reject_motion_if_busy()) return;
        lift_auto_abort();
        p = lift_skip_ws(p + 5);
        if (lift_starts_with(p, "LEFT")) {
            bsp_lift_servo_left();
            lift_puts("OK servo left\r\n");
        } else if (lift_starts_with(p, "HOME")) {
            bsp_lift_servo_home();
            lift_puts("OK servo home\r\n");
        } else if (lift_starts_with(p, "RIGHT")) {
            bsp_lift_servo_right();
            lift_puts("OK servo right\r\n");
        } else if (lift_starts_with(p, "US")) {
            p += 2;
            long us = lift_arg_or_default(&p, LIFT_SERVO_HOME_US);
            bsp_lift_servo_us((uint16_t)us);
            lift_puts("OK servo us\r\n");
        } else {
            lift_puts("ERR servo expects LEFT|HOME|RIGHT|US <us>\r\n");
        }
    } else if (lift_starts_with(p, "AUTO")) {
        if (lift_reject_motion_if_busy()) return;
        p += 4;
        long steps = lift_arg_or_default(&p, LIFT_AUTO_TRAVEL_STEPS);
        long pps = lift_arg_or_default(&p, LIFT_AUTO_SPEED_PPS);
        long extend_ms = lift_arg_or_default(&p, LIFT_AUTO_EXTEND_MS);
        long retract_ms = lift_arg_or_default(&p, LIFT_AUTO_RETRACT_MS);
        long hold_pps = lift_arg_or_default(&p, LIFT_AUTO_TOP_HOLD_PPS);
        lift_auto_start(steps, pps, extend_ms, retract_ms, hold_pps);
        lift_puts("OK auto start\r\n");
    } else if (lift_starts_with(p, "DEMO")) {
        if (lift_reject_motion_if_busy()) return;
        p += 4;
        long steps = lift_arg_or_default(&p, LIFT_DEMO_LIFT_STEPS);
        long pps = lift_arg_or_default(&p, LIFT_DEMO_SPEED_PPS);
        long nav_ms = lift_arg_or_default(&p, LIFT_DEMO_NAV_DELAY_MS);
        lift_puts("OK demo start\r\n");
        lift_demo_start(steps, pps, nav_ms);
    } else if (lift_starts_with(p, "RAW")) {
        if (lift_reject_motion_if_busy()) return;
        lift_auto_abort();
        p += 3;
        long en_hi = lift_arg_or_default(&p, 0);
        long dir_hi = lift_arg_or_default(&p, 0);
        long pul_hi = lift_arg_or_default(&p, 0);
        bsp_lift_raw_pins((uint8_t)(en_hi != 0),
                          (uint8_t)(dir_hi != 0),
                          (uint8_t)(pul_hi != 0));
        lift_puts("OK raw\r\n");
    } else if (lift_starts_with(p, "BITPULSE")) {
        if (lift_reject_motion_if_busy()) return;
        lift_auto_abort();
        p += 8;
        long en_hi = lift_arg_or_default(&p, 0);
        long dir_hi = lift_arg_or_default(&p, 1);
        long pulses = lift_arg_or_default(&p, 200);
        long half_ms = lift_arg_or_default(&p, 5);
        if (pulses < 0) pulses = -pulses;
        if (half_ms < 1) half_ms = 1;
        bsp_lift_bitbang_pulses((uint8_t)(en_hi != 0),
                                (uint8_t)(dir_hi != 0),
                                (uint32_t)pulses,
                                (uint32_t)half_ms);
        lift_puts("OK bitpulse started\r\n");
    } else {
        lift_puts("ERR unknown; send HELP\r\n");
    }
}

int main(void)
{
    bsp_clock_init();
    bsp_systick_init();
    bsp_led_init();
    bsp_uart_init();
    bsp_lift_init();

    bsp_led_set_mode(LED_MODE_SLOW_BLINK);
    lift_help();
    lift_send_status();

    char line[128];
    uint16_t line_len = 0;
    uint32_t t_prev_status = millis();

    for (;;) {
        uint8_t buf[64];
        uint16_t n = host_uart_read(buf, sizeof(buf));
        for (uint16_t i = 0; i < n; ++i) {
            char c = (char)buf[i];
            if (c == '\r' || c == '\n') {
                line[line_len] = '\0';
                lift_process_line(line);
                line_len = 0;
            } else if (line_len < (sizeof(line) - 1U)) {
                line[line_len++] = c;
            } else {
                line_len = 0;
                lift_puts("ERR line-too-long\r\n");
            }
        }

        {
            uint32_t now = millis();
            bsp_lift_service(now);
            lift_deferred_tick(now);
            lift_auto_tick();
            lift_demo_tick(now);
        }

        if ((millis() - t_prev_status) >= 500U) {
            t_prev_status = millis();
            if (bsp_lift_busy() || s_auto.active) {
                bsp_led_set_mode(LED_MODE_FAST_BLINK);
                lift_send_status();
            } else {
                bsp_led_set_mode(bsp_lift_enabled() ? LED_MODE_ON : LED_MODE_SLOW_BLINK);
            }
        }

        delay_ms(1);
        bsp_led_tick_1ms();
    }
}

#elif TEST_MODE == 4

/* ============================================================
 * mode=4: standalone lift stepper spin test.
 *
 * This mode does not require the car X5. After flashing, the F407 directly
 * drives the lift-stage 42 stepper pins:
 *   EN=PD10, DIR=PD7, PUL=PC9.
 *
 * Keep one hand on power. There are still no limit switches.
 * ============================================================ */
int main(void)
{
    bsp_clock_init();
    bsp_systick_init();
    bsp_led_init();
    bsp_lift_init();

    bsp_led_set_mode(LED_MODE_FAST_BLINK);
    delay_ms(500);

    bsp_lift_enable(1);
    delay_ms(100);

    for (int32_t pps = LIFT_SPIN_RAMP_START_PPS;
         pps < LIFT_SPIN_TARGET_PPS;
         pps += LIFT_SPIN_RAMP_STEP_PPS) {
        bsp_lift_set_speed_pps(pps);
        delay_ms(LIFT_SPIN_RAMP_HOLD_MS);
    }
    bsp_lift_set_speed_pps(LIFT_SPIN_TARGET_PPS);

    for (;;) {
        delay_ms(1);
        bsp_led_tick_1ms();
    }
}

#elif TEST_MODE == 5

/* ============================================================
 * mode=5: first-round boot demo fallback.
 *
 * This mode bypasses the X5 serial command path. After flashing and power-on,
 * the F407 directly runs the first-round fixture actions, now upgraded with
 * the validated X42S UART PREP recovery:
 *   left/start -> right/work -> extend once -> magnet on
 *   lift to top + lock -> left/start -> stationary 10s -> right/work
 *   controlled lift return -> magnet off/drop -> left/start
 *   -> final actuator full retract (double retract time)
 *
	 * Use this for the first-round video fallback. Return TEST_MODE to 0 for
 * the normal X5 chassis firmware, or to 3 for lift-stage UART debugging.
 * ============================================================ */
static void zdt_send(const uint8_t *frame, uint16_t len)
{
    bsp_lift_uart5_send(frame, len);
    delay_ms(1);
}

static void zdt_enable(uint8_t on)
{
    uint8_t frame[6] = {
        LIFT_ZDT_ADDR, 0xF3U, 0xABU, on ? 1U : 0U, 0x00U, 0x6BU
    };
    zdt_send(frame, sizeof(frame));
}

static void zdt_clear_stall(void)
{
    uint8_t frame[4] = {LIFT_ZDT_ADDR, 0x0EU, 0x52U, 0x6BU};
    zdt_send(frame, sizeof(frame));
}

static void zdt_stop(void)
{
    uint8_t frame[5] = {LIFT_ZDT_ADDR, 0xFEU, 0x98U, 0x00U, 0x6BU};
    zdt_send(frame, sizeof(frame));
}

static void zdt_reset_position(void)
{
    uint8_t frame[4] = {LIFT_ZDT_ADDR, 0x0AU, 0x6DU, 0x6BU};
    zdt_send(frame, sizeof(frame));
}

static void zdt_move(uint8_t dir, uint32_t pulses, uint8_t mode)
{
    uint16_t rpm = LIFT_ZDT_UP_RPM;
    uint8_t frame[13] = {
        LIFT_ZDT_ADDR,
        0xFDU,
        dir,
        (uint8_t)(rpm >> 8),
        (uint8_t)(rpm & 0xFFU),
        LIFT_ZDT_UP_ACC,
        (uint8_t)(pulses >> 24),
        (uint8_t)(pulses >> 16),
        (uint8_t)(pulses >> 8),
        (uint8_t)(pulses & 0xFFU),
        mode,
        0x00U,
        0x6BU
    };
    zdt_send(frame, sizeof(frame));
}

static void zdt_lock_current_position(void)
{
    zdt_stop();
    delay_ms(20);
    zdt_clear_stall();
    delay_ms(20);
    zdt_reset_position();
    delay_ms(20);
    zdt_enable(1);
}

static void lift_demo_servo_left_loaded(void)
{
    bsp_lift_servo_us(1550U);
    delay_ms(900);
    bsp_lift_servo_us(1700U);
    delay_ms(900);
    bsp_lift_servo_us(1900U);
    delay_ms(900);
    bsp_lift_servo_us(2100U);
    delay_ms(900);
    bsp_lift_servo_us(LIFT_SERVO_LEFT_US);
    delay_ms(1500);
}

static void lift_demo_servo_right_loaded(void)
{
    bsp_lift_servo_us(2100U);
    delay_ms(900);
    bsp_lift_servo_us(1900U);
    delay_ms(900);
    bsp_lift_servo_us(1700U);
    delay_ms(900);
    bsp_lift_servo_us(1550U);
    delay_ms(900);
    bsp_lift_servo_us(LIFT_SERVO_RIGHT_US);
    delay_ms(LIFT_DEMO_RIGHT_SETTLE_MS);
}

int main(void)
{
    uint32_t start_ms;
    uint32_t last_clear_ms;

    bsp_clock_init();
    bsp_systick_init();
    bsp_led_init();
    bsp_motor_init();
    bsp_lift_init();

    bsp_led_set_mode(LED_MODE_FAST_BLINK);
    delay_ms(1500);

    bsp_motor_disable_all();
    bsp_lift_enable(1);
    bsp_lift_actuator_stop();
    bsp_lift_magnet_set(0);

    /* The operator places the mechanism at the retracted left/start pose. */
    bsp_lift_servo_us(LIFT_SERVO_LEFT_US);
    delay_ms(3000);

    /* Move to the right/work pose and extend to the repeatable pickup length. */
    bsp_lift_servo_us(LIFT_SERVO_RIGHT_US);
    delay_ms(2500);

    bsp_lift_actuator_extend();
    delay_ms(LIFT_DEMO_ACT_EXTEND_MS);
    bsp_lift_actuator_stop();
    delay_ms(500);

    bsp_lift_magnet_set(1);
    delay_ms(2500);

    /* Validated PREP: rise, clear the brief top stall, cancel overtravel. */
    bsp_lift_stop();
    bsp_lift_enable(1);
    zdt_clear_stall();
    delay_ms(50);
    zdt_enable(1);
    delay_ms(100);
    zdt_move(LIFT_ZDT_UP_DIR, LIFT_ZDT_UP_PULSES, 0x00U);

    start_ms = millis();
    last_clear_ms = start_ms;
    while ((uint32_t)(millis() - start_ms) < LIFT_ZDT_PREP_DURATION_MS) {
        uint32_t now = millis();
        if ((uint32_t)(now - last_clear_ms) >= LIFT_ZDT_CLEAR_PERIOD_MS) {
            zdt_clear_stall();
            last_clear_ms = now;
        }
        delay_ms(1);
        bsp_led_tick_1ms();
    }
    zdt_lock_current_position();

    /* Carry left at the top, then keep the chassis disabled for 10 seconds. */
    lift_demo_servo_left_loaded();
    bsp_led_set_mode(LED_MODE_SLOW_BLINK);
    bsp_motor_disable_all();
    delay_ms(LIFT_DEMO_TOP_WAIT_MS);

    /* Return the still-extended actuator to the right/work pose. */
    lift_demo_servo_right_loaded();

    /* Controlled return to the calibrated soft bottom, then lock the axis. */
    bsp_led_set_mode(LED_MODE_FAST_BLINK);
    zdt_move(LIFT_ZDT_DOWN_DIR, LIFT_ZDT_DOWN_PULSES, 0x02U);
    delay_ms(LIFT_ZDT_DOWN_WAIT_MS);
    zdt_lock_current_position();

    /* Release at the bottom without changing the calibrated extension. */
    bsp_lift_magnet_set(0);
    delay_ms(LIFT_DEMO_PLACE_MS);

    /* Return left/start first, then retract fully only once at the very end. */
    lift_demo_servo_left_loaded();
    bsp_lift_actuator_retract();
    delay_ms(LIFT_DEMO_FINAL_RETRACT_MS);
    bsp_lift_actuator_stop();
    bsp_led_set_mode(LED_MODE_ON);

    for (;;) {
        delay_ms(1);
        bsp_led_tick_1ms();
    }
}

#elif TEST_MODE == 6

/* ============================================================
 * mode=6: ZDT X42S UART PREP top-hold test.
 *
 * This reproduces the supplied, validated PREP sequence on the existing
 * UART5 wiring (PC12 TX / PD2 RX). The X42S default stall policy disables
 * the drive and releases the shaft; repeatedly clearing protection during
 * the hard-stop interval, then explicitly enabling again, keeps the axis
 * locked instead of allowing the lift to fall.
 *
 * There is no limit switch. The return uses a calibrated 10000-pulse stroke
 * so it stops short of the hard bottom instead of using another collision.
 * ============================================================ */
static void zdt_send(const uint8_t *frame, uint16_t len)
{
    bsp_lift_uart5_send(frame, len);
    delay_ms(1);
}

static void zdt_enable(uint8_t on)
{
    uint8_t frame[6] = {
        LIFT_ZDT_ADDR, 0xF3U, 0xABU, on ? 1U : 0U, 0x00U, 0x6BU
    };
    zdt_send(frame, sizeof(frame));
}

static void zdt_clear_stall(void)
{
    uint8_t frame[4] = {LIFT_ZDT_ADDR, 0x0EU, 0x52U, 0x6BU};
    zdt_send(frame, sizeof(frame));
}

static void zdt_stop(void)
{
    uint8_t frame[5] = {LIFT_ZDT_ADDR, 0xFEU, 0x98U, 0x00U, 0x6BU};
    zdt_send(frame, sizeof(frame));
}

static void zdt_reset_position(void)
{
    uint8_t frame[4] = {LIFT_ZDT_ADDR, 0x0AU, 0x6DU, 0x6BU};
    zdt_send(frame, sizeof(frame));
}

static void zdt_move_up_prep(void)
{
    uint32_t pulses = LIFT_ZDT_UP_PULSES;
    uint16_t rpm = LIFT_ZDT_UP_RPM;
    uint8_t frame[13] = {
        LIFT_ZDT_ADDR,
        0xFDU,
        LIFT_ZDT_UP_DIR,
        (uint8_t)(rpm >> 8),
        (uint8_t)(rpm & 0xFFU),
        LIFT_ZDT_UP_ACC,
        (uint8_t)(pulses >> 24),
        (uint8_t)(pulses >> 16),
        (uint8_t)(pulses >> 8),
        (uint8_t)(pulses & 0xFFU),
        0x00U, /* relative to the previous input target, matching PREP */
        0x00U, /* execute immediately */
        0x6BU
    };
    zdt_send(frame, sizeof(frame));
}

static void zdt_move_down_safe(void)
{
    uint32_t pulses = LIFT_ZDT_DOWN_PULSES;
    uint16_t rpm = LIFT_ZDT_UP_RPM;
    uint8_t frame[13] = {
        LIFT_ZDT_ADDR,
        0xFDU,
        LIFT_ZDT_DOWN_DIR,
        (uint8_t)(rpm >> 8),
        (uint8_t)(rpm & 0xFFU),
        LIFT_ZDT_UP_ACC,
        (uint8_t)(pulses >> 24),
        (uint8_t)(pulses >> 16),
        (uint8_t)(pulses >> 8),
        (uint8_t)(pulses & 0xFFU),
        0x02U, /* relative to current real position */
        0x00U, /* execute immediately */
        0x6BU
    };
    zdt_send(frame, sizeof(frame));
}

int main(void)
{
    uint32_t start_ms;
    uint32_t last_clear_ms;

    bsp_clock_init();
    bsp_systick_init();
    bsp_led_init();
    bsp_lift_init();

    bsp_led_set_mode(LED_MODE_FAST_BLINK);
    delay_ms(1500);

    /* Keep the legacy EN input active while UART mode owns motion. */
    bsp_lift_stop();
    bsp_lift_enable(1);
    zdt_clear_stall();
    delay_ms(50);
    zdt_enable(1);
    delay_ms(100);

    zdt_move_up_prep();
    start_ms = millis();
    last_clear_ms = start_ms;

    while ((uint32_t)(millis() - start_ms) < LIFT_ZDT_PREP_DURATION_MS) {
        uint32_t now = millis();
        if ((uint32_t)(now - last_clear_ms) >= LIFT_ZDT_CLEAR_PERIOD_MS) {
            zdt_clear_stall();
            last_clear_ms = now;
        }
        delay_ms(1);
        bsp_led_tick_1ms();
    }

    /*
     * Cancel the unreachable overtravel target before the final recovery.
     * Otherwise the closed loop resumes pushing into the hard stop after
     * CLEAR, trips protection again, releases the shaft, and drops the lift.
     */
    zdt_stop();
    delay_ms(20);
    zdt_clear_stall();
    delay_ms(20);
    zdt_reset_position();
    delay_ms(20);
    zdt_enable(1);

    /* Keep the successful top hold visible before the calibrated return. */
    bsp_led_set_mode(LED_MODE_SLOW_BLINK);
    delay_ms(LIFT_ZDT_TOP_HOLD_MS);

    bsp_led_set_mode(LED_MODE_FAST_BLINK);
    zdt_move_down_safe();
    delay_ms(LIFT_ZDT_DOWN_WAIT_MS);

    /* Finish at the soft bottom with no residual position target. */
    zdt_stop();
    delay_ms(20);
    zdt_clear_stall();
    delay_ms(20);
    zdt_reset_position();
    delay_ms(20);
    zdt_enable(1);
    bsp_led_set_mode(LED_MODE_ON);

    for (;;) {
        delay_ms(1);
        bsp_led_tick_1ms();
    }
}

#else

/* ============================================================
 * mode=0: 正常模式 (0xAA55 协议 + IMU + 50Hz odom 上行)
 * ============================================================ */
static uint8_t s_emergency_active = 0;

static void apply_auxiliary_safety_stop(void)
{
    if (bsp_lift_busy()) bsp_lift_stop();
    if (bsp_lift_actuator_state() != LIFT_ACT_STOPPED) bsp_lift_actuator_stop();
    if (bsp_lift_magnet_on()) bsp_lift_magnet_set(0);
    bsp_lift_servo_hold();
}

static void apply_target_to_motors(float v, float w)
{
    int32_t pps[4];
    odom_cmdvel_to_wheels(v, w, pps);
    for (int i = 0; i < 4; ++i) {
        bsp_motor_set_speed_pps((uint8_t)i, pps[i]);
    }
}

int main(void)
{
    bsp_clock_init();
    bsp_systick_init();
    bsp_led_init();
    bsp_uart_init();
    bsp_motor_init();
    bsp_imu_init();
    bsp_lift_init();
    proto_init();

    bsp_motor_enable_all();          /* 上电使能 4 路 X57S */
    bsp_led_set_mode(LED_MODE_SLOW_BLINK);
    proto_send_error(0x00, "stm32f407 nav boot");
    proto_send_firmware_info(PROTO_PROTOCOL_VERSION, PROTO_CAPABILITIES,
                             PROTO_FIRMWARE_BUILD_ID, (uint8_t)TEST_MODE,
                             PROTO_HW_VARIANT);

    uint32_t t_prev_loop      = millis();
    uint32_t t_prev_odom_pub  = t_prev_loop;
    uint32_t t_prev_ext_pub   = t_prev_loop;
    uint32_t t_prev_odom_calc = t_prev_loop;

    for (;;) {
        uint32_t now = millis();

        if ((now - t_prev_loop) >= LOOP_DT_MS) {
            t_prev_loop = now;

            uint8_t buf[128];
            uint16_t n = host_uart_read(buf, sizeof(buf));
            if (n > 0) proto_feed_rx(buf, n, now);

            bsp_imu_poll(now);
            bsp_led_tick_1ms();
            bsp_led_tick_1ms();
        }

        ProtoState *ps = proto_state();

        if (ps->emergency_stop_request) {
            ps->emergency_stop_request = 0;
            s_emergency_active = 1;
        }

        if (ps->estop_latched) {
            s_emergency_active = 1;
        }

        if (now > BOOT_GRACE_MS) {
            if (ps->estop_latched || ps->last_heartbeat_ms == 0 ||
                (now - ps->last_heartbeat_ms) > HEARTBEAT_TIMEOUT_MS) {
                s_emergency_active = 1;
                apply_auxiliary_safety_stop();
            } else if (s_emergency_active) {
                if (ps->target_linear_v == 0.0f && ps->target_angular_w == 0.0f) {
                    s_emergency_active = 0;
                    bsp_motor_enable_all();
                }
            }
        }

        /* Also aborts any protocol fixture state when estop/link freshness fails. */
        proto_service(now);
        if (!s_emergency_active) bsp_lift_service(now);

        float v_target = ps->target_linear_v;
        float w_target = ps->target_angular_w;
        if (now > BOOT_GRACE_MS &&
            (ps->last_cmd_vel_ms == 0 || (now - ps->last_cmd_vel_ms) > CMD_VEL_FRESH_MS)) {
            v_target = 0.0f;
            w_target = 0.0f;
        }

        if (s_emergency_active) {
            apply_auxiliary_safety_stop();
            bsp_motor_emergency_stop();
            odom_zero_velocity();
            bsp_led_set_mode(LED_MODE_ON);
        } else {
            apply_target_to_motors(v_target, w_target);
            bsp_led_set_mode(
                (ps->last_heartbeat_ms != 0 && (now - ps->last_heartbeat_ms) < 500)
                    ? LED_MODE_FAST_BLINK : LED_MODE_SLOW_BLINK);
        }

        if ((now - t_prev_odom_calc) >= (1000U / ODOM_PUBLISH_HZ)) {
            float dt = (float)(now - t_prev_odom_calc) / 1000.0f;
            t_prev_odom_calc = now;

            int32_t d1 = bsp_motor_take_step_delta(0);
            int32_t d2 = bsp_motor_take_step_delta(1);
            int32_t d3 = bsp_motor_take_step_delta(2);
            int32_t d4 = bsp_motor_take_step_delta(3);

            const ImuData *imu = bsp_imu_data();
            uint8_t imu_valid = (imu->last_angle_ms != 0) && ((now - imu->last_angle_ms) < 500);
            float yaw_rad = imu->yaw_deg * 0.01745329252f;

            odom_update(d1, d2, d3, d4, dt, 1, imu_valid, yaw_rad);
        }

        if ((now - t_prev_odom_pub) >= (1000U / ODOM_PUBLISH_HZ)) {
            t_prev_odom_pub = now;
            float x, y, vx, wz, yaw_deg;
            odom_get(&x, &y, &vx, &wz, &yaw_deg);
            proto_send_basic_odom(x, y, vx, wz, yaw_deg);
        }

        if ((now - t_prev_ext_pub) >= (1000U / EXT_TELEMETRY_HZ)) {
            float fixture_height_m = proto_fixture_height_m();
            float lift_height_m = fixture_height_m >= 0.0f
                ? fixture_height_m
                : ((float)bsp_lift_position_steps()) / 25000.0f;
            float lift_velocity_mps = (proto_fixture_busy() || bsp_lift_busy())
                ? 0.01f : 0.0f;
            t_prev_ext_pub = now;
            const ImuData *imu = bsp_imu_data();
            proto_send_ext_telemetry(
                lift_height_m,
                lift_velocity_mps,
                0, 0, bsp_lift_magnet_on(), 0, /* no physical limit/home feedback installed */
                imu->accel_x, imu->accel_y, imu->accel_z,
                imu->gyro_x,  imu->gyro_y,  imu->gyro_z,
                25.0f, 12.0f);
            proto_send_safety_state(
                ps->estop_latched,
                s_emergency_active,
                ps->estop_blocked_command_count);
            proto_send_firmware_info(PROTO_PROTOCOL_VERSION, PROTO_CAPABILITIES,
                                     PROTO_FIRMWARE_BUILD_ID, (uint8_t)TEST_MODE,
                                     PROTO_HW_VARIANT);
        }
    }
}

#endif  /* TEST_MODE */
