#include "proto.h"
#include "bsp_uart.h"
#include "bsp_lift.h"
#include <string.h>

/* ============================================================
 * 解析状态机 + payload 处理
 * ============================================================ */

typedef enum {
    ST_WAIT_H0 = 0,
    ST_WAIT_H1,
    ST_WAIT_TYPE,
    ST_WAIT_LEN,
    ST_WAIT_PAYLOAD,
    ST_WAIT_CKS
} ParseState;

static ParseState s_st        = ST_WAIT_H0;
static uint8_t    s_type      = 0;
static uint8_t    s_len       = 0;
static uint8_t    s_payload[PROTO_FRAME_MAX];
static uint8_t    s_idx       = 0;
static uint8_t    s_sum_accum = 0;        /* 累加 hdr0..payload 末 */

static ProtoState s_state;

typedef enum {
    VIDEO_FIXTURE_IDLE = 0,
    VIDEO_PICK_LEFT_WAIT,
    VIDEO_PICK_RIGHT_CONFIRM_WAIT,
    VIDEO_PICK_RIGHT_FINAL_WAIT,
    VIDEO_PICK_EXTEND_WAIT,
    VIDEO_PICK_EXTEND_SETTLE_WAIT,
    VIDEO_PICK_GRIP_WAIT,
    VIDEO_PICK_PREP_ENABLE_WAIT,
    VIDEO_PICK_PREP_START_WAIT,
    VIDEO_PICK_PREP_ACTIVE,
    VIDEO_PICK_LOCK_CLEAR_WAIT,
    VIDEO_PICK_LOCK_RESET_WAIT,
    VIDEO_PICK_LOCK_ENABLE_WAIT,
    VIDEO_PICK_LEFT_1550_WAIT,
    VIDEO_PICK_LEFT_1700_WAIT,
    VIDEO_PICK_LEFT_1900_WAIT,
    VIDEO_PICK_LEFT_2100_WAIT,
    VIDEO_PICK_LEFT_FINAL_WAIT,
    VIDEO_PLACE_SERVO_2100_WAIT,
    VIDEO_PLACE_SERVO_1900_WAIT,
    VIDEO_PLACE_SERVO_1700_WAIT,
    VIDEO_PLACE_SERVO_1550_WAIT,
    VIDEO_PLACE_RIGHT_WAIT,
    VIDEO_PLACE_DOWN_WAIT,
    VIDEO_PLACE_LOCK_CLEAR_WAIT,
    VIDEO_PLACE_LOCK_RESET_WAIT,
    VIDEO_PLACE_LOCK_ENABLE_WAIT,
    VIDEO_PLACE_RELEASE_WAIT,
    VIDEO_PLACE_LEFT_FINAL_WAIT,
    VIDEO_PLACE_RETRACT_WAIT,
    VIDEO_HOME_RETRACT_WAIT
} VideoFixtureState;

typedef struct {
    VideoFixtureState state;
    uint32_t deadline_ms;
    uint32_t last_clear_ms;
    float height_m;
    uint8_t tracking;
} VideoFixtureCtx;

static VideoFixtureCtx s_video_fixture = {
    VIDEO_FIXTURE_IDLE, 0u, 0u, 0.0f, 0u
};

typedef struct {
    uint32_t deadline_ms;
    uint8_t active;
    uint8_t move_right;
    uint8_t index;
} VideoServoRampCtx;

static VideoServoRampCtx s_video_servo_ramp = {0u, 0u, 0u, 0u};

#define VIDEO_FIXTURE_PICK_COMMAND       (-1.0f)
#define VIDEO_FIXTURE_PLACE_COMMAND      (-2.0f)
#define VIDEO_SERVO_RIGHT_COMMAND        (-3.0f)
#define VIDEO_SERVO_LEFT_COMMAND         (-4.0f)
#define VIDEO_FIXTURE_COMMAND_TOLERANCE  0.01f
#define VIDEO_FIXTURE_TOP_HEIGHT_M       0.40f
#define VIDEO_ZDT_ADDR                   0u
#define VIDEO_ZDT_UP_DIR                 1u
#define VIDEO_ZDT_DOWN_DIR               0u
#define VIDEO_ZDT_RPM                    150u
#define VIDEO_ZDT_ACC                    50u
#define VIDEO_ZDT_UP_PULSES              15000ul
#define VIDEO_ZDT_DOWN_PULSES            10000ul
#define VIDEO_ZDT_PREP_MS                2800u
#define VIDEO_ZDT_CLEAR_PERIOD_MS        10u
#define VIDEO_ZDT_DOWN_WAIT_MS           4500u
#define VIDEO_FINAL_RETRACT_MS           (2u * LIFT_DEMO_ACT_RETRACT_MS)
#define VIDEO_SERVO_RAMP_STEP_MS         500u
#define VIDEO_EMPTY_SERVO_SETTLE_MS      3000u
#define VIDEO_TOP_LEFT_READY_MS           250u

static uint8_t command_link_fresh(uint32_t now_ms)
{
    return (s_state.last_heartbeat_ms != 0u) &&
           ((uint32_t)(now_ms - s_state.last_heartbeat_ms) <= PROTO_COMMAND_LINK_TIMEOUT_MS);
}

static uint8_t motion_interlock_status(uint32_t now_ms)
{
    if (s_state.estop_latched) return PROTO_ACK_ESTOP_LATCHED;
    if (!command_link_fresh(now_ms)) return PROTO_ACK_LINK_STALE;
    return PROTO_ACK_OK;
}

void proto_init(void)
{
    memset(&s_state, 0, sizeof(s_state));
    /* A reset must never silently clear the firmware safety latch. */
    s_state.estop_latched = 1u;
    s_state.emergency_stop_request = 1u;
    s_st = ST_WAIT_H0;
    s_video_fixture.state = VIDEO_FIXTURE_IDLE;
    s_video_fixture.deadline_ms = 0u;
    s_video_fixture.last_clear_ms = 0u;
    s_video_fixture.height_m = 0.0f;
    s_video_fixture.tracking = 0u;
    s_video_servo_ramp.deadline_ms = 0u;
    s_video_servo_ramp.active = 0u;
    s_video_servo_ramp.move_right = 0u;
    s_video_servo_ramp.index = 0u;
}

ProtoState* proto_state(void) { return &s_state; }

uint8_t proto_fixture_busy(void)
{
    return (s_video_fixture.state != VIDEO_FIXTURE_IDLE) || s_video_servo_ramp.active;
}

float proto_fixture_height_m(void)
{
    return s_video_fixture.tracking ? s_video_fixture.height_m : -1.0f;
}

#define LIFT_VIDEO_STEPS_PER_M      25000.0f
#define LIFT_VIDEO_MAX_TARGET_STEPS 5000L

static void video_fixture_stop_motion_target(void)
{
    s_state.target_linear_v = 0.0f;
    s_state.target_angular_w = 0.0f;
}

static void video_zdt_send(const uint8_t *frame, uint16_t len)
{
    bsp_lift_uart5_send(frame, len);
}

static void video_zdt_enable(uint8_t on)
{
    uint8_t frame[6] = {
        VIDEO_ZDT_ADDR, 0xF3u, 0xABu, on ? 1u : 0u, 0x00u, 0x6Bu
    };
    video_zdt_send(frame, sizeof(frame));
}

static void video_zdt_clear_stall(void)
{
    uint8_t frame[4] = {VIDEO_ZDT_ADDR, 0x0Eu, 0x52u, 0x6Bu};
    video_zdt_send(frame, sizeof(frame));
}

static void video_zdt_stop(void)
{
    uint8_t frame[5] = {VIDEO_ZDT_ADDR, 0xFEu, 0x98u, 0x00u, 0x6Bu};
    video_zdt_send(frame, sizeof(frame));
}

static void video_zdt_reset_position(void)
{
    uint8_t frame[4] = {VIDEO_ZDT_ADDR, 0x0Au, 0x6Du, 0x6Bu};
    video_zdt_send(frame, sizeof(frame));
}

static void video_zdt_move(uint8_t dir, uint32_t pulses, uint8_t mode)
{
    uint8_t frame[13] = {
        VIDEO_ZDT_ADDR,
        0xFDu,
        dir,
        (uint8_t)(VIDEO_ZDT_RPM >> 8),
        (uint8_t)(VIDEO_ZDT_RPM & 0xFFu),
        VIDEO_ZDT_ACC,
        (uint8_t)(pulses >> 24),
        (uint8_t)(pulses >> 16),
        (uint8_t)(pulses >> 8),
        (uint8_t)(pulses & 0xFFu),
        mode,
        0x00u,
        0x6Bu
    };
    video_zdt_send(frame, sizeof(frame));
}

static uint8_t video_fixture_command_matches(float target, float command)
{
    float delta = target - command;
    if (delta < 0.0f) delta = -delta;
    return delta <= VIDEO_FIXTURE_COMMAND_TOLERANCE;
}

static uint8_t video_fixture_time_reached(uint32_t now_ms, uint32_t deadline_ms)
{
    return ((int32_t)(now_ms - deadline_ms) >= 0) ? 1u : 0u;
}

static void video_servo_ramp_start(uint8_t move_right, uint32_t now_ms)
{
    static const uint16_t right_pulses[] = {2100u, 1900u, 1700u, 1550u, LIFT_SERVO_RIGHT_US};
    static const uint16_t left_pulses[] = {1550u, 1700u, 1900u, 2100u, LIFT_SERVO_LEFT_US};
    const uint16_t *pulses = move_right ? right_pulses : left_pulses;

    s_video_servo_ramp.active = 1u;
    s_video_servo_ramp.move_right = move_right ? 1u : 0u;
    s_video_servo_ramp.index = 0u;
    bsp_lift_servo_us(pulses[0]);
    s_video_servo_ramp.deadline_ms = now_ms + VIDEO_SERVO_RAMP_STEP_MS;
}

static void video_servo_ramp_service(uint32_t now_ms)
{
    static const uint16_t right_pulses[] = {2100u, 1900u, 1700u, 1550u, LIFT_SERVO_RIGHT_US};
    static const uint16_t left_pulses[] = {1550u, 1700u, 1900u, 2100u, LIFT_SERVO_LEFT_US};
    const uint16_t *pulses;

    if (!s_video_servo_ramp.active ||
        !video_fixture_time_reached(now_ms, s_video_servo_ramp.deadline_ms)) return;

    s_video_servo_ramp.index++;
    if (s_video_servo_ramp.index >= 5u) {
        s_video_servo_ramp.active = 0u;
        return;
    }
    pulses = s_video_servo_ramp.move_right ? right_pulses : left_pulses;
    bsp_lift_servo_us(pulses[s_video_servo_ramp.index]);
    s_video_servo_ramp.deadline_ms = now_ms + VIDEO_SERVO_RAMP_STEP_MS;
}

static void video_fixture_abort(uint8_t magnet_off)
{
    s_video_servo_ramp.active = 0u;
    s_video_servo_ramp.index = 0u;
    s_video_fixture.state = VIDEO_FIXTURE_IDLE;
    s_video_fixture.deadline_ms = 0u;
    s_video_fixture.last_clear_ms = 0u;
    video_fixture_stop_motion_target();
    bsp_lift_stop();
    video_zdt_stop();
    bsp_lift_actuator_stop();
    if (magnet_off) bsp_lift_magnet_set(0);
}

static void video_fixture_pick_start(uint32_t now_ms)
{
    video_fixture_abort(1u);

    s_video_fixture.tracking = 1u;
    s_video_fixture.height_m = 0.0f;
    bsp_lift_enable(1);
    bsp_lift_actuator_stop();
    bsp_lift_servo_us(LIFT_SERVO_LEFT_US);
    s_video_fixture.deadline_ms = now_ms + 3000u;
    s_video_fixture.state = VIDEO_PICK_LEFT_WAIT;
}

static void video_fixture_place_start(uint32_t now_ms)
{
    video_fixture_abort(0u);       /* retain a held bottle until release stage */

    s_video_fixture.tracking = 1u;
    s_video_fixture.height_m = VIDEO_FIXTURE_TOP_HEIGHT_M;
    bsp_lift_enable(1);
    bsp_lift_servo_us(2100U);
    s_video_fixture.deadline_ms = now_ms + 900u;
    s_video_fixture.state = VIDEO_PLACE_SERVO_2100_WAIT;
}

static void video_fixture_home_start(uint32_t now_ms)
{
    video_fixture_abort(1u);
    s_video_fixture.tracking = 0u;
    bsp_lift_actuator_retract();
    s_video_fixture.deadline_ms = now_ms + LIFT_DEMO_ACT_HOME_MS;
    s_video_fixture.state = VIDEO_HOME_RETRACT_WAIT;
}

void proto_service(uint32_t now_ms)
{
    if ((s_video_fixture.state != VIDEO_FIXTURE_IDLE || s_video_servo_ramp.active) &&
        motion_interlock_status(now_ms) != PROTO_ACK_OK) {
        video_fixture_abort(1u);
        return;
    }
    video_servo_ramp_service(now_ms);
    if (s_video_fixture.state == VIDEO_FIXTURE_IDLE) return;
    if (s_video_fixture.state == VIDEO_PICK_PREP_ACTIVE) {
        if ((uint32_t)(now_ms - s_video_fixture.last_clear_ms) >=
            VIDEO_ZDT_CLEAR_PERIOD_MS) {
            video_zdt_clear_stall();
            s_video_fixture.last_clear_ms = now_ms;
        }
        if (video_fixture_time_reached(now_ms, s_video_fixture.deadline_ms)) {
            video_zdt_stop();
            s_video_fixture.deadline_ms = now_ms + 20u;
            s_video_fixture.state = VIDEO_PICK_LOCK_CLEAR_WAIT;
        }
        return;
    }
    if (!video_fixture_time_reached(now_ms, s_video_fixture.deadline_ms)) return;

    switch (s_video_fixture.state) {
        case VIDEO_PICK_LEFT_WAIT:
            /* Ramp the unloaded arm to avoid a full-stroke current/protection trip. */
            video_servo_ramp_start(1u, now_ms);
            s_video_fixture.deadline_ms = now_ms + 50u;
            s_video_fixture.state = VIDEO_PICK_RIGHT_CONFIRM_WAIT;
            break;
        case VIDEO_PICK_RIGHT_CONFIRM_WAIT:
            if (s_video_servo_ramp.active) {
                s_video_fixture.deadline_ms = now_ms + 50u;
                break;
            }
            /* Reassert the final PWM after the ramp before actuator extension. */
            bsp_lift_servo_us(LIFT_SERVO_RIGHT_US);
            s_video_fixture.deadline_ms = now_ms + VIDEO_EMPTY_SERVO_SETTLE_MS;
            s_video_fixture.state = VIDEO_PICK_RIGHT_FINAL_WAIT;
            break;
        case VIDEO_PICK_RIGHT_FINAL_WAIT:
            bsp_lift_actuator_extend();
            s_video_fixture.deadline_ms = now_ms + LIFT_DEMO_ACT_EXTEND_MS;
            s_video_fixture.state = VIDEO_PICK_EXTEND_WAIT;
            break;
        case VIDEO_PICK_EXTEND_WAIT:
            bsp_lift_actuator_stop();
            s_video_fixture.deadline_ms = now_ms + 500u;
            s_video_fixture.state = VIDEO_PICK_EXTEND_SETTLE_WAIT;
            break;
        case VIDEO_PICK_EXTEND_SETTLE_WAIT:
            bsp_lift_magnet_set(1);
            s_video_fixture.deadline_ms = now_ms + 2500u;
            s_video_fixture.state = VIDEO_PICK_GRIP_WAIT;
            break;
        case VIDEO_PICK_GRIP_WAIT:
            bsp_lift_stop();
            bsp_lift_enable(1);
            video_zdt_clear_stall();
            s_video_fixture.deadline_ms = now_ms + 50u;
            s_video_fixture.state = VIDEO_PICK_PREP_ENABLE_WAIT;
            break;
        case VIDEO_PICK_PREP_ENABLE_WAIT:
            video_zdt_enable(1u);
            s_video_fixture.deadline_ms = now_ms + 100u;
            s_video_fixture.state = VIDEO_PICK_PREP_START_WAIT;
            break;
        case VIDEO_PICK_PREP_START_WAIT:
            video_zdt_move(VIDEO_ZDT_UP_DIR, VIDEO_ZDT_UP_PULSES, 0x00u);
            s_video_fixture.last_clear_ms = now_ms;
            s_video_fixture.deadline_ms = now_ms + VIDEO_ZDT_PREP_MS;
            s_video_fixture.state = VIDEO_PICK_PREP_ACTIVE;
            break;
        case VIDEO_PICK_LOCK_CLEAR_WAIT:
            video_zdt_clear_stall();
            s_video_fixture.deadline_ms = now_ms + 20u;
            s_video_fixture.state = VIDEO_PICK_LOCK_RESET_WAIT;
            break;
        case VIDEO_PICK_LOCK_RESET_WAIT:
            video_zdt_reset_position();
            s_video_fixture.deadline_ms = now_ms + 20u;
            s_video_fixture.state = VIDEO_PICK_LOCK_ENABLE_WAIT;
            break;
        case VIDEO_PICK_LOCK_ENABLE_WAIT:
            video_zdt_enable(1u);
            s_video_fixture.height_m = VIDEO_FIXTURE_TOP_HEIGHT_M;
            bsp_lift_servo_us(1550u);
            s_video_fixture.deadline_ms = now_ms + 900u;
            s_video_fixture.state = VIDEO_PICK_LEFT_1550_WAIT;
            break;
        case VIDEO_PICK_LEFT_1550_WAIT:
            bsp_lift_servo_us(1700u);
            s_video_fixture.deadline_ms = now_ms + 900u;
            s_video_fixture.state = VIDEO_PICK_LEFT_1700_WAIT;
            break;
        case VIDEO_PICK_LEFT_1700_WAIT:
            bsp_lift_servo_us(1900u);
            s_video_fixture.deadline_ms = now_ms + 900u;
            s_video_fixture.state = VIDEO_PICK_LEFT_1900_WAIT;
            break;
        case VIDEO_PICK_LEFT_1900_WAIT:
            bsp_lift_servo_us(2100u);
            s_video_fixture.deadline_ms = now_ms + 900u;
            s_video_fixture.state = VIDEO_PICK_LEFT_2100_WAIT;
            break;
        case VIDEO_PICK_LEFT_2100_WAIT:
            bsp_lift_servo_us(LIFT_SERVO_LEFT_US);
            /* Loaded arm is at start; hand control to navigation promptly. */
            s_video_fixture.deadline_ms = now_ms + VIDEO_TOP_LEFT_READY_MS;
            s_video_fixture.state = VIDEO_PICK_LEFT_FINAL_WAIT;
            break;
        case VIDEO_PICK_LEFT_FINAL_WAIT:
            s_video_fixture.state = VIDEO_FIXTURE_IDLE;
            break;

        case VIDEO_PLACE_SERVO_2100_WAIT:
            bsp_lift_servo_us(1900u);
            s_video_fixture.deadline_ms = now_ms + 900u;
            s_video_fixture.state = VIDEO_PLACE_SERVO_1900_WAIT;
            break;
        case VIDEO_PLACE_SERVO_1900_WAIT:
            bsp_lift_servo_us(1700u);
            s_video_fixture.deadline_ms = now_ms + 900u;
            s_video_fixture.state = VIDEO_PLACE_SERVO_1700_WAIT;
            break;
        case VIDEO_PLACE_SERVO_1700_WAIT:
            bsp_lift_servo_us(1550u);
            s_video_fixture.deadline_ms = now_ms + 900u;
            s_video_fixture.state = VIDEO_PLACE_SERVO_1550_WAIT;
            break;
        case VIDEO_PLACE_SERVO_1550_WAIT:
            bsp_lift_servo_us(LIFT_SERVO_RIGHT_US);
            s_video_fixture.deadline_ms = now_ms + LIFT_DEMO_RIGHT_SETTLE_MS;
            s_video_fixture.state = VIDEO_PLACE_RIGHT_WAIT;
            break;
        case VIDEO_PLACE_RIGHT_WAIT:
            /* Reassert the loaded work position before any vertical motion. */
            bsp_lift_servo_us(LIFT_SERVO_RIGHT_US);
            video_zdt_move(VIDEO_ZDT_DOWN_DIR, VIDEO_ZDT_DOWN_PULSES, 0x02u);
            s_video_fixture.deadline_ms = now_ms + VIDEO_ZDT_DOWN_WAIT_MS;
            s_video_fixture.state = VIDEO_PLACE_DOWN_WAIT;
            break;
        case VIDEO_PLACE_DOWN_WAIT:
            video_zdt_stop();
            s_video_fixture.deadline_ms = now_ms + 20u;
            s_video_fixture.state = VIDEO_PLACE_LOCK_CLEAR_WAIT;
            break;
        case VIDEO_PLACE_LOCK_CLEAR_WAIT:
            video_zdt_clear_stall();
            s_video_fixture.deadline_ms = now_ms + 20u;
            s_video_fixture.state = VIDEO_PLACE_LOCK_RESET_WAIT;
            break;
        case VIDEO_PLACE_LOCK_RESET_WAIT:
            video_zdt_reset_position();
            s_video_fixture.deadline_ms = now_ms + 20u;
            s_video_fixture.state = VIDEO_PLACE_LOCK_ENABLE_WAIT;
            break;
        case VIDEO_PLACE_LOCK_ENABLE_WAIT:
            video_zdt_enable(1u);
            s_video_fixture.height_m = 0.0f;
            bsp_lift_magnet_set(0);
            s_video_fixture.deadline_ms = now_ms + LIFT_DEMO_PLACE_MS;
            s_video_fixture.state = VIDEO_PLACE_RELEASE_WAIT;
            break;
        case VIDEO_PLACE_RELEASE_WAIT:
            /* Retract fully before the empty arm makes its direct return. */
            bsp_lift_actuator_retract();
            s_video_fixture.deadline_ms = now_ms + VIDEO_FINAL_RETRACT_MS;
            s_video_fixture.state = VIDEO_PLACE_RETRACT_WAIT;
            break;
        case VIDEO_PLACE_RETRACT_WAIT:
            bsp_lift_actuator_stop();
            bsp_lift_servo_us(LIFT_SERVO_LEFT_US);
            s_video_fixture.deadline_ms = now_ms + VIDEO_EMPTY_SERVO_SETTLE_MS;
            s_video_fixture.state = VIDEO_PLACE_LEFT_FINAL_WAIT;
            break;
        case VIDEO_PLACE_LEFT_FINAL_WAIT:
            s_video_fixture.state = VIDEO_FIXTURE_IDLE;
            break;

        case VIDEO_HOME_RETRACT_WAIT:
            bsp_lift_actuator_stop();
            bsp_lift_servo_us(LIFT_SERVO_LEFT_US);
            bsp_lift_zero_position();
            s_video_fixture.state = VIDEO_FIXTURE_IDLE;
            break;
        default:
            video_fixture_abort(1u);
            break;
    }
}

static void set_lift_target_m(float target_m, uint32_t now_ms)
{
    long target_steps;

    /* Diagnostic commands touch only the PB8/TIM4 servo ramp. */
    if (video_fixture_command_matches(target_m, VIDEO_SERVO_RIGHT_COMMAND)) {
        if (s_video_fixture.state != VIDEO_FIXTURE_IDLE) video_fixture_abort(0u);
        s_video_fixture.tracking = 0u;
        video_servo_ramp_start(1u, now_ms);
        return;
    }
    if (video_fixture_command_matches(target_m, VIDEO_SERVO_LEFT_COMMAND)) {
        if (s_video_fixture.state != VIDEO_FIXTURE_IDLE) video_fixture_abort(0u);
        s_video_fixture.tracking = 0u;
        video_servo_ramp_start(0u, now_ms);
        return;
    }

    if (video_fixture_command_matches(target_m, VIDEO_FIXTURE_PLACE_COMMAND)) {
        video_fixture_place_start(now_ms);
        return;
    }
    if (video_fixture_command_matches(target_m, VIDEO_FIXTURE_PICK_COMMAND)) {
        video_fixture_pick_start(now_ms);
        return;
    }

    if (s_video_fixture.state != VIDEO_FIXTURE_IDLE) video_fixture_abort(0u);
    s_video_fixture.tracking = 0u;
    if (target_m < 0.0f) target_m = 0.0f;
    target_steps = (long)(target_m * LIFT_VIDEO_STEPS_PER_M);
    if (target_steps > LIFT_VIDEO_MAX_TARGET_STEPS) {
        target_steps = LIFT_VIDEO_MAX_TARGET_STEPS;
    }
    bsp_lift_move_to((int32_t)target_steps, LIFT_DEFAULT_PPS);
}

/* ============================================================
 * 帧分发
 * ============================================================ */
static void on_complete_frame(uint8_t type, const uint8_t *payload, uint8_t len, uint32_t now_ms)
{
    switch (type) {
        case DN_CMD_VEL: {
            if (len == 8) {
                float v, w;
                uint8_t interlock;
                memcpy(&v, &payload[0], 4);
                memcpy(&w, &payload[4], 4);
                s_state.last_cmd_vel_ms  = now_ms;
                interlock = motion_interlock_status(now_ms);
                if (interlock != PROTO_ACK_OK && (v != 0.0f || w != 0.0f)) {
                    s_state.target_linear_v = 0.0f;
                    s_state.target_angular_w = 0.0f;
                    s_state.estop_blocked_command_count++;
                    proto_send_ack(DN_CMD_VEL, interlock);
                } else {
                    s_state.target_linear_v  = v;
                    s_state.target_angular_w = w;
                    proto_send_ack(DN_CMD_VEL, PROTO_ACK_OK);
                }
            } else {
                proto_send_ack(DN_CMD_VEL, PROTO_ACK_BAD_LENGTH);
            }
            break;
        }
        case DN_HEARTBEAT:
            s_state.last_heartbeat_ms = now_ms;
            /* 心跳不需要回 ACK, 减负载 */
            break;
        case DN_EMERGENCY_STOP:
            s_state.estop_latched = 1;
            s_state.emergency_stop_request = 1;
            s_state.target_linear_v = 0.0f;
            s_state.target_angular_w = 0.0f;
            video_fixture_abort(1u);
            proto_send_ack(DN_EMERGENCY_STOP, PROTO_ACK_OK);
            break;
        case DN_CLEAR_ESTOP:
            if (len == 0) {
                if (!command_link_fresh(now_ms)) {
                    proto_send_ack(DN_CLEAR_ESTOP, PROTO_ACK_LINK_STALE);
                } else {
                    s_state.target_linear_v = 0.0f;
                    s_state.target_angular_w = 0.0f;
                    video_fixture_abort(1u);
                    s_state.estop_latched = 0;
                    proto_send_ack(DN_CLEAR_ESTOP, PROTO_ACK_OK);
                }
            } else {
                proto_send_ack(DN_CLEAR_ESTOP, PROTO_ACK_BAD_LENGTH);
            }
            break;
        case DN_SET_LIFT_HEIGHT:
            if (len == 4) {
                uint8_t interlock = motion_interlock_status(now_ms);
                if (interlock != PROTO_ACK_OK) {
                    s_state.estop_blocked_command_count++;
                    proto_send_ack(DN_SET_LIFT_HEIGHT, interlock);
                } else {
                    float target_m;
                    memcpy(&target_m, &payload[0], 4);
                    set_lift_target_m(target_m, now_ms);
                    proto_send_ack(DN_SET_LIFT_HEIGHT, PROTO_ACK_OK);
                }
            } else {
                proto_send_ack(DN_SET_LIFT_HEIGHT, PROTO_ACK_BAD_LENGTH);
            }
            break;
        case DN_SET_ELECTROMAGNET:
            if (len == 1) {
                uint8_t interlock = motion_interlock_status(now_ms);
                if (interlock != PROTO_ACK_OK && payload[0] != 0) {
                    s_state.estop_blocked_command_count++;
                    proto_send_ack(DN_SET_ELECTROMAGNET, interlock);
                } else {
                    bsp_lift_magnet_set(payload[0] != 0);
                    proto_send_ack(DN_SET_ELECTROMAGNET, PROTO_ACK_OK);
                }
            } else {
                proto_send_ack(DN_SET_ELECTROMAGNET, PROTO_ACK_BAD_LENGTH);
            }
            break;
        case DN_LIFT_HOME:
            if (len == 0) {
                uint8_t interlock = motion_interlock_status(now_ms);
                if (interlock != PROTO_ACK_OK) {
                    s_state.estop_blocked_command_count++;
                    proto_send_ack(DN_LIFT_HOME, interlock);
                } else {
                    video_fixture_home_start(now_ms);
                    proto_send_ack(DN_LIFT_HOME, PROTO_ACK_OK);
                }
            } else {
                proto_send_ack(DN_LIFT_HOME, PROTO_ACK_BAD_LENGTH);
            }
            break;
        case 0xA1:
        case 0xA2:
        case 0xA3:
            /* 本固件没接升降台/电磁铁, 回 status=1 unsupported */
            proto_send_ack(type, PROTO_ACK_UNSUPPORTED);
            break;
        default:
            proto_send_error(0xE0, "unknown down type");
            break;
    }
    (void)len; (void)payload;
}

/* ============================================================
 * 字节流喂入 (从主循环 / DMA ring 取出后送进来)
 * ============================================================ */
static void reset_parser(void)
{
    s_st = ST_WAIT_H0;
    s_idx = 0;
    s_sum_accum = 0;
}

void proto_feed_rx(const uint8_t *data, uint16_t n, uint32_t now_ms)
{
    for (uint16_t i = 0; i < n; ++i) {
        uint8_t b = data[i];
        switch (s_st) {
            case ST_WAIT_H0:
                if (b == PROTO_HDR0) { s_sum_accum = b; s_st = ST_WAIT_H1; }
                break;
            case ST_WAIT_H1:
                if (b == PROTO_HDR1) { s_sum_accum = (uint8_t)(s_sum_accum + b); s_st = ST_WAIT_TYPE; }
                else if (b == PROTO_HDR0) { s_sum_accum = b; }      /* 连续 0xAA */
                else { reset_parser(); }
                break;
            case ST_WAIT_TYPE:
                s_type = b;
                s_sum_accum = (uint8_t)(s_sum_accum + b);
                s_st = ST_WAIT_LEN;
                break;
            case ST_WAIT_LEN:
                s_len = b;
                s_sum_accum = (uint8_t)(s_sum_accum + b);
                s_idx = 0;
                if (s_len == 0) {
                    s_st = ST_WAIT_CKS;
                } else if (s_len > PROTO_FRAME_MAX - 5) {
                    reset_parser();      /* 长度异常 */
                } else {
                    s_st = ST_WAIT_PAYLOAD;
                }
                break;
            case ST_WAIT_PAYLOAD:
                s_payload[s_idx++] = b;
                s_sum_accum = (uint8_t)(s_sum_accum + b);
                if (s_idx >= s_len) s_st = ST_WAIT_CKS;
                break;
            case ST_WAIT_CKS:
                if (b == s_sum_accum) {
                    on_complete_frame(s_type, s_payload, s_len, now_ms);
                }
                /* 校验错就丢, 不报 (ROS 端会算 cks 不匹配 warn) */
                reset_parser();
                break;
        }
    }
}

/* ============================================================
 * 发包: 内部组帧 + host UART 输出 (USART2 @ PD5/PD6)
 * ============================================================ */
static void send_frame(uint8_t type, const void *payload, uint8_t len)
{
    uint8_t f[PROTO_FRAME_MAX];
    uint16_t k = 0;
    f[k++] = PROTO_HDR0;
    f[k++] = PROTO_HDR1;
    f[k++] = type;
    f[k++] = len;
    if (len > 0 && payload) {
        memcpy(&f[k], payload, len);
        k = (uint16_t)(k + len);
    }
    uint8_t sum = 0;
    for (uint16_t i = 0; i < k; ++i) sum = (uint8_t)(sum + f[i]);
    f[k++] = sum;

    host_uart_send(f, k);
}

void proto_send_basic_odom(float x, float y, float vx, float wz, float yaw_deg)
{
    uint8_t pl[20];
    memcpy(&pl[0],  &x,        4);
    memcpy(&pl[4],  &y,        4);
    memcpy(&pl[8],  &vx,       4);
    memcpy(&pl[12], &wz,       4);
    memcpy(&pl[16], &yaw_deg,  4);
    send_frame(UP_BASIC_ODOM, pl, 20);
}

void proto_send_ext_telemetry(float lift_h, float lift_v,
                              uint8_t home_sw, uint8_t top_sw,
                              uint8_t em_state, uint8_t homed,
                              float ax, float ay, float az,
                              float gx, float gy, float gz,
                              float cpu_temp_c, float bus_v)
{
    /* PayloadExtTelemetry 字节布局 (44B):
     *   float lift_height_m       [0..3]
     *   float lift_velocity_mps   [4..7]
     *   u8    home_switch         [8]
     *   u8    top_switch          [9]
     *   u8    electromagnet_state [10]
     *   u8    homed               [11]
     *   float accel_x             [12..15]
     *   float accel_y             [16..19]
     *   float accel_z             [20..23]
     *   float gyro_x              [24..27]
     *   float gyro_y              [28..31]
     *   float gyro_z              [32..35]
     *   float cpu_temp_c          [36..39]
     *   float bus_voltage_v       [40..43]
     */
    uint8_t pl[44];
    memcpy(&pl[0],  &lift_h, 4);
    memcpy(&pl[4],  &lift_v, 4);
    pl[8]  = home_sw;
    pl[9]  = top_sw;
    pl[10] = em_state;
    pl[11] = homed;
    memcpy(&pl[12], &ax, 4);
    memcpy(&pl[16], &ay, 4);
    memcpy(&pl[20], &az, 4);
    memcpy(&pl[24], &gx, 4);
    memcpy(&pl[28], &gy, 4);
    memcpy(&pl[32], &gz, 4);
    memcpy(&pl[36], &cpu_temp_c, 4);
    memcpy(&pl[40], &bus_v, 4);
    send_frame(UP_EXT_TELEMETRY, pl, 44);
}

void proto_send_safety_state(uint8_t estop_latched,
                             uint8_t emergency_active,
                             uint16_t blocked_command_count)
{
    uint8_t pl[4];
    pl[0] = estop_latched ? 1u : 0u;
    pl[1] = emergency_active ? 1u : 0u;
    pl[2] = (uint8_t)(blocked_command_count & 0xFFu);
    pl[3] = (uint8_t)((blocked_command_count >> 8) & 0xFFu);
    send_frame(UP_SAFETY_STATE, pl, 4);
}

void proto_send_firmware_info(uint16_t protocol_version,
                              uint16_t capabilities,
                              uint32_t build_id,
                              uint8_t test_mode,
                              uint8_t hw_variant)
{
    uint8_t pl[12];
    pl[0] = (uint8_t)(protocol_version & 0xFFu);
    pl[1] = (uint8_t)((protocol_version >> 8) & 0xFFu);
    pl[2] = (uint8_t)(capabilities & 0xFFu);
    pl[3] = (uint8_t)((capabilities >> 8) & 0xFFu);
    pl[4] = (uint8_t)(build_id & 0xFFu);
    pl[5] = (uint8_t)((build_id >> 8) & 0xFFu);
    pl[6] = (uint8_t)((build_id >> 16) & 0xFFu);
    pl[7] = (uint8_t)((build_id >> 24) & 0xFFu);
    pl[8] = test_mode;
    pl[9] = hw_variant;
    pl[10] = 0u;
    pl[11] = 0u;
    send_frame(UP_FIRMWARE_INFO, pl, 12);
}

void proto_send_ack(uint8_t ack_for_type, uint8_t status)
{
    uint8_t pl[3];
    pl[0] = ack_for_type;
    pl[1] = status;
    pl[2] = 0;
    send_frame(UP_ACK, pl, 3);
}

void proto_send_error(uint8_t code, const char *msg)
{
    uint8_t pl[1 + 32];
    pl[0] = code;
    uint8_t n = 0;
    if (msg) {
        while (n < 31 && msg[n]) { pl[1 + n] = (uint8_t)msg[n]; n++; }
    }
    send_frame(UP_ERROR, pl, (uint8_t)(1 + n));
}
