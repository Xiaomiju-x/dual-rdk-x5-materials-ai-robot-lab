/*
 * Public STM32F407 station sequence skeleton.
 *
 * This file is hardware-neutral pseudocode for review. It intentionally omits
 * pin mapping, timer parameters, driver settings, and direct deployment logic.
 */

#include <stdbool.h>
#include <stdint.h>

typedef enum {
    ST_IDLE = 0,
    ST_MOVE_TO_WORK,
    ST_EXTEND_PUSHER,
    ST_MAGNET_ON,
    ST_RETRACT_PUSHER,
    ST_MOVE_TO_TRANSFER,
    ST_RELEASE,
    ST_RETURN_HOME,
    ST_FAULT
} StationState;

typedef struct {
    bool top_limit;
    bool bottom_limit;
    bool driver_alarm;
    bool current_over_limit;
    bool operator_stop;
} SafetyInputs;

typedef struct {
    StationState state;
    uint32_t entered_ms;
    bool holding_bottle;
} StationContext;

static uint32_t now_ms(void);
static void servo_to_work_pose(void);
static void servo_to_transfer_pose(void);
static void pusher_extend_safe(void);
static void pusher_retract_safe(void);
static void magnet_set(bool enable);
static void lift_stop(void);
static void signal_status(StationState state);

static bool safety_fault(const SafetyInputs *in)
{
    return in->driver_alarm || in->current_over_limit || in->operator_stop;
}

static bool elapsed(const StationContext *ctx, uint32_t duration_ms)
{
    return (now_ms() - ctx->entered_ms) >= duration_ms;
}

static void enter_state(StationContext *ctx, StationState next)
{
    ctx->state = next;
    ctx->entered_ms = now_ms();
    signal_status(next);
}

void station_init(StationContext *ctx)
{
    ctx->state = ST_IDLE;
    ctx->entered_ms = now_ms();
    ctx->holding_bottle = false;
    magnet_set(false);
    pusher_retract_safe();
    lift_stop();
}

void station_start_pick(StationContext *ctx, const SafetyInputs *in)
{
    if (ctx->state != ST_IDLE) {
        return;
    }
    if (safety_fault(in)) {
        enter_state(ctx, ST_FAULT);
        return;
    }
    enter_state(ctx, ST_MOVE_TO_WORK);
}

void station_tick(StationContext *ctx, const SafetyInputs *in)
{
    if (safety_fault(in)) {
        magnet_set(false);
        lift_stop();
        enter_state(ctx, ST_FAULT);
        return;
    }

    switch (ctx->state) {
    case ST_IDLE:
        lift_stop();
        break;

    case ST_MOVE_TO_WORK:
        servo_to_work_pose();
        if (elapsed(ctx, 600U)) {
            enter_state(ctx, ST_EXTEND_PUSHER);
        }
        break;

    case ST_EXTEND_PUSHER:
        pusher_extend_safe();
        if (elapsed(ctx, 500U)) {
            enter_state(ctx, ST_MAGNET_ON);
        }
        break;

    case ST_MAGNET_ON:
        magnet_set(true);
        ctx->holding_bottle = true;
        if (elapsed(ctx, 250U)) {
            enter_state(ctx, ST_RETRACT_PUSHER);
        }
        break;

    case ST_RETRACT_PUSHER:
        pusher_retract_safe();
        if (elapsed(ctx, 500U)) {
            enter_state(ctx, ST_MOVE_TO_TRANSFER);
        }
        break;

    case ST_MOVE_TO_TRANSFER:
        servo_to_transfer_pose();
        if (elapsed(ctx, 700U)) {
            enter_state(ctx, ST_RELEASE);
        }
        break;

    case ST_RELEASE:
        magnet_set(false);
        ctx->holding_bottle = false;
        if (elapsed(ctx, 250U)) {
            enter_state(ctx, ST_RETURN_HOME);
        }
        break;

    case ST_RETURN_HOME:
        pusher_retract_safe();
        if (elapsed(ctx, 500U)) {
            enter_state(ctx, ST_IDLE);
        }
        break;

    case ST_FAULT:
    default:
        magnet_set(false);
        lift_stop();
        pusher_retract_safe();
        break;
    }
}
