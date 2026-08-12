#ifndef BSP_LIFT_H
#define BSP_LIFT_H

#include <stdint.h>

/*
 * Lift stage wiring, connector group 5:
 *   host command UART remains USART2 PD5/PD6 in bsp_uart.c
 *   stepper driver UART: PC12=UART5_TX, PD2=UART5_RX, 115200 8N1
 *   stepper enable:      PD10
 *   stepper pulse:       PC9 = TIM8_CH4 AF3
 *   stepper direction:   PD7
 *   actuator relay IN1:  PC13, active high
 *   actuator relay IN2:  PC0,  active high
 *   servo PWM:           PB8 = TIM4_CH3 AF2
 *   electromagnet IO:    PE0, active high
 *
 * Relay truth used by the bottle-transfer fixture:
 *   EXTEND  = IN1 low,  IN2 high  (NC1 +12V to red, black to GND)
 *   RETRACT = IN1 high, IN2 high  (NO1 reverse rail to red, black to GND)
 *   STOP    = IN1 low,  IN2 low   (black line disconnected)
 */

#define LIFT_EN_ACTIVE_LOW       1
#define LIFT_DIR_INVERT          0
#define LIFT_MIN_PPS             1U
#define LIFT_MAX_PPS             16000U
#define LIFT_DEBUG_MAX_PPS       10U
#define LIFT_DEFAULT_PPS         10U
#define LIFT_DEFAULT_UP_STEPS    25L
#define LIFT_PICK_DELAY_MS       500U

#define LIFT_SAFE_ZERO_MARGIN_STEPS 500L
#define LIFT_SAFE_ZERO_MAX_STEPS    5000L
#define LIFT_SAFE_ZERO_PPS          10U

#define LIFT_RAMP_MIN_PPS       20U
#define LIFT_RAMP_MAX_STEPS     1200U

#define LIFT_SEGMENT_STEPS      25U
#define LIFT_SEGMENT_DWELL_MS   500U

#define LIFT_JOG_STEPS          25L
#define LIFT_JOG_PPS            10U
#define LIFT_STEP_TEST_STEPS    25L
#define LIFT_STEP_TEST_PPS      10U

#define LIFT_ACT_RELAY_GUARD_MS  80U

#define LIFT_SERVO_MIN_US        500U
#define LIFT_SERVO_MAX_US        2500U
#define LIFT_SERVO_LEFT_US       2300U
#define LIFT_SERVO_RIGHT_US      1400U
#define LIFT_SERVO_HOME_US       LIFT_SERVO_LEFT_US

#define LIFT_AUTO_TRAVEL_STEPS   500L
#define LIFT_AUTO_SPEED_PPS      10U
#define LIFT_AUTO_GRIP_MS        700U
#define LIFT_AUTO_EXTEND_MS      6500U
#define LIFT_AUTO_RETRACT_MS     6500U
#define LIFT_AUTO_SERVO_MS       900U
#define LIFT_AUTO_RELEASE_MS     700U
#define LIFT_AUTO_TOP_HOLD_PPS   0U

#define LIFT_DEMO_LIFT_STEPS     50L
#define LIFT_DEMO_SPEED_PPS      10U
#define LIFT_DEMO_NAV_DELAY_MS   2500U
#define LIFT_DEMO_SERVO_MS       900U
#define LIFT_DEMO_GRIP_MS        900U
#define LIFT_DEMO_RELEASE_MS     900U
#define LIFT_DEMO_FAULT_UP_PPS   1200U
#define LIFT_DEMO_FAULT_UP_MS    10000U
#define LIFT_DEMO_DROP_WAIT_MS   2000U
#define LIFT_DEMO_RIGHT_SETTLE_MS 5000U
#define LIFT_DEMO_ACT_HOME_MS    11000U
#define LIFT_DEMO_ACT_EXTEND_MS  10000U
#define LIFT_DEMO_ACT_RETRACT_MS 10000U
#define LIFT_DEMO_PLACE_MS       1200U

typedef enum {
    LIFT_ACT_STOPPED = 0,
    LIFT_ACT_EXTENDING = 1,
    LIFT_ACT_RETRACTING = 2
} LiftActuatorState;

void    bsp_lift_init(void);
void    bsp_lift_enable(uint8_t on);
uint8_t bsp_lift_enabled(void);

void    bsp_lift_move_steps(int32_t steps, uint32_t pps);
void    bsp_lift_move_to(int32_t target_steps, uint32_t pps);
void    bsp_lift_set_speed_pps(int32_t pps);
/* Call on every main-loop pass while finite moves run. */
void    bsp_lift_service(uint32_t now_ms);
void    bsp_lift_stop(void);
uint8_t bsp_lift_busy(void);

void    bsp_lift_zero_position(void);
int32_t bsp_lift_position_steps(void);
int32_t bsp_lift_remaining_steps(void);

void    bsp_lift_uart5_send(const uint8_t *data, uint16_t len);

void    bsp_lift_magnet_set(uint8_t on);
uint8_t bsp_lift_magnet_on(void);

void    bsp_lift_actuator_extend(void);
void    bsp_lift_actuator_retract(void);
void    bsp_lift_actuator_stop(void);
uint8_t bsp_lift_actuator_state(void);

void     bsp_lift_servo_us(uint16_t pulse_us);
void     bsp_lift_servo_home(void);
void     bsp_lift_servo_left(void);
void     bsp_lift_servo_right(void);
uint16_t bsp_lift_servo_pulse_us(void);
void     bsp_lift_servo_hold(void);
uint8_t  bsp_lift_servo_pwm_valid(void);

void    bsp_lift_raw_pins(uint8_t en_hi, uint8_t dir_hi, uint8_t pul_hi);
void    bsp_lift_bitbang_pulses(uint8_t en_hi, uint8_t dir_hi,
                                uint32_t pulses, uint32_t half_ms);

#endif
