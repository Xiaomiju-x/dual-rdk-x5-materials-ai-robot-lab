#include "bsp_lift.h"
#include "bsp_clock.h"
#include "bsp_systick.h"
#include "stm32f4xx.h"

typedef enum {
    LIFT_MOTION_IDLE = 0,
    LIFT_MOTION_FINITE_GPIO,
    LIFT_MOTION_CONTINUOUS_TIMER,
    LIFT_MOTION_BITBANG_GPIO
} lift_motion_mode_t;

typedef struct {
    volatile uint8_t enabled;
    volatile uint8_t busy;
    volatile uint8_t phase;
    volatile uint8_t stop_after_fall;
    volatile uint8_t motion_mode;
    volatile uint8_t restore_af_on_finish;
    volatile int8_t dir;
    volatile uint32_t half_iv;
    volatile uint32_t half_ms;
    volatile uint32_t next_edge_ms;
    volatile uint32_t start_pps;
    volatile uint32_t target_pps;
    volatile uint32_t ramp_steps;
    volatile uint32_t move_abs_steps;
    volatile uint32_t moved_steps;
    volatile int32_t pos_steps;
} lift_state_t;

static lift_state_t s_lift;
static volatile uint8_t  s_magnet_on = 0;
static volatile uint8_t  s_actuator_state = LIFT_ACT_STOPPED;
static volatile uint16_t s_servo_us = LIFT_SERVO_HOME_US;

static uint32_t lift_pps_to_half_iv(uint32_t pps)
{
    uint32_t half_iv;
    if (pps < LIFT_MIN_PPS) pps = LIFT_MIN_PPS;
    if (pps > LIFT_MAX_PPS) pps = LIFT_MAX_PPS;
    half_iv = 1000000UL / (2UL * pps);
    if (half_iv < 8U) half_iv = 8U;
    if (half_iv > 0xFFFFU) half_iv = 0xFFFFU;
    return half_iv;
}

static void gpio_af(GPIO_TypeDef *port, uint8_t pin, uint8_t af)
{
    port->MODER   &= ~(3U << (pin * 2));
    port->MODER   |=  (2U << (pin * 2));
    port->OTYPER  &= ~(1U << pin);
    port->OSPEEDR |=  (3U << (pin * 2));
    port->PUPDR   &= ~(3U << (pin * 2));
    if (pin < 8) {
        port->AFR[0] &= ~(0xFU << (pin * 4));
        port->AFR[0] |=  ((uint32_t)af << (pin * 4));
    } else {
        port->AFR[1] &= ~(0xFU << ((pin - 8) * 4));
        port->AFR[1] |=  ((uint32_t)af << ((pin - 8) * 4));
    }
}

static void gpio_out(GPIO_TypeDef *port, uint8_t pin, uint8_t hi)
{
    port->MODER   &= ~(3U << (pin * 2));
    port->MODER   |=  (1U << (pin * 2));
    port->OTYPER  &= ~(1U << pin);
    port->OSPEEDR |=  (2U << (pin * 2));
    port->PUPDR   &= ~(3U << (pin * 2));
    if (hi) port->BSRR = (uint32_t)(1U << pin);
    else    port->BSRR = (uint32_t)(1U << pin) << 16;
}

static inline void gpio_write(GPIO_TypeDef *port, uint8_t pin, uint8_t hi)
{
    if (hi) port->BSRR = (uint32_t)(1U << pin);
    else    port->BSRR = (uint32_t)(1U << pin) << 16;
}

static void lift_pul_gpio_mode(uint8_t hi)
{
    gpio_out(GPIOC, 9, hi);
}

static void lift_pul_af_mode(void)
{
    gpio_af(GPIOC, 9, 3);
}

static void lift_oc_mode(uint32_t mode3)
{
    uint32_t v = TIM8->CCMR2;
    v &= ~(7U << 12);       /* CH4 OC4M */
    v |=  (mode3 << 12);
    TIM8->CCMR2 = v;
}

static void tim8_ch4_init(void)
{
    RCC->APB2ENR |= RCC_APB2ENR_TIM8EN;

    TIM8->CR1   = 0;
    TIM8->CR2   = 0;
    TIM8->PSC   = (uint16_t)(APB2_TIMCLK / 1000000UL - 1U);
    TIM8->ARR   = 0xFFFFU;
    TIM8->CCR4  = 0;
    TIM8->CCMR2 = (4U << 12);             /* CH4 force inactive while idle */
    TIM8->CCER  = TIM_CCER_CC4E;
    TIM8->BDTR |= TIM_BDTR_MOE;
    TIM8->EGR   = TIM_EGR_UG;
    TIM8->SR    = 0;
    TIM8->DIER  = 0;

    NVIC_SetPriority(TIM8_CC_IRQn, 1);
    NVIC_EnableIRQ(TIM8_CC_IRQn);

    TIM8->CR1 |= TIM_CR1_CEN;
}

static void tim4_ch3_servo_init(uint16_t pulse_us)
{
    RCC->APB1ENR |= RCC_APB1ENR_TIM4EN;
    gpio_af(GPIOB, 8, 2);                   /* restore PB8 = TIM4_CH3 AF2 */

    TIM4->CR1   = 0;
    TIM4->CR2   = 0;
    TIM4->PSC   = (uint16_t)(APB1_TIMCLK / 1000000UL - 1U);
    TIM4->ARR   = 20000U - 1U;              /* 50 Hz servo frame */
    TIM4->CCR3  = pulse_us;
    TIM4->CCMR2 = (6U << 4) | TIM_CCMR2_OC3PE;  /* CH3 PWM mode 1 */
    TIM4->CCER  = TIM_CCER_CC3E;
    TIM4->EGR   = TIM_EGR_UG;
    TIM4->SR    = 0;
    TIM4->CR1   = TIM_CR1_ARPE | TIM_CR1_CEN;
}

static void uart5_driver_init(void)
{
    RCC->APB1ENR |= RCC_APB1ENR_UART5EN;

    gpio_af(GPIOC, 12, 8);  /* UART5_TX */
    gpio_af(GPIOD, 2,  8);  /* UART5_RX */

    UART5->CR1 = 0;
    UART5->CR2 = 0;
    UART5->CR3 = 0;
    UART5->BRR = (uint32_t)(APB1_HZ / 115200U);
    UART5->CR1 = USART_CR1_RE | USART_CR1_TE | USART_CR1_UE;
}

void bsp_lift_init(void)
{
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOBEN |
                    RCC_AHB1ENR_GPIOCEN |
                    RCC_AHB1ENR_GPIODEN |
                    RCC_AHB1ENR_GPIOEEN;

    gpio_af(GPIOC, 9, 3);       /* TIM8_CH4 pulse */
    gpio_out(GPIOD, 7, 0);      /* direction */

#if LIFT_EN_ACTIVE_LOW
    gpio_out(GPIOD, 10, 1);     /* disabled */
#else
    gpio_out(GPIOD, 10, 0);     /* disabled */
#endif

    gpio_out(GPIOC, 13, 0);     /* actuator relay IN1, off/select extend */
    gpio_out(GPIOC, 0,  0);     /* actuator relay IN2, off */
    gpio_out(GPIOE, 0,  0);     /* electromagnet off */
    gpio_af(GPIOB, 8, 2);       /* TIM4_CH3 servo PWM */

    s_lift.enabled = 0;
    s_lift.busy = 0;
    s_lift.phase = 0;
    s_lift.stop_after_fall = 0;
    s_lift.motion_mode = LIFT_MOTION_IDLE;
    s_lift.restore_af_on_finish = 0;
    s_lift.dir = 1;
    s_lift.half_iv = 0;
    s_lift.half_ms = 0;
    s_lift.next_edge_ms = 0;
    s_lift.start_pps = LIFT_RAMP_MIN_PPS;
    s_lift.target_pps = LIFT_DEFAULT_PPS;
    s_lift.ramp_steps = 0;
    s_lift.move_abs_steps = 0;
    s_lift.moved_steps = 0;
    s_lift.pos_steps = 0;
    s_magnet_on = 0;
    s_actuator_state = LIFT_ACT_STOPPED;
    s_servo_us = LIFT_SERVO_HOME_US;

    uart5_driver_init();
    tim8_ch4_init();
    tim4_ch3_servo_init(LIFT_SERVO_HOME_US);
}

void bsp_lift_enable(uint8_t on)
{
    if (on) {
#if LIFT_EN_ACTIVE_LOW
        gpio_write(GPIOD, 10, 0);
#else
        gpio_write(GPIOD, 10, 1);
#endif
        s_lift.enabled = 1;
    } else {
        bsp_lift_stop();
#if LIFT_EN_ACTIVE_LOW
        gpio_write(GPIOD, 10, 1);
#else
        gpio_write(GPIOD, 10, 0);
#endif
        s_lift.enabled = 0;
    }
}

uint8_t bsp_lift_enabled(void)
{
    return s_lift.enabled;
}

static uint32_t lift_pps_to_half_ms(uint32_t pps)
{
    uint32_t half_ms;
    if (pps < LIFT_MIN_PPS) pps = LIFT_MIN_PPS;
    if (pps > LIFT_DEBUG_MAX_PPS) pps = LIFT_DEBUG_MAX_PPS;
    half_ms = 1000UL / (2UL * pps);
    if (half_ms < 1U) half_ms = 1U;
    if (half_ms > 1000U) half_ms = 1000U;
    return half_ms;
}

void bsp_lift_move_steps(int32_t steps, uint32_t pps)
{
    uint8_t dir_hi;
    uint32_t abs_steps;
    uint32_t half_ms;

    if (steps == 0) {
        bsp_lift_stop();
        return;
    }

    bsp_lift_stop();
    if (!s_lift.enabled) bsp_lift_enable(1);

    dir_hi = (steps > 0) ? 1U : 0U;
#if LIFT_DIR_INVERT
    dir_hi = !dir_hi;
#endif

    abs_steps = (steps > 0)
        ? (uint32_t)steps
        : (uint32_t)(-(int64_t)steps);
    half_ms = lift_pps_to_half_ms(pps);

    TIM8->DIER &= ~TIM_DIER_CC4IE;
    lift_oc_mode(4U);
    lift_pul_gpio_mode(0);
    gpio_write(GPIOD, 7, dir_hi);

    __disable_irq();
    s_lift.dir = (steps > 0) ? 1 : -1;
    s_lift.half_iv = 0;
    s_lift.half_ms = half_ms;
    s_lift.next_edge_ms = millis() + 20U;
    s_lift.start_pps = pps;
    s_lift.target_pps = pps;
    s_lift.ramp_steps = 0;
    s_lift.move_abs_steps = abs_steps;
    s_lift.moved_steps = 0;
    s_lift.stop_after_fall = 0;
    s_lift.motion_mode = LIFT_MOTION_FINITE_GPIO;
    s_lift.restore_af_on_finish = 0;
    s_lift.phase = 0;
    s_lift.busy = 1;
    __enable_irq();
}

void bsp_lift_move_to(int32_t target_steps, uint32_t pps)
{
    int32_t now = bsp_lift_position_steps();
    bsp_lift_move_steps(target_steps - now, pps);
}

void bsp_lift_set_speed_pps(int32_t pps)
{
    if (pps == 0) {
        bsp_lift_stop();
        return;
    }
    if (pps > (int32_t)LIFT_MAX_PPS) pps = (int32_t)LIFT_MAX_PPS;
    if (pps < -(int32_t)LIFT_MAX_PPS) pps = -(int32_t)LIFT_MAX_PPS;

    uint32_t abs_pps = (pps > 0) ? (uint32_t)pps : (uint32_t)(-pps);
    if (abs_pps < LIFT_MIN_PPS) {
        bsp_lift_stop();
        return;
    }

    bsp_lift_stop();
    if (!s_lift.enabled) bsp_lift_enable(1);
    lift_pul_af_mode();

    uint8_t dir_hi = (pps > 0) ? 1U : 0U;
#if LIFT_DIR_INVERT
    dir_hi = !dir_hi;
#endif
    gpio_write(GPIOD, 7, dir_hi);

    uint32_t half_iv = lift_pps_to_half_iv(abs_pps);

    __disable_irq();
    s_lift.dir = (pps > 0) ? 1 : -1;
    s_lift.half_iv = half_iv;
    s_lift.start_pps = abs_pps;
    s_lift.target_pps = abs_pps;
    s_lift.ramp_steps = 0;
    s_lift.move_abs_steps = 0;       /* 0 means continuous */
    s_lift.moved_steps = 0;
    s_lift.stop_after_fall = 0;
    s_lift.motion_mode = LIFT_MOTION_CONTINUOUS_TIMER;
    s_lift.restore_af_on_finish = 0;
    s_lift.phase = 0;
    s_lift.busy = 1;
    TIM8->CCR4 = (uint16_t)(TIM8->CNT + half_iv);
    TIM8->SR = ~TIM_SR_CC4IF;
    lift_oc_mode(3U);
    TIM8->DIER |= TIM_DIER_CC4IE;
    __enable_irq();
}

static int lift_time_reached(uint32_t now_ms, uint32_t deadline_ms)
{
    return ((int32_t)(now_ms - deadline_ms) >= 0);
}

static void lift_finish_gpio_motion(void)
{
    uint8_t restore_af = s_lift.restore_af_on_finish;

    gpio_write(GPIOC, 9, 0);
    __disable_irq();
    s_lift.busy = 0;
    s_lift.phase = 0;
    s_lift.stop_after_fall = 0;
    s_lift.motion_mode = LIFT_MOTION_IDLE;
    s_lift.restore_af_on_finish = 0;
    s_lift.half_ms = 0;
    s_lift.next_edge_ms = 0;
    s_lift.move_abs_steps = 0;
    s_lift.moved_steps = 0;
    __enable_irq();

    if (restore_af) lift_pul_af_mode();
}

void bsp_lift_service(uint32_t now_ms)
{
    uint8_t mode = s_lift.motion_mode;

    if (!s_lift.busy ||
        (mode != LIFT_MOTION_FINITE_GPIO && mode != LIFT_MOTION_BITBANG_GPIO) ||
        !lift_time_reached(now_ms, s_lift.next_edge_ms)) {
        return;
    }

    /* phase 2 preserves the final low half-period before reporting complete. */
    if (s_lift.phase == 2U) {
        lift_finish_gpio_motion();
        return;
    }

    if (s_lift.phase == 0U) {
        gpio_write(GPIOC, 9, 1);
        s_lift.phase = 1U;
        if (mode == LIFT_MOTION_FINITE_GPIO) {
            s_lift.pos_steps += s_lift.dir;
        }
        s_lift.moved_steps++;
        s_lift.next_edge_ms = now_ms + s_lift.half_ms;
        return;
    }

    gpio_write(GPIOC, 9, 0);
    s_lift.phase = 0U;
    if (s_lift.moved_steps >= s_lift.move_abs_steps) {
        s_lift.phase = 2U;
        s_lift.next_edge_ms = now_ms + s_lift.half_ms;
    } else if (mode == LIFT_MOTION_FINITE_GPIO &&
               LIFT_SEGMENT_STEPS != 0U &&
               (s_lift.moved_steps % LIFT_SEGMENT_STEPS) == 0U) {
        s_lift.next_edge_ms = now_ms + s_lift.half_ms + LIFT_SEGMENT_DWELL_MS;
    } else {
        s_lift.next_edge_ms = now_ms + s_lift.half_ms;
    }
}

void bsp_lift_stop(void)
{
    __disable_irq();
    TIM8->DIER &= ~TIM_DIER_CC4IE;
    lift_oc_mode(4U);                  /* force inactive */
    s_lift.busy = 0;
    s_lift.phase = 0;
    s_lift.stop_after_fall = 0;
    s_lift.motion_mode = LIFT_MOTION_IDLE;
    s_lift.restore_af_on_finish = 0;
    s_lift.half_ms = 0;
    s_lift.next_edge_ms = 0;
    s_lift.ramp_steps = 0;
    s_lift.move_abs_steps = 0;
    s_lift.moved_steps = 0;
    __enable_irq();
    lift_pul_gpio_mode(0);
}

uint8_t bsp_lift_busy(void)
{
    return s_lift.busy;
}

void bsp_lift_zero_position(void)
{
    __disable_irq();
    s_lift.pos_steps = 0;
    s_lift.moved_steps = 0;
    __enable_irq();
}

int32_t bsp_lift_position_steps(void)
{
    int32_t pos;
    __disable_irq();
    pos = s_lift.pos_steps;
    __enable_irq();
    return pos;
}

int32_t bsp_lift_remaining_steps(void)
{
    int32_t rem;
    __disable_irq();
    rem = s_lift.busy
        ? ((int32_t)s_lift.move_abs_steps - (int32_t)s_lift.moved_steps)
        : 0;
    __enable_irq();
    return rem > 0 ? rem : 0;
}

void bsp_lift_uart5_send(const uint8_t *data, uint16_t len)
{
    for (uint16_t i = 0; i < len; ++i) {
        while (!(UART5->SR & USART_SR_TXE)) { }
        UART5->DR = data[i];
    }
    while (!(UART5->SR & USART_SR_TC)) { }
}

void bsp_lift_magnet_set(uint8_t on)
{
    gpio_write(GPIOE, 0, on ? 1U : 0U);
    s_magnet_on = on ? 1U : 0U;
}

uint8_t bsp_lift_magnet_on(void)
{
    return s_magnet_on;
}

void bsp_lift_actuator_extend(void)
{
    gpio_write(GPIOC, 13, 0);   /* select NC1 +12V before enabling black line */
    delay_ms(LIFT_ACT_RELAY_GUARD_MS);
    gpio_write(GPIOC, 0, 1);
    s_actuator_state = LIFT_ACT_EXTENDING;
}

void bsp_lift_actuator_retract(void)
{
    gpio_write(GPIOC, 0, 1);    /* enable black line first */
    delay_ms(LIFT_ACT_RELAY_GUARD_MS);
    gpio_write(GPIOC, 13, 1);   /* switch red line to reverse rail */
    s_actuator_state = LIFT_ACT_RETRACTING;
}

void bsp_lift_actuator_stop(void)
{
    /* Disconnect the powered black line first; emergency stop must not wait. */
    gpio_write(GPIOC, 0, 0);
    gpio_write(GPIOC, 13, 0);
    s_actuator_state = LIFT_ACT_STOPPED;
}

uint8_t bsp_lift_actuator_state(void)
{
    return s_actuator_state;
}

void bsp_lift_servo_us(uint16_t pulse_us)
{
    if (pulse_us < LIFT_SERVO_MIN_US) pulse_us = LIFT_SERVO_MIN_US;
    if (pulse_us > LIFT_SERVO_MAX_US) pulse_us = LIFT_SERVO_MAX_US;
    s_servo_us = pulse_us;
    if (!bsp_lift_servo_pwm_valid()) {
        tim4_ch3_servo_init(pulse_us);
        return;
    }
    TIM4->CCR3 = pulse_us;
}

void bsp_lift_servo_home(void)
{
    bsp_lift_servo_us(LIFT_SERVO_HOME_US);
}

void bsp_lift_servo_left(void)
{
    bsp_lift_servo_us(LIFT_SERVO_LEFT_US);
}

void bsp_lift_servo_right(void)
{
    bsp_lift_servo_us(LIFT_SERVO_RIGHT_US);
}

uint16_t bsp_lift_servo_pulse_us(void)
{
    return s_servo_us;
}

uint8_t bsp_lift_servo_pwm_valid(void)
{
    const uint32_t expected_psc = APB1_TIMCLK / 1000000UL - 1U;
    const uint32_t expected_ccmr2 = (6U << 4) | TIM_CCMR2_OC3PE;

    if ((RCC->AHB1ENR & RCC_AHB1ENR_GPIOBEN) == 0U) return 0U;
    if ((RCC->APB1ENR & RCC_APB1ENR_TIM4EN) == 0U) return 0U;
    if (((GPIOB->MODER >> 16) & 0x3U) != 0x2U) return 0U;
    if ((GPIOB->AFR[1] & 0xFU) != 0x2U) return 0U;
    if (TIM4->PSC != expected_psc || TIM4->ARR != (20000U - 1U)) return 0U;
    if ((TIM4->CCMR2 & 0xFFU) != expected_ccmr2) return 0U;
    if ((TIM4->CCER & TIM_CCER_CC3E) == 0U) return 0U;
    if ((TIM4->CR1 & TIM_CR1_CEN) == 0U) return 0U;
    return 1U;
}

void bsp_lift_servo_hold(void)
{
    /* Halt motion on estop without allowing a held payload to rotate freely. */
    if (!bsp_lift_servo_pwm_valid()) tim4_ch3_servo_init(s_servo_us);
}

void bsp_lift_raw_pins(uint8_t en_hi, uint8_t dir_hi, uint8_t pul_hi)
{
    bsp_lift_stop();
    gpio_write(GPIOD, 10, en_hi ? 1U : 0U);
    gpio_write(GPIOD, 7,  dir_hi ? 1U : 0U);
    lift_pul_gpio_mode(pul_hi ? 1U : 0U);
}

void bsp_lift_bitbang_pulses(uint8_t en_hi, uint8_t dir_hi,
                             uint32_t pulses, uint32_t half_ms)
{
    if (half_ms < 1U) half_ms = 1U;
    if (half_ms > 1000U) half_ms = 1000U;
    if (pulses > 20000U) pulses = 20000U;

    bsp_lift_stop();
    gpio_write(GPIOD, 10, en_hi ? 1U : 0U);
    gpio_write(GPIOD, 7,  dir_hi ? 1U : 0U);
    lift_pul_gpio_mode(0);

    if (pulses == 0U) {
        lift_pul_af_mode();
        return;
    }

    __disable_irq();
    TIM8->DIER &= ~TIM_DIER_CC4IE;
    s_lift.busy = 1;
    s_lift.phase = 0;
    s_lift.stop_after_fall = 0;
    s_lift.motion_mode = LIFT_MOTION_BITBANG_GPIO;
    s_lift.restore_af_on_finish = 1;
    s_lift.half_ms = half_ms;
    s_lift.next_edge_ms = millis() + 20U;
    s_lift.move_abs_steps = pulses;
    s_lift.moved_steps = 0;
    __enable_irq();
}

void TIM8_CC_IRQHandler(void)
{
    if (!(TIM8->SR & TIM_SR_CC4IF)) return;
    TIM8->SR = ~TIM_SR_CC4IF;

    if (!s_lift.busy || s_lift.motion_mode != LIFT_MOTION_CONTINUOUS_TIMER) return;

    if (s_lift.phase == 0) {
        s_lift.phase = 1;
        s_lift.pos_steps += s_lift.dir;
        s_lift.moved_steps++;
        if (s_lift.move_abs_steps != 0 &&
            s_lift.moved_steps >= s_lift.move_abs_steps) {
            s_lift.stop_after_fall = 1;
        } else if (s_lift.move_abs_steps != 0 && s_lift.ramp_steps != 0) {
            uint32_t done = s_lift.moved_steps;
            uint32_t remaining = s_lift.move_abs_steps - s_lift.moved_steps;
            uint32_t ramp_idx = (done < remaining) ? done : remaining;
            uint32_t pps;
            if (ramp_idx > s_lift.ramp_steps) ramp_idx = s_lift.ramp_steps;
            pps = s_lift.start_pps;
            if (s_lift.target_pps > s_lift.start_pps) {
                pps += ((s_lift.target_pps - s_lift.start_pps) * ramp_idx) /
                       s_lift.ramp_steps;
            }
            s_lift.half_iv = lift_pps_to_half_iv(pps);
        }
    } else {
        s_lift.phase = 0;
        if (s_lift.stop_after_fall) {
            TIM8->DIER &= ~TIM_DIER_CC4IE;
            lift_oc_mode(4U);
            s_lift.busy = 0;
            s_lift.stop_after_fall = 0;
            return;
        }
    }

    {
        uint16_t next = (uint16_t)(TIM8->CCR4 + s_lift.half_iv);
        if ((int16_t)(next - (uint16_t)TIM8->CNT) <= 0)
            next = (uint16_t)(TIM8->CNT + s_lift.half_iv);
        TIM8->CCR4 = next;
    }
}
