#include "bsp_led.h"
#include "stm32f4xx.h"

/* PA1 板载 LED, 低电平点亮 */

static LedMode  s_mode = LED_MODE_SLOW_BLINK;
static uint32_t s_phase_ms = 0;

static inline void led_on (void) { GPIOA->BSRR = (uint32_t)(1U << 1) << 16; }   /* BR1: reset (低=亮) */
static inline void led_off(void) { GPIOA->BSRR =  1U << 1; }                    /* BS1: set   (高=灭) */

void bsp_led_init(void)
{
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;

    /* PA1 推挽输出, 中速 */
    GPIOA->MODER   &= ~(3U << (1 * 2));
    GPIOA->MODER   |=  (1U << (1 * 2));        /* 01: 通用输出 */
    GPIOA->OTYPER  &= ~(1U << 1);              /* 推挽 */
    GPIOA->OSPEEDR &= ~(3U << (1 * 2));
    GPIOA->OSPEEDR |=  (1U << (1 * 2));        /* 中速 */
    GPIOA->PUPDR   &= ~(3U << (1 * 2));

    led_off();
}

void bsp_led_set_mode(LedMode m)
{
    if (m != s_mode) { s_mode = m; s_phase_ms = 0; }
}

void bsp_led_tick_1ms(void)
{
    s_phase_ms++;

    switch (s_mode) {
        case LED_MODE_SLOW_BLINK:
            if (s_phase_ms >= 1000) { s_phase_ms = 0; }
            if (s_phase_ms < 500) led_on(); else led_off();
            break;
        case LED_MODE_FAST_BLINK:
            if (s_phase_ms >= 200) { s_phase_ms = 0; }
            if (s_phase_ms < 100) led_on(); else led_off();
            break;
        case LED_MODE_ON:  led_on();  break;
        case LED_MODE_OFF: led_off(); break;
    }
}
