#ifndef BSP_LED_H
#define BSP_LED_H

#include <stdint.h>

typedef enum {
    LED_MODE_SLOW_BLINK = 0,   /* 1Hz, 待机 */
    LED_MODE_FAST_BLINK = 1,   /* 5Hz, 正常运行 (有心跳) */
    LED_MODE_ON         = 2,   /* 常亮, 急停 / 心跳丢 */
    LED_MODE_OFF        = 3
} LedMode;

void bsp_led_init(void);
void bsp_led_set_mode(LedMode m);
void bsp_led_tick_1ms(void);

#endif
