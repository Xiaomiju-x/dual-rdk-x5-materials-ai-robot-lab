#ifndef BSP_SYSTICK_H
#define BSP_SYSTICK_H

#include <stdint.h>

void     bsp_systick_init(void);
uint32_t millis(void);
void     delay_ms(uint32_t ms);

#endif
