#ifndef BSP_CLOCK_H
#define BSP_CLOCK_H

#include "stm32f4xx.h"

#define SYSCLK_HZ    168000000UL
#define HCLK_HZ      168000000UL
#define APB1_HZ       42000000UL
#define APB2_HZ       84000000UL
#define APB1_TIMCLK   84000000UL
#define APB2_TIMCLK  168000000UL

void bsp_clock_init(void);

#endif
