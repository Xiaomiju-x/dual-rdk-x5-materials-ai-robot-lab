#include "bsp_systick.h"
#include "bsp_clock.h"
#include "stm32f4xx.h"

static volatile uint32_t s_tick_ms = 0;

void bsp_systick_init(void)
{
    SysTick_Config(SYSCLK_HZ / 1000UL);     /* 1ms tick */
    NVIC_SetPriority(SysTick_IRQn, 0);      /* 最高优先级 */
}

uint32_t millis(void) { return s_tick_ms; }

void delay_ms(uint32_t ms)
{
    uint32_t t0 = s_tick_ms;
    while ((s_tick_ms - t0) < ms) { __NOP(); }
}

void SysTick_Handler(void)
{
    s_tick_ms++;
}
