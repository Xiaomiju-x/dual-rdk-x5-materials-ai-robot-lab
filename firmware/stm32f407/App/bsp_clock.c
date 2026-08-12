#include "bsp_clock.h"

/* HSE 8MHz -> PLL -> SYSCLK 168MHz
 * 核心板焊死 8MHz 晶振, 不要被 system_stm32f4xx.c 默认 25MHz 误导.
 * PLL_M=8, PLL_N=336, PLL_P=2, PLL_Q=7  =>  VCO=336MHz, SYSCLK=168MHz, USB=48MHz
 */
void bsp_clock_init(void)
{
    /* 1. 打开 HSE, 等就绪 */
    RCC->CR |= RCC_CR_HSEON;
    {
        volatile uint32_t cnt = 0;
        while (!(RCC->CR & RCC_CR_HSERDY)) {
            if (++cnt > 0x100000UL) return;  /* HSE 起不来直接放弃 (会跑在 HSI 16MHz, 系统能起但定时器差 10×) */
        }
    }

    /* 2. 电源接口时钟, 切到 Scale 1 (>=144MHz 必须) */
    RCC->APB1ENR |= RCC_APB1ENR_PWREN;
    PWR->CR |= PWR_CR_VOS;

    /* 3. AHB/APB 分频: HCLK = SYSCLK/1, APB1 = HCLK/4 = 42MHz, APB2 = HCLK/2 = 84MHz */
    RCC->CFGR &= ~(RCC_CFGR_HPRE | RCC_CFGR_PPRE1 | RCC_CFGR_PPRE2);
    RCC->CFGR |= RCC_CFGR_HPRE_DIV1 | RCC_CFGR_PPRE1_DIV4 | RCC_CFGR_PPRE2_DIV2;

    /* 4. PLL: src=HSE, M=8, N=336, P=2, Q=7  ->  VCO=8/8*336=336, SYSCLK=336/2=168, USB=336/7=48 */
    RCC->PLLCFGR =
        (8UL   <<  0) |                       /* PLLM */
        (336UL <<  6) |                       /* PLLN */
        (0UL   << 16) |                       /* PLLP=2 (00) */
        (RCC_PLLCFGR_PLLSRC_HSE) |
        (7UL   << 24);                        /* PLLQ */

    RCC->CR |= RCC_CR_PLLON;
    while (!(RCC->CR & RCC_CR_PLLRDY)) { /* spin */ }

    /* 5. Flash latency: @168MHz 必须 5WS, 打开预取 + I-cache + D-cache */
    FLASH->ACR = FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN | FLASH_ACR_LATENCY_5WS;

    /* 6. 切到 PLL */
    RCC->CFGR &= ~RCC_CFGR_SW;
    RCC->CFGR |=  RCC_CFGR_SW_PLL;
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL) { /* spin */ }

    /* 7. 让 CMSIS SystemCoreClock 全局变量也对 */
    SystemCoreClock = SYSCLK_HZ;
}
