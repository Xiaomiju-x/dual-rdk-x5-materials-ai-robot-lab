#include "bsp_motor.h"
#include "bsp_clock.h"
#include "stm32f4xx.h"

/* ============================================================
 * TIM1 (APB2 timclk 168MHz) 4 通道 **OC Toggle 模式**, 每通道独立频率.
 *   PSC = 168-1  ->  1MHz tick (1us 分辨率)
 *   ARR = 0xFFFF 自由跑, 不当周期用
 *   CCRx 每次比较命中后 += half_interval (半周期), 输出 toggle
 *   → 2 次 toggle = 1 个完整脉冲, 4 路频率彼此独立 (Nav2 弧线左右不同速 OK)
 *
 * 计步: TIM1_CC_IRQHandler 里, 输出翻到高电平那一拍 (= X57S 上升沿吃步)
 * step_count += cur_dir. 计的是真实发出的脉冲沿, 不是估算.
 *
 * ── 2026-06-11 重写原因 (老 PWM Mode1 + 共享 ARR 两条命门) ──
 * 1. 老 set_speed 每次无条件 TIM1->EGR=UG: main loop 每圈都调 →
 *    UG 复位 CNT → CNT 永远到不了 CCR → PWM 冻在高电平 **没有脉冲**;
 *    同时每次 UG 触发 update 中断被当成"走了一步" → odom 步数 ×1790
 *    (2026-06-11 X5 实测: 命令 0.05 m/s, /odom 报 78 m/s).
 * 2. 共享 ARR: 4 路只能同频, Nav2 弧线 (v±wL/2 左右异速) 会被最后
 *    写入的轮速覆盖成直线.
 * 防复发: set_speed 带 req_pps 缓存, 同值直接 return (主循环每圈调也无害).
 * ============================================================ */

typedef struct {
    GPIO_TypeDef *dir_port;  uint8_t dir_pin;
    GPIO_TypeDef *en_port;   uint8_t en_pin;
    int8_t   invert;             /* MOTOR_INVERT_Mx */
    volatile uint8_t  enabled;   /* 1 = 正在输出脉冲; ISR 用 */
    volatile int8_t   cur_dir;   /* +1 / -1 (软件方向, 计步用) */
    volatile uint8_t  phase;     /* 0 = 输出当前为低; 翻高那拍计步 */
    volatile uint32_t half_iv;   /* 半周期 tick 数 = 1e6 / (2*pps) */
    volatile int32_t  step_count;
    int32_t  last_taken_count;
    int32_t  req_pps;            /* 上次请求值缓存 (含符号), 同值早退 */
} motor_t;

static motor_t s_motors[MOTOR_COUNT];

/* ---------- GPIO 工具 ---------- */
static void gpio_af_high_speed(GPIO_TypeDef *port, uint8_t pin, uint8_t af)
{
    port->MODER   &= ~(3U << (pin * 2));
    port->MODER   |=  (2U << (pin * 2));            /* 10: AF */
    port->OSPEEDR |=  (3U << (pin * 2));            /* very high */
    port->OTYPER  &= ~(1U <<  pin);                 /* 推挽 */
    port->PUPDR   &= ~(3U << (pin * 2));
    if (pin < 8) {
        port->AFR[0] &= ~(0xFU << (pin * 4));
        port->AFR[0] |=  ((uint32_t)af << (pin * 4));
    } else {
        port->AFR[1] &= ~(0xFU << ((pin - 8) * 4));
        port->AFR[1] |=  ((uint32_t)af << ((pin - 8) * 4));
    }
}

static void gpio_out_low(GPIO_TypeDef *port, uint8_t pin)
{
    port->MODER   &= ~(3U << (pin * 2));
    port->MODER   |=  (1U << (pin * 2));            /* 通用输出 */
    port->OTYPER  &= ~(1U <<  pin);
    port->OSPEEDR |=  (1U << (pin * 2));
    port->PUPDR   &= ~(3U << (pin * 2));
    port->BSRR     = (uint32_t)(1U << pin) << 16;   /* 初始低 */
}

static inline void gpio_write(GPIO_TypeDef *port, uint8_t pin, uint8_t hi)
{
    if (hi) port->BSRR =  (uint32_t)(1U << pin);
    else    port->BSRR =  (uint32_t)(1U << pin) << 16;
}

/* ---------- 通道寄存器访问 ---------- */
static __IO uint32_t *ccr_of(uint8_t idx)
{
    switch (idx) {
        case 0: return &TIM1->CCR1;
        case 1: return &TIM1->CCR2;
        case 2: return &TIM1->CCR3;
        case 3: return &TIM1->CCR4;
    }
    return &TIM1->CCR1;
}

/* OCxM 写 3 bit 模式: CH1/CH3 在 CCMRx 低半, CH2/CH4 在高半.
 * 011 = toggle (跑), 100 = force inactive (停, 输出强制低) */
static void oc_mode(uint8_t idx, uint32_t mode3)
{
    __IO uint32_t *ccmr = (idx < 2) ? &TIM1->CCMR1 : &TIM1->CCMR2;
    uint8_t hi = (idx & 1U);                        /* CH2/CH4 用高半字 */
    uint32_t shift = hi ? 12U : 4U;
    uint32_t v = *ccmr;
    v &= ~(7U << shift);
    v |=  (mode3 << shift);
    *ccmr = v;
}

static const uint32_t CC_IE [4] = { TIM_DIER_CC1IE, TIM_DIER_CC2IE, TIM_DIER_CC3IE, TIM_DIER_CC4IE };
static const uint32_t CC_IF [4] = { TIM_SR_CC1IF,   TIM_SR_CC2IF,   TIM_SR_CC3IF,   TIM_SR_CC4IF   };

/* ---------- TIM1 初始化: 自由跑 + 4 路 toggle 待命 ---------- */
static void tim1_init(void)
{
    RCC->APB2ENR |= RCC_APB2ENR_TIM1EN;

    TIM1->CR1   = 0;
    TIM1->CR2   = 0;
    TIM1->PSC   = (uint16_t)(APB2_TIMCLK / 1000000UL - 1);    /* 167 → 1MHz */
    TIM1->ARR   = 0xFFFF;                                      /* 自由跑 */

    TIM1->CCR1  = 0;
    TIM1->CCR2  = 0;
    TIM1->CCR3  = 0;
    TIM1->CCR4  = 0;

    /* 4 路初始 force-inactive (输出低), OCxPE **不开** (toggle 调度要直写 CCR) */
    TIM1->CCMR1 = (4U << 4) | (4U << 12);
    TIM1->CCMR2 = (4U << 4) | (4U << 12);

    /* CCER: 4 路输出使能, 极性默认高有效 */
    TIM1->CCER  = TIM_CCER_CC1E | TIM_CCER_CC2E | TIM_CCER_CC3E | TIM_CCER_CC4E;

    /* 高级定时器必须 MOE */
    TIM1->BDTR |= TIM_BDTR_MOE;

    /* PSC 立刻生效 (仅 init 这一次 UG; set_speed 永不再碰 EGR) */
    TIM1->EGR   = TIM_EGR_UG;
    TIM1->SR    = 0;

    /* CC 中断走 TIM1_CC_IRQHandler; DIER 的 CCxIE 由 set_speed 按通道开关 */
    TIM1->DIER  = 0;
    NVIC_SetPriority(TIM1_CC_IRQn, 1);
    NVIC_EnableIRQ(TIM1_CC_IRQn);

    TIM1->CR1  |= TIM_CR1_CEN;
}

/* ---------- 公共 ---------- */
void bsp_motor_init(void)
{
    /* GPIO 时钟: PE (PUL/DIR/EN 全在 PE) */
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOEEN;

    /* PUL (TIM1_CHx AF1): PE9, PE11, PE13, PE14 */
    gpio_af_high_speed(GPIOE, 9,  1);
    gpio_af_high_speed(GPIOE, 11, 1);
    gpio_af_high_speed(GPIOE, 13, 1);
    gpio_af_high_speed(GPIOE, 14, 1);

    /* DIR: PE1, PE3, PE5, PE7 */
    gpio_out_low(GPIOE, 1);
    gpio_out_low(GPIOE, 3);
    gpio_out_low(GPIOE, 5);
    gpio_out_low(GPIOE, 7);

    /* EN: PE2, PE4, PE6, PE8 (默认拉到 "失能" 电平, 等业务层显式 enable) */
    gpio_out_low(GPIOE, 2);
    gpio_out_low(GPIOE, 4);
    gpio_out_low(GPIOE, 6);
    gpio_out_low(GPIOE, 8);

    /* 软件状态 (req_pps 初值用一个不可能的值, 保证第一次 set_speed 不被缓存吞掉) */
    s_motors[MOTOR_M1] = (motor_t){ GPIOE, 1, GPIOE, 2, MOTOR_INVERT_M1, 0, 1, 0, 0, 0, 0, INT32_MIN };
    s_motors[MOTOR_M2] = (motor_t){ GPIOE, 3, GPIOE, 4, MOTOR_INVERT_M2, 0, 1, 0, 0, 0, 0, INT32_MIN };
    s_motors[MOTOR_M3] = (motor_t){ GPIOE, 5, GPIOE, 6, MOTOR_INVERT_M3, 0, 1, 0, 0, 0, 0, INT32_MIN };
    s_motors[MOTOR_M4] = (motor_t){ GPIOE, 7, GPIOE, 8, MOTOR_INVERT_M4, 0, 1, 0, 0, 0, 0, INT32_MIN };

    /* EN 默认: 全失能 (避免上电瞬间乱动) */
    bsp_motor_disable_all();

    tim1_init();
}

/* ---------- EN ---------- */
void bsp_motor_enable(uint8_t idx)
{
    if (idx >= MOTOR_COUNT) return;
    motor_t *m = &s_motors[idx];
#if EN_ACTIVE_LOW
    gpio_write(m->en_port, m->en_pin, 0);     /* LOW = 使能 */
#else
    gpio_write(m->en_port, m->en_pin, 1);
#endif
}

void bsp_motor_disable(uint8_t idx)
{
    if (idx >= MOTOR_COUNT) return;
    motor_t *m = &s_motors[idx];
#if EN_ACTIVE_LOW
    gpio_write(m->en_port, m->en_pin, 1);
#else
    gpio_write(m->en_port, m->en_pin, 0);
#endif
}

void bsp_motor_enable_all (void) { for (int i = 0; i < MOTOR_COUNT; ++i) bsp_motor_enable (i); }
void bsp_motor_disable_all(void) { for (int i = 0; i < MOTOR_COUNT; ++i) bsp_motor_disable(i); }

/* ---------- 单通道停 (输出强制低 + 关中断) ---------- */
static void channel_stop(uint8_t idx)
{
    motor_t *m = &s_motors[idx];
    TIM1->DIER &= ~CC_IE[idx];
    oc_mode(idx, 4U);                 /* force inactive → 输出低 */
    m->enabled = 0;
    m->phase   = 0;
}

/* ---------- 设速 (pulse/sec 带符号) ---------- */
void bsp_motor_set_speed_pps(uint8_t idx, int32_t pps)
{
    if (idx >= MOTOR_COUNT) return;
    motor_t *m = &s_motors[idx];

    if (pps >  MOTOR_MAX_PPS) pps =  MOTOR_MAX_PPS;
    if (pps < -MOTOR_MAX_PPS) pps = -MOTOR_MAX_PPS;

    /* 缓存: 主循环每圈都调, 同值直接走人 (老实现在这里每次 EGR=UG 闯的祸) */
    if (pps == m->req_pps) return;
    m->req_pps = pps;

    /* 方向 = 软件方向 ⊕ 安装反向 */
    int dir_hi = (pps >= 0) ? 1 : 0;
    if (m->invert) dir_hi = !dir_hi;
    gpio_write(m->dir_port, m->dir_pin, (uint8_t)dir_hi);

    m->cur_dir = (pps >= 0) ? 1 : -1;

    int32_t abs_pps = (pps >= 0) ? pps : -pps;

    if (abs_pps < MOTOR_MIN_PPS) {
        channel_stop(idx);
        return;
    }

    uint32_t half_iv = 1000000UL / (2UL * (uint32_t)abs_pps);   /* 半周期 tick */
    if (half_iv < 8U)       half_iv = 8U;                       /* ISR 兜底下限 */
    if (half_iv > 0xFFFFU)  half_iv = 0xFFFFU;
    m->half_iv = half_iv;

    if (!m->enabled) {
        /* 从停到跑: 排第一次比较点, 清残留 flag, 开中断 */
        __disable_irq();
        *ccr_of(idx) = (uint16_t)(TIM1->CNT + half_iv);
        TIM1->SR     = ~CC_IF[idx];                /* rc_w0: 只清本通道 */
        oc_mode(idx, 3U);                          /* toggle */
        TIM1->DIER  |= CC_IE[idx];
        m->phase     = 0;
        m->enabled   = 1;
        __enable_irq();
    }
    /* 在跑: ISR 下一拍自动用新 half_iv, 无需任何寄存器操作 */
}

void bsp_motor_emergency_stop(void)
{
    for (uint8_t i = 0; i < MOTOR_COUNT; ++i) {
        channel_stop(i);
        s_motors[i].cur_dir = 0;
        s_motors[i].req_pps = INT32_MIN;   /* 急停后第一次 set_speed 必须生效 */
    }
    bsp_motor_disable_all();
}

/* ---------- 步数 ---------- */
int32_t bsp_motor_take_step_delta(uint8_t idx)
{
    if (idx >= MOTOR_COUNT) return 0;
    motor_t *m = &s_motors[idx];

    __disable_irq();
    int32_t cur = m->step_count;
    __enable_irq();

    int32_t d = cur - m->last_taken_count;
    m->last_taken_count = cur;
    return d;
}

/* ============================================================
 * TIM1 CC ISR — 每个通道命中 = 输出翻转一次.
 * 翻到高那拍 (phase 0→1) 是 X57S 的上升沿 → 真实走了一步.
 * 重排下一比较点: CCRx += half_iv (16-bit 自然回绕).
 * ============================================================ */
void TIM1_CC_IRQHandler(void)
{
    uint32_t sr = TIM1->SR;
    for (uint8_t i = 0; i < MOTOR_COUNT; ++i) {
        if (!(sr & CC_IF[i])) continue;
        TIM1->SR = ~CC_IF[i];                      /* rc_w0 清本通道 */

        motor_t *m = &s_motors[i];
        if (!m->enabled) continue;

        if (m->phase == 0) {                       /* 低→高: 计 1 步 */
            m->phase = 1;
            m->step_count += m->cur_dir;
        } else {
            m->phase = 0;
        }

        /* 下一比较点; ISR 迟到落后于 CNT 时重新从当前点排, 防 65ms 全圈空等 */
        __IO uint32_t *ccr = ccr_of(i);
        uint16_t next = (uint16_t)(*ccr + m->half_iv);
        if ((int16_t)(next - (uint16_t)TIM1->CNT) <= 0)
            next = (uint16_t)(TIM1->CNT + m->half_iv);
        *ccr = next;
    }
}
