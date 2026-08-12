#include "bsp_uart.h"
#include "bsp_clock.h"
#include "stm32f4xx.h"

/* ============================================================
 * UART 资源 (2026-06-10 按《STM32F407引脚接线.docx》迁移):
 *   USART2: PD5 (TX) / PD6 (RX), AF7, APB1=42MHz   ← 上位机 (X5, CH340 USB-TTL)
 *     RX 走 DMA1_Stream5 Channel 4 (循环模式 ring buffer)
 *   USART3: PD8 (TX) / PD9 (RX), AF7, APB1=42MHz   ← IMU (JY901S 陀螺仪)
 *     RX 走 DMA1_Stream1 Channel 4 (循环模式 ring buffer)
 *
 * 接线 (接线图条目 6/7):
 *   CH340  RXD ← PD5 (USART2_TX),  CH340  TXD → PD6 (USART2_RX)
 *   JY901S RX  ← PD8 (USART3_TX),  JY901S TX  → PD9 (USART3_RX)
 *
 * 旧版 (USART1 PA9/PA10 host + USART2 PA2/PA3 IMU) 已废弃 —
 * PA2/PA3 现在归电机 1 的 TTL 配置口 (接线图条目 1).
 *
 * RX 用 DMA ring + 软件读指针. 不依赖 IDLE 中断 (但加上更稳).
 * TX 直接寄存器轮询 (帧很小, 不值得 DMA).
 * ============================================================ */

static volatile uint8_t s_host_rx_buf[HOST_RX_BUF_SIZE];
static volatile uint16_t s_host_rx_rd = 0;       /* 软件读指针 */

static volatile uint8_t s_imu_rx_buf[IMU_RX_BUF_SIZE];
static volatile uint16_t s_imu_rx_rd = 0;
static uint32_t s_imu_baudrate = IMU_BAUDRATE_FAST;

/* ============================================================
 * GPIO / 时钟初始化
 * ============================================================ */
static void gpio_config_for_uart(void)
{
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIODEN;

    /* PD5 (USART2_TX), PD6 (USART2_RX): AF7 */
    GPIOD->MODER   &= ~((3U << (5 * 2)) | (3U << (6 * 2)));
    GPIOD->MODER   |=  ((2U << (5 * 2)) | (2U << (6 * 2)));   /* 10: AF */
    GPIOD->OSPEEDR |=  ((2U << (5 * 2)) | (2U << (6 * 2)));   /* fast */
    GPIOD->PUPDR   &= ~((3U << (5 * 2)) | (3U << (6 * 2)));
    GPIOD->PUPDR   |=  ((1U << (6 * 2)));                     /* RX 上拉 */
    GPIOD->AFR[0]  &= ~((0xFU << (5 * 4)) | (0xFU << (6 * 4)));
    GPIOD->AFR[0]  |=  ((7U   << (5 * 4)) | (7U   << (6 * 4)));

    /* PD8 (USART3_TX), PD9 (USART3_RX): AF7 */
    GPIOD->MODER   &= ~((3U << (8 * 2)) | (3U << (9 * 2)));
    GPIOD->MODER   |=  ((2U << (8 * 2)) | (2U << (9 * 2)));
    GPIOD->OSPEEDR |=  ((2U << (8 * 2)) | (2U << (9 * 2)));
    GPIOD->PUPDR   &= ~((3U << (8 * 2)) | (3U << (9 * 2)));
    GPIOD->PUPDR   |=  ((1U << (9 * 2)));                     /* RX 上拉 */
    GPIOD->AFR[1]  &= ~((0xFU << ((8 - 8) * 4)) | (0xFU << ((9 - 8) * 4)));
    GPIOD->AFR[1]  |=  ((7U   << ((8 - 8) * 4)) | (7U   << ((9 - 8) * 4)));
}

/* ============================================================
 * USART2: 上位机 (115200, DMA1_Stream5 RX)
 * ============================================================ */
static void usart2_host_init(void)
{
    RCC->APB1ENR |= RCC_APB1ENR_USART2EN;
    RCC->AHB1ENR |= RCC_AHB1ENR_DMA1EN;

    USART2->CR1 = 0;
    USART2->CR2 = 0;
    USART2->CR3 = USART_CR3_DMAR;                    /* RX 走 DMA */

    /* 波特率: APB1=42MHz, OVER8=0, BRR = 42M / 115200 = 364.58 -> 0x16C */
    USART2->BRR = (uint32_t)(APB1_HZ / HOST_BAUDRATE);

    /* DMA1_Stream5 channel 4 = USART2_RX */
    DMA1_Stream5->CR = 0;
    while (DMA1_Stream5->CR & DMA_SxCR_EN) { /* 等关闭 */ }
    DMA1->HIFCR = 0x0F40;                            /* 清 stream5 全部 flag */
    DMA1_Stream5->PAR  = (uint32_t)&USART2->DR;
    DMA1_Stream5->M0AR = (uint32_t)s_host_rx_buf;
    DMA1_Stream5->NDTR = HOST_RX_BUF_SIZE;
    DMA1_Stream5->CR =
        (4U << 25) |                                 /* CHSEL = 4 */
        DMA_SxCR_MINC |                              /* memory incr */
        DMA_SxCR_CIRC |                              /* 循环模式 */
        (0U << 6);                                   /* 方向 P->M */
    DMA1_Stream5->CR |= DMA_SxCR_EN;

    USART2->CR1 = USART_CR1_RE | USART_CR1_TE | USART_CR1_UE;
}

/* ============================================================
 * USART3: IMU (115200, DMA1_Stream1 RX)
 * ============================================================ */
static uint8_t usart3_imu_dma_start(uint32_t baudrate)
{
    uint32_t disable_guard = 100000u;

    USART3->CR1 = 0;
    USART3->CR2 = 0;
    USART3->CR3 = 0;

    /* APB1=42MHz, BRR = 42M / 115200 = 364.58 -> 0x16C */
    USART3->BRR = (uint32_t)(APB1_HZ / baudrate);

    /* DMA1_Stream1 channel 4 = USART3_RX */
    DMA1_Stream1->CR &= ~DMA_SxCR_EN;
    while ((DMA1_Stream1->CR & DMA_SxCR_EN) && disable_guard > 0u) --disable_guard;
    if (DMA1_Stream1->CR & DMA_SxCR_EN) return 0u;
    DMA1_Stream1->CR = 0;
    DMA1->LIFCR = 0x0F40;                            /* 清 stream1 全部 flag (bit6-11) */
    s_imu_rx_rd = 0;
    for (uint16_t i = 0; i < IMU_RX_BUF_SIZE; ++i) s_imu_rx_buf[i] = 0;
    DMA1_Stream1->PAR  = (uint32_t)&USART3->DR;
    DMA1_Stream1->M0AR = (uint32_t)s_imu_rx_buf;
    DMA1_Stream1->NDTR = IMU_RX_BUF_SIZE;
    DMA1_Stream1->CR =
        (4U << 25) |
        DMA_SxCR_MINC |
        DMA_SxCR_CIRC |
        (0U << 6);
    DMA1_Stream1->CR |= DMA_SxCR_EN;

    USART3->CR1 = USART_CR1_RE | USART_CR1_UE;       /* IMU 只接收, 不发 */
    USART3->CR3 = USART_CR3_DMAR;
    s_imu_baudrate = baudrate;
    return 1u;
}

static void usart3_imu_init(void)
{
    RCC->APB1ENR |= RCC_APB1ENR_USART3EN;
    RCC->AHB1ENR |= RCC_AHB1ENR_DMA1EN;
    (void)usart3_imu_dma_start(IMU_BAUDRATE_FAST);
}

void bsp_uart_init(void)
{
    gpio_config_for_uart();
    usart2_host_init();
    usart3_imu_init();
}

/* ============================================================
 * TX (USART2 host) — 轮询, 帧只有几十字节, 不值 DMA
 * ============================================================ */
void host_uart_send(const uint8_t *data, uint16_t len)
{
    for (uint16_t i = 0; i < len; ++i) {
        while (!(USART2->SR & USART_SR_TXE)) { }
        USART2->DR = data[i];
    }
    while (!(USART2->SR & USART_SR_TC)) { }
}

/* ============================================================
 * RX 读 — 从环形缓冲读出新字节
 * DMA 在 buffer 里循环写, 我们维护软件 rd 指针, 用 (size - NDTR) 算出硬件 wr 指针.
 * ============================================================ */
static uint16_t read_ring(volatile uint8_t *buf, uint16_t size, volatile uint16_t *rd,
                          uint16_t hw_wr, uint8_t *out, uint16_t max)
{
    uint16_t avail;
    if (hw_wr >= *rd) avail = (uint16_t)(hw_wr - *rd);
    else              avail = (uint16_t)(size - *rd + hw_wr);

    if (avail > max) avail = max;

    for (uint16_t i = 0; i < avail; ++i) {
        out[i] = buf[*rd];
        *rd = (uint16_t)((*rd + 1) % size);
    }
    return avail;
}

uint16_t host_uart_read(uint8_t *out, uint16_t max)
{
    uint16_t hw_wr = (uint16_t)(HOST_RX_BUF_SIZE - DMA1_Stream5->NDTR);
    return read_ring(s_host_rx_buf, HOST_RX_BUF_SIZE, &s_host_rx_rd, hw_wr, out, max);
}

uint16_t imu_uart_read(uint8_t *out, uint16_t max)
{
    uint16_t hw_wr = (uint16_t)(IMU_RX_BUF_SIZE - DMA1_Stream1->NDTR);
    return read_ring(s_imu_rx_buf, IMU_RX_BUF_SIZE, &s_imu_rx_rd, hw_wr, out, max);
}

void imu_uart_set_baud(uint32_t baudrate)
{
    if (baudrate != IMU_BAUDRATE_FAST && baudrate != IMU_BAUDRATE_FALLBACK) return;
    if (baudrate == s_imu_baudrate) return;
    (void)usart3_imu_dma_start(baudrate);
}

uint32_t imu_uart_baud(void)
{
    return s_imu_baudrate;
}
