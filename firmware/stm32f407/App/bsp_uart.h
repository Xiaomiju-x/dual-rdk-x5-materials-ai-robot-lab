#ifndef BSP_UART_H
#define BSP_UART_H

#include <stdint.h>
#include <stddef.h>

#define HOST_BAUDRATE          115200u
#define IMU_BAUDRATE_FAST      115200u
#define IMU_BAUDRATE_FALLBACK    9600u

#define HOST_RX_BUF_SIZE  512u
#define IMU_RX_BUF_SIZE   256u

void   bsp_uart_init(void);

/* USART2 @ PD5/PD6 (host, CH340→X5)  --- 阻塞 TX, DMA ring RX */
void   host_uart_send(const uint8_t *data, uint16_t len);
/* 返回从环形缓冲读出的字节数, 不阻塞. out 至少 max 字节. */
uint16_t host_uart_read(uint8_t *out, uint16_t max);

/* USART3 @ PD8/PD9 (IMU JY901S)  --- 只读, DMA ring */
uint16_t imu_uart_read(uint8_t *out, uint16_t max);
void     imu_uart_set_baud(uint32_t baudrate);
uint32_t imu_uart_baud(void);

#endif
