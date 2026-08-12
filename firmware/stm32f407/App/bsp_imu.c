#include "bsp_imu.h"
#include "bsp_uart.h"
#include <string.h>

/* ============================================================
 * 维特智能 (WIT) JY901S 默认输出协议
 * 帧 11 字节: 0x55 type d0..d7 checksum
 *   type:
 *     0x51 加速度   (acc xyz + temp)
 *     0x52 角速度   (gyro xyz + temp)
 *     0x53 角度     (roll pitch yaw + temp_high_byte? 实际填温度低字节)
 *     0x54 磁场     (忽略)
 *     0x59 四元数   (忽略)
 *
 * 单位:
 *   acc   = (int16) / 32768.0 * 16g       (g)   -> *9.80665 -> m/s^2
 *   gyro  = (int16) / 32768.0 * 2000°/s   (°/s) -> *π/180   -> rad/s
 *   angle = (int16) / 32768.0 * 180°      (°)   保持 °
 *   temp  = (int16) / 100.0               (°C)
 *
 * checksum = sum 前 10 字节 & 0xFF
 * ============================================================ */

static ImuData s_imu;

/* 状态机: 找 0x55, 找 type ∈ {0x50..0x5A}, 然后再读 9 字节 (8 data + 1 cks). */
static uint8_t  s_buf[11];
static uint8_t  s_idx = 0;
static uint8_t  s_angle_frame_seen = 0;
static uint8_t  s_baud_locked = 0;
static uint32_t s_last_valid_angle_ms = 0;
static uint32_t s_last_baud_switch_ms = 0;

#define IMU_BAUD_PROBE_INTERVAL_MS  1500u
#define IMU_LINK_STALE_MS           1000u

static const float G_TO_MS2  = 9.80665f;
static const float DEG_TO_RAD = 0.017453292519943295f;

static int16_t i16le(const uint8_t *p) { return (int16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)); }

static void parse_frame(const uint8_t *f)
{
    /* f[0]=0x55, f[1]=type, f[2..9]=data, f[10]=cks */
    uint8_t sum = 0;
    for (int i = 0; i < 10; ++i) sum = (uint8_t)(sum + f[i]);
    if (sum != f[10]) return;       /* 校验失败丢弃 */

    switch (f[1]) {
        case 0x51: {
            int16_t ax = i16le(&f[2]);
            int16_t ay = i16le(&f[4]);
            int16_t az = i16le(&f[6]);
            int16_t t  = i16le(&f[8]);
            s_imu.accel_x = (float)ax / 32768.0f * 16.0f * G_TO_MS2;
            s_imu.accel_y = (float)ay / 32768.0f * 16.0f * G_TO_MS2;
            s_imu.accel_z = (float)az / 32768.0f * 16.0f * G_TO_MS2;
            s_imu.temp_c  = (float)t  / 100.0f;
            break;
        }
        case 0x52: {
            int16_t gx = i16le(&f[2]);
            int16_t gy = i16le(&f[4]);
            int16_t gz = i16le(&f[6]);
            s_imu.gyro_x = (float)gx / 32768.0f * 2000.0f * DEG_TO_RAD;
            s_imu.gyro_y = (float)gy / 32768.0f * 2000.0f * DEG_TO_RAD;
            s_imu.gyro_z = (float)gz / 32768.0f * 2000.0f * DEG_TO_RAD;
            break;
        }
        case 0x53: {
            int16_t r = i16le(&f[2]);
            int16_t p = i16le(&f[4]);
            int16_t y = i16le(&f[6]);
            s_imu.roll_deg  = (float)r / 32768.0f * 180.0f;
            s_imu.pitch_deg = (float)p / 32768.0f * 180.0f;
            s_imu.yaw_deg   = (float)y / 32768.0f * 180.0f;
            /* last_angle_ms 在 poll() 调用方写, 这里只标 flag */
            s_angle_frame_seen = 1;
            break;
        }
        default:
            /* 0x54 magnetic / 0x59 quaternion / 其他: 不处理 */
            break;
    }
}

static void feed_byte(uint8_t b)
{
    if (s_idx == 0) {
        if (b == 0x55) { s_buf[s_idx++] = b; }
    } else if (s_idx == 1) {
        if (b >= 0x50 && b <= 0x5A) { s_buf[s_idx++] = b; }
        else { s_idx = (b == 0x55) ? 1 : 0; if (s_idx == 1) s_buf[0] = 0x55; }
    } else {
        s_buf[s_idx++] = b;
        if (s_idx == 11) {
            parse_frame(s_buf);
            s_idx = 0;
        }
    }
}

void bsp_imu_init(void)
{
    memset(&s_imu, 0, sizeof(s_imu));
    s_idx = 0;
    s_angle_frame_seen = 0;
    s_baud_locked = 0;
    s_last_valid_angle_ms = 0;
    s_last_baud_switch_ms = 0;
}

void bsp_imu_poll(uint32_t now_ms)
{
    uint8_t buf[64];
    uint16_t n = imu_uart_read(buf, sizeof(buf));
    for (uint16_t i = 0; i < n; ++i) feed_byte(buf[i]);

    /* 角度帧落了时间戳 */
    if (s_angle_frame_seen) {
        s_angle_frame_seen = 0;
        s_last_valid_angle_ms = now_ms;
        s_imu.last_angle_ms = (now_ms == 0u) ? 1u : now_ms;
        s_baud_locked = 1;
    }

    if (s_baud_locked && (uint32_t)(now_ms - s_last_valid_angle_ms) >= IMU_LINK_STALE_MS) {
        s_baud_locked = 0;
        s_last_baud_switch_ms = now_ms;
    }

    if (!s_baud_locked &&
        (uint32_t)(now_ms - s_last_baud_switch_ms) >= IMU_BAUD_PROBE_INTERVAL_MS) {
        uint32_t next_baud = (imu_uart_baud() == IMU_BAUDRATE_FAST)
                           ? IMU_BAUDRATE_FALLBACK : IMU_BAUDRATE_FAST;
        imu_uart_set_baud(next_baud);
        memset(&s_imu, 0, sizeof(s_imu));
        s_idx = 0;
        s_angle_frame_seen = 0;
        s_last_valid_angle_ms = 0;
        s_last_baud_switch_ms = now_ms;
    }
}

const ImuData* bsp_imu_data(void) { return &s_imu; }
