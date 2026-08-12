#include "odom.h"
#include "bsp_motor.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* 米/步 = 周长 / (步/圈) */
#define METERS_PER_STEP   ((2.0f * M_PI * WHEEL_RADIUS_M) / (float)STEPS_PER_REV)

static float s_x_m   = 0.0f;
static float s_y_m   = 0.0f;
static float s_yaw   = 0.0f;     /* rad, 内部用 */
static float s_vx    = 0.0f;
static float s_wz    = 0.0f;

void odom_cmdvel_to_wheels(float linear_v, float angular_w, int32_t pps_out[4])
{
    /* 差速:
     *   v_left  = v - w * L/2
     *   v_right = v + w * L/2
     * pps = v_wheel / METERS_PER_STEP
     */
    float v_left  = linear_v - angular_w * (WHEEL_BASE_M * 0.5f);
    float v_right = linear_v + angular_w * (WHEEL_BASE_M * 0.5f);

    float pps_l = v_left  / METERS_PER_STEP;
    float pps_r = v_right / METERS_PER_STEP;

#if CRUISE_PPS > 0
    /* 固定巡航: 把主轮 (绝对值大的那侧) 归一化到 CRUISE_PPS, 两轮等比缩放.
     * → 直走: 两轮都 CRUISE_PPS (恒速); 纯转: ±CRUISE_PPS (原地恒速转);
     *   边走边转: 外轮 CRUISE_PPS, 内轮按比例. cmd_vel 只决定方向/转弯半径, 不决定快慢.
     * 阈值 1.0: 防 cmd_vel 噪声 (<0.0006 m/s) 被放大成全速; 真零指令 → 停 (走 watchdog). */
    {
        float mag = fabsf(pps_l) > fabsf(pps_r) ? fabsf(pps_l) : fabsf(pps_r);
        if (mag > 1.0f) {
            float scale = (float)CRUISE_PPS / mag;
            pps_l *= scale;
            pps_r *= scale;
        }
    }
#endif

    /* M1=前左, M2=前右, M3=后左, M4=后右 */
    pps_out[0] = (int32_t)pps_l;  /* M1 左 */
    pps_out[1] = (int32_t)pps_r;  /* M2 右 */
    pps_out[2] = (int32_t)pps_l;  /* M3 左 */
    pps_out[3] = (int32_t)pps_r;  /* M4 右 */
}

void odom_update(int32_t d_m1, int32_t d_m2, int32_t d_m3, int32_t d_m4, float dt_s,
                 uint8_t use_imu_yaw, uint8_t imu_yaw_valid, float imu_yaw_rad)
{
    if (dt_s <= 0.0f) return;

    /* 左/右轮平均 step delta */
    float d_left  = ((float)d_m1 + (float)d_m3) * 0.5f;
    float d_right = ((float)d_m2 + (float)d_m4) * 0.5f;

    float ds_left  = d_left  * METERS_PER_STEP;
    float ds_right = d_right * METERS_PER_STEP;

    float ds = (ds_left + ds_right) * 0.5f;
    float dyaw_step = (ds_right - ds_left) / WHEEL_BASE_M;     /* 推算的 dyaw */

    /* yaw 来源选择 */
    if (use_imu_yaw && imu_yaw_valid) {
        s_yaw = imu_yaw_rad;
    } else {
        s_yaw += dyaw_step;
        /* 归一化 [-π, π] */
        while (s_yaw >  M_PI) s_yaw -= 2.0f * M_PI;
        while (s_yaw < -M_PI) s_yaw += 2.0f * M_PI;
    }

    /* 积分到 x/y (用 dt 内中点 yaw 减误差) */
    float mid_yaw = s_yaw - 0.5f * dyaw_step;
    s_x_m += ds * cosf(mid_yaw);
    s_y_m += ds * sinf(mid_yaw);

    /* 瞬时速度 (用于上行 telemetry) */
    s_vx = ds        / dt_s;
    s_wz = dyaw_step / dt_s;
}

void odom_get(float *x_m, float *y_m, float *vx_mps, float *wz_radps, float *yaw_deg)
{
    if (x_m)      *x_m      = s_x_m;
    if (y_m)      *y_m      = s_y_m;
    if (vx_mps)   *vx_mps   = s_vx;
    if (wz_radps) *wz_radps = s_wz;
    if (yaw_deg)  *yaw_deg  = s_yaw * 180.0f / M_PI;
}

void odom_zero_velocity(void)
{
    s_vx = 0.0f;
    s_wz = 0.0f;
}
