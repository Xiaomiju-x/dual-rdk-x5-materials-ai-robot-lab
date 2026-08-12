# 烧结炉显示屏 OCR 读数
# 来源: my_robot_perception/furnace_ocr_node 处理 pt_camera 拉到的画面
# 频率: 1 Hz (炉子温度变化慢, 不需要更高)

std_msgs/Header header

# === 读数 ===
float32 pv         # Process Value: 实际温度 (°C)
float32 sv         # Set Value:     设定温度 (°C)
float32 mv         # Manipulated Value: 加热输出百分比 (0~100)

# === 置信度 ===
float32 pv_confidence    # 七段 OCR 匹配置信度 (0~1)
float32 sv_confidence
float32 mv_confidence

# === 状态标志 ===
bool power_indicator_on  # 红色 Power Indicator 灯是否亮 (HSV 检测)
bool screen_visible      # 显示屏整体是否亮 (整体亮度阈值)

# === 火焰 / 烟雾 (I6) ===
bool fire_detected         # HSV 红橙色像素聚集 + 运动差分超阈值
float32 fire_confidence    # 0~1
bool smoke_detected        # 灰色像素聚类 + 运动差分
float32 smoke_confidence   # 0~1

# === 复核标记 ===
# 当 OCR 置信度 < 阈值 (e.g. 0.7) 时, 由 alert_dispatcher 上传 Qwen-VL 复核
bool needs_vl_recheck

# === 原图 base64 (可选, 复核时用) ===
string snapshot_b64      # JPEG base64, 仅 needs_vl_recheck=true 或 fire_detected/smoke_detected 时填充
