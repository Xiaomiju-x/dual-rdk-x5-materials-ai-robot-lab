# 异常报警消息
# 来源: my_robot_agents/furnace_monitor_agent (烧结炉相关) 或其他 Agent

std_msgs/Header header

# === 异常类别 (跟 ADR-EB-5 的 I1+I2+I4+I6 对齐) ===
uint8 SOURCE_TEMPERATURE_OUT_OF_RANGE = 1   # I1: PV 超 [-10, 1600]°C
uint8 SOURCE_PV_SV_DEVIATION          = 2   # I2: |PV-SV| > 100°C 持续 30s
uint8 SOURCE_POWER_LIGHT_OFF          = 3   # I4: 红色 Power Indicator 灭了
uint8 SOURCE_FIRE_OR_SMOKE            = 4   # I6: 烟雾火焰
uint8 SOURCE_OTHER                    = 99  # 其他 (留扩展)
uint8 source

# === 严重等级 ===
uint8 LEVEL_INFO     = 1   # 信息, 仅记录
uint8 LEVEL_WARNING  = 2   # 警告, 语音播报
uint8 LEVEL_CRITICAL = 3   # 严重, 语音 + 邮件 + 微信全开
uint8 level

# === 推荐通知通道 (位掩码: 1=TTS 2=Email 4=WeChat 8=ROS2log) ===
uint8 channels

# === 文字描述 ===
string title          # 标题, 60 字内, e.g. "烧结炉 PV 偏差过大"
string description    # 详细描述, 含 OCR 数值, 多行可

# === 关联数据 ===
string snapshot_b64   # 触发现场截图 (PNG base64), 用于邮件附件
string ros2_topic     # 上游 topic 名, 便于追溯
