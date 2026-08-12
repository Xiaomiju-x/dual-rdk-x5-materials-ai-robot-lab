# 升降台状态
# 来源: my_robot_drivers/serial_f407_node (从 STM32F407 上行解析)
# 频率: 50 Hz

std_msgs/Header header

float32 height_m              # 当前高度 (米); 当前 TEST_MODE=0 为开环步数换算, 不是编码器实测
float32 target_height_m       # 目标高度 (米)
float32 velocity_mps          # 当前升降速度 (m/s, 正向上)

bool home_switch_triggered    # 下限位开关按下; 当前无开关时固定 false
bool top_switch_triggered     # 上限位开关按下; 当前无开关时固定 false
bool homed                    # 已 home; 当前无开关时仅表示固件开环零点已设定
bool moving                   # 正在移动
bool electromagnet_on         # 电磁铁通电状态
