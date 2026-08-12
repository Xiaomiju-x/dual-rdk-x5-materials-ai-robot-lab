# 取料目标 (200W USB 摄像头检测的瓶顶铁片位姿)
# 来源: my_robot_perception/pickup_target_node
# 频率: 视觉伺服时 5-10 Hz

std_msgs/Header header

# === 检测来源 ===
uint8 SOURCE_APRILTAG = 1   # AprilTag (优先, 给 6D 位姿 + ID)
uint8 SOURCE_YOLO     = 2   # BPU YOLO (兜底, 仅 2D bbox)
uint8 SOURCE_HOUGH    = 3   # OpenCV Hough 圆检测 (最末端兜底)
uint8 source

# === 位姿 (在 lift_camera_optical_frame 下) ===
geometry_msgs/Pose pose      # 仅 source=APRILTAG 时全字段有效

# === 2D bbox (像素坐标, 像素中心 + 宽高) ===
int32 bbox_cx
int32 bbox_cy
int32 bbox_w
int32 bbox_h
int32 image_w     # 检测时图像分辨率 (1920)
int32 image_h     # (1080)

# === 标签 / ID ===
int32 tag_id          # AprilTag id, 同时也是瓶子编号; YOLO 时 = 类别 id
string label          # e.g. "SYGO-1", "YCAS-3"
float32 confidence    # 0~1

# === 是否找到 ===
bool valid
