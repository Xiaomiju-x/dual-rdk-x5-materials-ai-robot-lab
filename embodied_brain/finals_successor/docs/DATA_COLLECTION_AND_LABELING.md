# X5-TriBEV-Flow 数据采集与标注合同

状态：`CANDIDATE-ONLY / OFFLINE-FIRST / NO-CONTROL-AUTHORITY`

本文件只定义决赛 successor 候选的数据层。它不修改、不替换也不阻塞：

- `bash ~/tools/finals_lift_nav_demo.sh`
- F407 build `2026071907`
- 已验证的升降台动作
- `0.50 m odom` 闭环直行
- 现有 SLAM、RViz 和 Lab-FSD 观察链

候选训练、回放或数据校验失败时，只能把候选标记为离线，不能改变复赛演示行为。

## 1. 固定模型输入合同

每个 NPZ episode 保存 `tribev_input[5,8,64,64]`。TinyOccFlow 输入按历史帧优先展平：

```text
[5 history, 8 channels, 64, 64]  (storage: oldest -> newest)
            |
            | reverse history axis, then reshape
            v
[1 batch, 40 channels, 64, 64]
```

模型通道索引为：

```text
model_channel = history_index * 8 + frame_channel
```

模型侧 `history_index=0` 固定为 `t0`，随后为 `t-1 ... t-4`。NPZ
存储仍按严格递增时间戳保存，因此必须统一调用
`flatten_tribev_history()`，禁止直接 `reshape`。

不得加入速度、目标点、时间戳等全局标量来改变 40 通道合同。此类信息只能留在 `metadata_json` 或独立评测记录中。

### 每帧 8 通道固定顺序

| 帧内索引 | 名称 | 语义 |
|---:|---|---|
| 0 | `lidar_occupancy` | 当前历史帧 LiDAR 在 `base_link` BEV 中观测到的占用概率 |
| 1 | `lidar_visibility` | 当前历史帧 LiDAR 已观测区域；不能把未观测区当自由空间 |
| 2 | `depth_near` | 当前历史帧深度占用，v1 距离带 `[0,1.5 m)` |
| 3 | `depth_mid` | 当前历史帧深度占用，v1 距离带 `[1.5,3.0 m)` |
| 4 | `depth_far` | 当前历史帧深度占用，v1 距离带 `[3.0 m,有效量程]` |
| 5 | `camera_semantic_risk` | 当前历史帧真实或合成相机语义风险，范围 `[0,1]` |
| 6 | `sensor_validity_fraction` | 该历史帧 LiDAR、深度、4K 三个 validity 标志的均值；里程计 warp 后无覆盖格为 0，其余格为 `1/3`、`2/3` 或 `1` |
| 7 | `fused_occupancy` | **仅融合当前历史帧**通道 0、2、3、4、5 的逐格最大值 |

`fused_occupancy` 不是 dynamic proxy，也不能读取 `future_occupancy`、`future_flow_m`、`dynamic_mask` 或未来时间戳。生成器和校验器都执行这一约束，避免未来标签泄漏。

BEV 固定为 `base_link`：`+x` 向前、`+y` 向左，分辨率 `0.10 m`，`64x64`，覆盖 `x=[-1.2,5.2) m`、`y=[-3.2,3.2) m`。

## 2. NPZ episode 成员

所有文件必须用 `numpy.load(..., allow_pickle=False)` 读取，禁止 object array。

| 成员 | dtype / shape | 含义 |
|---|---|---|
| `schema_version` | Unicode scalar | 固定 `x5-tribev-episode.v1` |
| `episode_id` | Unicode scalar | 全数据集唯一 episode 标识 |
| `session_id` | Unicode scalar | 防泄漏划分的最小组标识 |
| `scenario_id` | Unicode scalar | 场景类别 |
| `metadata_json` | Unicode scalar | 符合 `tribev_episode.schema.json` 的 JSON |
| `timestamps_ns` | `int64[5]` | 五个历史帧时间戳，严格递增，末帧为 `t0` |
| `history_offsets_s` | `float32[5]` | 相对 `t0` 的历史时间 |
| `future_timestamps_ns` | `int64[3]` | 三个未来标签时间戳 |
| `future_horizons_s` | `float32[3]` | v1 为 `0.4/0.8/1.2 s` |
| `tribev_input` | `float32[5,8,64,64]` | 仅使用 `t<=t0` 的模型输入 |
| `future_occupancy` | `float32[3,64,64]` | 三个未来时域占用标签 |
| `future_flow_m` | `float32[3,2,64,64]` | 动态占用格从 `t0` 到各时域的 `x/y` 位移，单位 m |
| `dynamic_mask` | `float32[3,64,64]` | flow 有效的动态占用区域 |
| `uncertainty_target` | `float32[3,64,64]` | 未观测、边界、标签冲突等监督目标 |
| `trajectory_soft_labels` | `float32[9]` | 九条固定弧轨迹的概率分布，总和为 1 |
| `trajectory_token_omega_rad_s` | `float32[9]` | 严格递增的九个角速度 token |
| `sensor_validity` | `uint8[5,3]` | 按 `lidar/depth/vision_4k` 排列的逐源有效标志 |
| `sensor_age_s` | `float32[5,3]` | 有效源的消息年龄；无效源固定为 `-1` |
| `sensor_provenance` | Unicode `[5,3]` | 逐源来源状态 |
| `vision_image_supplied` | `uint8[5]` | 4K 请求是否确实带入了图像 |

JSON Schema 校验 `metadata_json` 的结构；`dataset.py` 继续校验数组 dtype、shape、时间单调性、soft-label 总和、来源真实性、同帧融合和 flow mask 等跨数组约束。

## 3. 真实采集话题

开始采集前先保存：

```bash
ros2 topic list -t
ros2 node list
ros2 param dump /slam_toolbox
```

不同 Orbbec/IMU 驱动的原始话题名称可能不同。采集器必须在 session manifest 中记录“实际话题名、ROS 类型、帧数和 hash”，不能只按下表猜测。

| 数据 | 必需/建议 ROS 话题 |
|---|---|
| LiDAR 原始几何 | `/scan`；同时保留驱动原始 scan 话题（如存在） |
| 现有深度降维线 | `/scan_depth`，用于和冻结链对照，不替代原始深度 |
| 原始深度 | 实际部署的 `/camera/depth/image_raw` 或 `/camera/depth/image_rect_raw` |
| 深度内参 | 实际部署的 `/camera/depth/camera_info` |
| 深度点云 | `/camera/depth/points` 或实际等价话题，建议保留 |
| IMU | `/imu/data_raw`、`/imu/data` 或实际部署等价话题 |
| 里程计 | `/odom` |
| 坐标变换 | `/tf`、`/tf_static` |
| 实际控制证据 | `/cmd_vel`；只作为标签/审计数据，候选不发布该话题 |
| 规划教师 | `/plan`、Nav2 全局/局部 path 和实际 costmap 话题 |
| SLAM 对照 | `/map`、`/map_metadata` |
| 现有 shadow | `/lab_fsd/bev`、`/lab_fsd/future_bev`、`/lab_fsd/policy_tokens`、`/lab_fsd/trajectory_scores`、`/lab_fsd/input_status`、`/lab_fsd/safety_gate` |
| 健康与机构状态 | `/diagnostics`、`/lift_status`、`/f407/estop_latched`、`/f407/cmd_vel_expired` |

AI 脑 IMX415 4K 相机当前不是具身 X5 的本地 ROS 相机。真实采集需要把单帧图像、AI 脑服务端时间、请求时间、响应时间、相机锁状态和完整 provenance JSON 与车端 rosbag 放进同一 session manifest。后续若增加候选 ROS bridge，也必须保留原 HTTP 证据，不能只保留投影后的 BEV。

## 4. 真实 4K 判定

某历史帧只有同时满足以下条件，才允许作为真实 4K 语义输入：

```text
sensor_validity[t, vision_4k] == 1
sensor_provenance[t, vision_4k] == "live_camera"
vision_image_supplied[t] == 1
```

此外应保留源图 hash、采集时间、相机内参/外参版本和共享锁正常释放证据。

以下来源一律不能冒充真实 4K：

- `cached_camera`
- `fixture_prior`
- `unavailable`
- `modality_dropout`
- 没有原图或原图 hash 的推理结果
- 只有网页截图、没有服务端采集证据的画面

这些情况必须将视觉 validity 置 0、相机通道置零，并保留实际 provenance。合成数据使用 `synthetic`，`vision_image_supplied=0`，只能用于管线和训练增强。

## 5. 采集规模与六类场景

首轮真实数据目标为 **45-90 分钟**，至少覆盖 6 类独立场景。每类应分多个独立 session 采集，而不是一条长 bag 切成随机帧。

| 场景 | 建议有效时长 | 关键变化 |
|---|---:|---|
| 开阔直走/普通走廊 | 8-15 min | 不同光照、速度和墙距 |
| 窄走廊/门口 | 6-12 min | 可通行宽度和近场深度 |
| 静态杂物 | 8-15 min | 箱体、推车、桌腿、反光物 |
| 动态人员横穿 | 8-15 min | 不同速度、距离和遮挡 |
| 同向动态目标/移动推车 | 8-15 min | occupancy flow 和 TTC |
| 模态缺失/延迟 | 7-18 min | 分别断开视觉、深度或模拟消息过期 |

原始 session 不做破坏性清洗。转换失败、来源不清或时间同步异常的样本保留在隔离区，不能静默删除后重新编号。

## 6. session 级划分

`session_id` 表示一次连续采集，其中场地、设备标定、参与者集合、网络配置和 provenance 合同不变。由同一 rosbag、同一视频或相邻滑窗产生的所有 episode 必须保持同一 `session_id`。

固定划分为：

```text
train       60%
calibration 20%
test        20%
```

`dataset.py` 对 `seed:session_id` 做稳定 SHA-256 hash 分桶。严禁：

- 随机按帧划分；
- 同一 bag 的不同窗口进入不同 split；
- 用 test 调模型、阈值或 conformal 分位数；
- 看到测试结果后改 session_id 重新分桶。

小数据集可能因 hash 分桶出现空 split，此时应增加独立 session，而不是移动单帧。`assert_no_session_leakage()` 是训练前硬门。

## 7. 时间对齐与标签生成

1. 选择末历史帧为 `t0`，目标历史时刻为 `-0.8/-0.6/-0.4/-0.2/0.0 s`。
2. 使用 `/odom` 与 `/tf` 将各历史观测运动补偿到 `t0` 的 `base_link`。
3. 超过该传感器配置同步容差的样本不插值伪造，置 invalid 并记录实际 age。
4. 先完成 8 通道历史输入并锁定 hash，再读取 `t>t0` 数据生成未来标签。
5. 将未来 LiDAR 与原始深度反投影到 `t0`，生成 `future_occupancy`；未观测区不能标自由。
6. 通过跨时域实例关联、几何差分或离线教师生成 `dynamic_mask` 和 `future_flow_m`；静态及无效格 flow 为零。
7. `uncertainty_target` 综合未观测区域、占用边界、跨模态冲突和标签器分歧，但不能被当作概率安全保证。
8. 九轨迹 soft label 来自冻结 Nav2/MPPI 回放、未来占用碰撞代价和可解释几何代价；它是 shadow 教师标签，不是电机命令。

防未来泄漏验收至少包含：

- 改写未来三个标签后，`tribev_input` 的 hash 必须完全不变；
- 第 8 通道必须逐格等于同一帧通道 `0/2/3/4/5` 的最大值；
- 每个 episode 的最大输入时间必须 `<=t0`，最小标签时间必须 `>t0`；
- 归一化统计只能从 train session 计算。

## 8. 合成数据

`synthetic.py` 可复现生成六类室内 episode，包括走廊、静态杂物、动态人员和模态 dropout：

```bash
python3 embodied_brain/finals_successor/x5_tribev_flow/synthetic.py \
  --output /tmp/x5_tribev_synthetic \
  --sessions-per-scenario 3 \
  --episodes-per-session 4 \
  --seed 20260728
```

完整校验与防泄漏划分：

```bash
python3 embodied_brain/finals_successor/x5_tribev_flow/dataset.py \
  validate /tmp/x5_tribev_synthetic

python3 embodied_brain/finals_successor/x5_tribev_flow/dataset.py \
  split /tmp/x5_tribev_synthetic --seed 20260728
```

相同代码版本、NumPy 版本和 seed 产生相同数组内容；NPZ ZIP 容器的字节 hash 可能因压缩库版本不同而变化，因此数据证据应同时记录解码数组 hash、生成器版本和环境版本。

## 9. 无 Torch 与 PyTorch 使用

元数据发现、读取、验证和 session 划分只依赖 NumPy。模块在没有 Torch 时仍可导入：

```python
from dataset import build_episode_refs, read_episode_metadata

metadata = read_episode_metadata("episode.npz")
refs = build_episode_refs("/data/x5_tribev")
```

仅实例化 `TriBEVEpisodeDataset` 时要求 PyTorch。Dataset 同时返回：

- `tribev_input`: `[5,8,64,64]`
- `model_input`: `[40,64,64]`
- future occupancy/flow/dynamic/uncertainty/trajectory targets
- 时间、validity 和 provenance 元数据

## 10. 许可与隐私

- 每个真实 session 必须有明确 `license_id`、原始 manifest/hash、采集主体和使用范围。
- 原始 4K 帧可能包含人脸、工牌、屏幕、实验记录和位置特征，默认私有保存，不进入公开仓库。
- 有人员的采集需取得适用的知情同意或机构批准；不能取得时应停止采集，而不是事后假定许可。
- 对外导出前执行人脸/人体/屏幕审查、访问控制和保留期限策略，并保留不可逆处理记录。
- 合成数据标记为 `project-generated-synthetic-v1`；它不能证明真实传感器精度。
- 外部数据、教师权重和自动标注器分别记录来源及许可证，许可证不兼容的数据不得混入发布集。
- LLM/API 生成结果只能作为待审候选或弱标签，不能直接作为 ground truth。
- 任何含密钥、账号、SSID、私人地址或原始身份信息的内容不得写入 NPZ metadata、文档或公开证据。

## 11. 准入检查

训练前必须全部满足：

1. `dataset.py validate` 对全部 episode 返回成功。
2. Schema、代码、原始 session 和校准文件均有 hash。
3. session leakage 为 false。
4. test split 从未参与训练、阈值选择或手工修正。
5. 真实 4K 三条件可从原始证据复核。
6. 合成、replay、real 三类来源分别统计，不混写为真实样本量。
7. `tribev_input` 固定 `5x8x64x64`，展平后固定 `1x40x64x64`。
8. 候选数据工具没有发布 `/cmd_vel`、F407 命令或权威 TF。
