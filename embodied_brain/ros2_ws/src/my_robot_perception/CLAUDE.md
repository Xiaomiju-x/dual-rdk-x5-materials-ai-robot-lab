# my_robot_perception

> BPU 感知 launch 集合 (yolo_world + edgesam). **Round 4 BPU Sprint 全完工 2026-04-30** (yolo_world 在 my_robot_agents 用 Python 节点; perception 包仅放 launch + config). Phase 3 烧结炉 OCR 用 7段 OpenCV 路径在 my_robot_agents 包.

## 是什么

把 BPU 推理封装成 ROS2 节点, 输入图像 topic, 输出结构化检测结果. 复用 AI 脑 [predict_engine/bpu_qwen.py](../../../../ai_brain/predict_engine/bpu_qwen.py) 套路, 但模型不一样 (这边是 YOLO/OCR, AI 脑是 Qwen2 24 层 Transformer).

## 节点 / launch (含 Round 4 BPU Sprint 集成的)

| Launch | Phase | 内容 | BPU bin |
|---|---|---|---|
| **yolo_world.launch.py** | Round 4 A1 ✅ | hobot_yolo_world 开放词检测, 订 /pt_camera/image_raw, 离线 CLIP 词表 30 类 | 官方 archive |
| **edgesam.launch.py** | Round 4 A2 ✅ | mono_edgesam 像素级分割 (RepViT-M1 1024×1024), 串 yolo_world 出框 → mask | 官方 |

**Phase 4 PP-OCRv4 试剂瓶 OCR / Phase 3 D1 XFeat / Phase 3 C2 MPPI / Phase 4 E1/E2/E3 麦阵**: 全部以 **Python 节点形式**放在 `my_robot_agents` 包 (因为 hobot_dnn Python binding 比 C++ 写得快, Phase 3 烧结炉 OCR 也是 Python 路径). 详见 [my_robot_agents/CLAUDE.md](../my_robot_agents/CLAUDE.md).

## BPU 模型管理 (X5 实际部署)

X5 上 `~/bpu_models/` 目录 (2026-04-30 实测):

```
/home/rdk/bpu_models/
├── ppocr_det.bin             2.7M   Round 4 A4 PP-OCRv4 det BPU 6ms 163FPS
├── lcd_yolov8n.bin           3.5M   Round 4 A3 7段 LCD 数字检测
├── xfeat.bin                 985K   Round 4 D1 视觉关键点 17ms 57FPS
├── cost_mlp.bin              264K   Round 4 C2 MPPI cost MLP 1.14ms/batch
├── smolvlm_decoder.bin       137M   Round 4 C1 SmolVLM-256M decoder INT8
├── smolvlm_decoder_kl.bin    137M   Round 4 C1 SmolVLM decoder kl 量化备选
├── smolvlm_vision.bin        119M   Round 4 C1 SmolVLM vision encoder
├── smolvlm_vision_part0.bin   57M   Round 4 C1 切段
└── smolvlm_vision_part1.bin   62M   Round 4 C1 切段
```

总 ~519MB INT8 (单次 CMA 391MB 装不下全部, 大模型用 subprocess swap-load).

## 7-段 OCR 流程 (Phase 3 H5 方案 - 走 my_robot_agents 包)

详见 [my_robot_agents/CLAUDE.md](../my_robot_agents/CLAUDE.md). 烧结炉 OCR 全部用 OpenCV + Qwen-VL 兜底, 不走 BPU.

## 已知坑 (传承自 AI 脑 Round 8 + 9)

- BPU Bayes-e v1.2.8 算子白名单: Linear + ReLU + Dropout 优先, MobileNet/BN/复杂算子不支持
- ONNX export 必须 `opset=11`, opset 17 fused ops (RotaryEmbedding/SimplifiedLayerNorm) hb_mapper 不认
- BPU CMA 391MB 硬限, > 200MB 模型必须切段 + subprocess per-forward；个人 Claude memory 不属于公开仓库，发布内的转换与分段边界见[公开 BPU LLM 工具链说明](../../../../ai_brain/icmat_foundry/finals_50model/bpu_llm_toolchain/README.md)
- 校准数据 float32 [0,255] 不归一化, MLP 用 `max`, YOLO 用 `kl`
