# XRD智能分析系统 — 数值线

> 首个在嵌入式AI芯片(RDK X5 BPU)上运行的粉末XRD智能分析系统

## 功能特性

- **BPU INT8加速推理**: 两级级联分类, <1ms确定性延迟
- **理论图谱正演**: 从CIF晶体结构计算理论XRD, 与实测叠加对比, Rwp残差分析
- **TDA持久同调**: 首创拓扑数据分析方法用于XRD峰位分类
- **Conformal Prediction**: 95%覆盖率数学保证的不确定性量化
- **AI Agent**: DeepSeek-R1推理模型 + RAG(197篇论文/2255段落) + 工具调用
- **晶体学分析**: Scherrer微晶尺寸 / 系统消光验证 / Nelson-Riley晶格精修
- **语音交互**: VAD + 百度ASR + TTS, ALSA设备自动检测
- **Web可视化**: Canvas谱图 + 知识图谱 + 3D晶体 + 暗色模式 + PWA

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 训练模型
python scripts/train.py

# 导出ONNX
python scripts/export_onnx.py

# 部署到RDK X5
bash deploy_to_x5.sh 198.51.100.103

# X5上启动
python3 web_demo.py --port 8080
```

## 硬件要求

| 组件 | 型号 | 用途 |
|------|------|------|
| 主控 | RDK X5 (Bayes-e BPU, 10TOPS) | 推理+Web服务 |
| 音箱 | M260C USB智能音箱 | 语音交互 |
| 摄像头 | IMX415 4K (视觉线用) | XRD图像检测 |

## 技术架构

```
.raw → 峰提取 → 190D特征 → BPU级联 → 峰匹配 → RAG检索 → AI Agent → 语音播报
                                ↓           ↓          ↓
                          Conformal    消光验证    理论图谱+Rwp
                          MC Dropout   TDA        Scherrer
                          XAI归因    Nelson-Riley  知识图谱+3D晶体
```

## 竞赛信息

- **比赛**: 2026全国大学生嵌入式芯片与系统设计竞赛
- **赛道**: 地瓜机器人 - 选题一 (RDK X5)
- **平台**: NodeHub开源提交

## 许可证

MIT License
