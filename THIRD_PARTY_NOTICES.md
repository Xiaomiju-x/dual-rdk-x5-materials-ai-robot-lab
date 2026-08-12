# 第三方来源与许可声明

Apache-2.0 不能自动覆盖第三方数据、模型、论文、字体、网页库、照片或混合来源资产。使用者必须同时遵守各资源自身的许可证和使用条款。

## 权威机器可读清单

材料智能资源的版本、URL、许可证、再分发状态、风险和断言边界记录在：

- [`ai_brain/icmat_foundry/contracts/source_catalog.v1.json`](ai_brain/icmat_foundry/contracts/source_catalog.v1.json)

该目录当前至少记录了 NIST JARVIS-DFT、UCI SECOM、NIST SEM 数据、Qwen 和 DeepSeek 模型来源，也明确标记了不能批量再分发的内部/逐文档许可内容。`downloaded` 只表示当时取得资源，不表示该资源一定随本仓库分发。

## 代表性资源

| 资源 | 上游许可/状态 | 本项目边界 |
| --- | --- | --- |
| NIST JARVIS-DFT 3D | CC BY 4.0；详见 source catalog | 计算材料性质筛选，不等于实验或产线验证 |
| UCI SECOM | CC BY 4.0；DOI `10.24432/C54305` | 2008 匿名公开基准，不是团队或现代晶圆厂数据 |
| NIST SEM segmentation data | NIST Open License；候选资源 | 模拟 SEM 图像，不证明真实晶圆缺陷泛化 |
| Qwen 基座模型 | 各模型卡所列 Apache-2.0 | 下载不证明领域适配、BPU 转换或 X5 运行 |
| DeepSeek-R1-Distill-Qwen | 上游 MIT 与 Qwen 基座声明 | 仅证据辅助推理候选，不能替代测量和工程判断 |

## 随仓分发的浏览器组件

| 文件 | 上游组件 | 版本/版权 | 许可文本 |
| --- | --- | --- | --- |
| `public_site_static/three.min.js` | three.js | r128；Copyright © 2010–2021 three.js authors | [MIT](third_party/licenses/threejs-MIT.txt) |
| `public_site_static/GLTFLoader.js` | three.js GLTFLoader | 与归档站 r128 配套 | [MIT](third_party/licenses/threejs-MIT.txt) |
| `public_site_static/model-viewer.min.js` | Google model-viewer | Copyright 2017 Google LLC | [BSD-3-Clause](third_party/licenses/model-viewer-BSD-3-Clause.txt) |

上述归档文件保留原始内联版权头。`workstation_frontend_public/package-lock.json` 记录前端依赖的精确解析版本；安装依赖时还须遵守各 npm 包自身携带的许可证。

## 发布检查

每个实际随版本分发的第三方文件都应具有：来源 URL、固定版本或 revision、文件哈希、许可证、版权声明、再分发依据和本项目用途。来源不明、禁止再分发或授权未确认的文件不得进入发布制品。

媒体还必须记录拍摄者、人物授权、EXIF/GPS 清理和适用许可。详见 [公开边界](docs/safety/PUBLICATION_BOUNDARY.md)。
