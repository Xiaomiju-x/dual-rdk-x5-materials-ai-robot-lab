# 软件物料清单（SBOM）

仓库根目录的 [`sbom.spdx.json`](../../sbom.spdx.json) 是可离线重建的
SPDX 2.3 JSON 软件物料清单。它用于回答“本次公开源码中声明或随附了哪些软件组件”，
并让依赖漂移在 CI 中直接失败；它不等同于漏洞扫描报告、许可证法律意见或完整的设备镜像清单。

## 覆盖范围

- 正式项目“基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人”
  的 `v1.0.2` 顶层项目包；
- `workstation_frontend_public/package-lock.json` 的 npm 根包及全部锁定的直接、传递和开发依赖；
- `requirements/*.txt` 中声明的 Python 依赖；
- `public_site_static/` 与 `web/command_center/static/` 随仓分发的 three.js r128、
  GLTFLoader r128 和 model-viewer 浏览器文件，并记录每个实际文件的 SHA-1 与 SHA-256；
- 项目、前端、依赖与 vendored 文件之间的 `DESCRIBES`、`CONTAINS` 和
  `DEPENDS_ON` 关系。

SBOM 不包含未进入公开仓库的模型权重、数据集、设备镜像、凭据、私有证据、构建缓存或
现场环境包。第三方来源与再分发说明仍以
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) 和各上游许可证文本为准。

## 精度边界

npm 依赖来自 lockfile v3：每个条目使用锁文件中的精确版本、下载位置、SRI 校验值和
许可证字段；SRI 的 Base64 摘要被无损转换为 SPDX 十六进制校验值。

Python 文件目前只给出版本范围，并不是锁文件。生成器因此刻意不填写 `versionInfo`，而在
每个 Python 包的 `comment` 中记录原始范围和 `Resolution status: unresolved`。这避免把某台
开发机偶然安装的版本伪装成可复现事实。需要字节级可复现的 Python 环境时，应另行发布带哈希的
平台锁文件或 wheelhouse，并据此扩展 SBOM。

three.js 与 GLTFLoader 的随仓版本可由文件内容确认是 r128；当前 model-viewer 压缩文件没有
可信的上游版本标识，所以 SBOM 保留包名、BSD-3-Clause 声明和文件哈希，但不猜测版本。

## 离线生成与校验

生成器只使用 Python 标准库，读取仓库内已有文件，不访问网络、相机、串口、机器人或其他硬件：

```bash
python -B tools/publication/generate_sbom.py
python -B tools/publication/generate_sbom.py --check
```

输出是确定性的：包、文件与关系均稳定排序，创建时间固定为公开版本纪元，文档命名空间后缀是
依赖清单与 vendored 文件内容的 SHA-256。相同输入会产生逐字节相同的 JSON；任何输入变化都要求
显式重新生成并审阅。CI 执行 `--check` 和 `tests_public/test_sbom.py`，防止提交陈旧或降级的清单。

## 审阅建议

在发布、合并依赖升级或处理安全通告时，至少同时核对：

1. `python -B tools/publication/generate_sbom.py --check` 通过；
2. npm 新增条目具有精确版本与完整性摘要；
3. Python 范围依赖仍被标为未解析，除非仓库确实增加了可验证锁文件；
4. 新的随仓第三方文件具有来源、许可证和文件级哈希；
5. SBOM 边界与公开发布边界、第三方声明保持一致。
