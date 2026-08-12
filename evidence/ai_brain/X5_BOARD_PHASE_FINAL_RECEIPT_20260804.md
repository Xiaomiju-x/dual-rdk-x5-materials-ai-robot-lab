# X5-ICMat Foundry 板端阶段最终回执

状态：`COMPLETE_WITH_REJECTIONS_AND_EXPERIMENTALS`

## 结论

- 官方 50 模型 registry 与两份 release 未改动；板端结果以本独立 overlay 记录。
- 38 个决赛新增候选已全部形成逐模型状态：`31 X5_VALIDATED / 3 BOARD_REJECTED / 4 BOARD_EXPERIMENTAL`。
- 24/24 个 BPU-primary 均在 actual X5 BPU 执行；14 个 CPU-primary 中 11 个完成 actual X5 CPU 推理，3 个因 staging 运行资产不完整而拒绝。
- 三套分段 BPU LLM 的 6 个 bin 均逐段 actual X5 执行且 part1/part2 content hash 绑定正确；固定 next-token 不一致，因此三者均保留 `BOARD_EXPERIMENTAL`。
- 最终生产非干扰检查 `33/33 PASS`：五端口、生产 9000–9003 LLM 槽、Dashboard 健康接口、BPU slot 健康接口、相机状态、生产哈希和 staging 哈希均保持；候选进程、19010/19011 和候选 systemd 单元均不存在。

## 模型会计

| 项目 | 数量 |
|---|---:|
| 注册表唯一逻辑模型 | 50 |
| X5-local 逻辑模型状态完整 | 49（冻结生产 11 + 决赛新增 38） |
| PC-only MACE-MPA-0 | 1 |
| 新增 X5_VALIDATED | 31 |
| 新增 BOARD_REJECTED | 3 |
| 新增 BOARD_EXPERIMENTAL | 4 |
| actual X5 backend 已执行 | 35（CPU 11 + BPU 24） |

## 三套 BPU LLM 固定任务结果

| 模型 | 期望 token | 实际 token | part1/part2 加载 ms | part1/part2 推理 ms | 状态 |
|---|---:|---:|---:|---:|---|
| F-LLM-03 | 4913 | 48380 | 5308.2 / 4637.2 | 213.1 / 218.0 | BOARD_EXPERIMENTAL |
| F-LLM-04 | 7424 | 1071 | 5010.6 / 4966.5 | 221.9 / 212.9 | BOARD_EXPERIMENTAL |
| F-LLM-05 | 7481 | 88582 | 4963.3 / 4753.7 | 225.1 / 217.8 | BOARD_EXPERIMENTAL |

该结果只证明固定输入下两段 BPU 与 CPU head 的一次 next-token 合同执行，不证明自由生成、通用问答或 FP32 hidden-state 数值一致。

## 例外边界

- `F-KNW-01/02/04`：`BOARD_REJECTED`，仅有 safetensors，缺少完整 config/tokenizer/可执行 loader；未在 X5 上伪装成成功推理。
- `F-KNW-03`：`BOARD_EXPERIMENTAL`，actual BPU 已执行，但固定标量任务门未通过（NRMSE 0.52269）。
- `F-LLM-03/04/05`：`BOARD_EXPERIMENTAL`，6 个 bin 均 actual BPU 执行，但固定 next-token 不一致。
- `F-PROC-03` 继续为 `QUALITY_LIMITED_NOT_PROMOTED`；`F-PKG-01/02/03/04` 的 `SIM_ONLY` 边界继续保留。
- `X5_VALIDATED` 表示按需单模型固定任务合同通过，不表示 49 个 X5 模型同时常驻。
- PC acceptance 中 `x5_board_verified=0` 是冻结的上电前事实；没有改写，板端结果只记录在本 overlay。
- PC acceptance 工作区实际 SHA-256 为 `128a9d14050af63882054cccd9c3b30e41f8acda190719e0c25af280cf47a9ce`，且与两份冻结 release 内嵌 `acceptance.json` 字节一致；但 AGENTS/ledger/README 仍记录旧哈希 `aa03341e9bc44c5e47e63935035cebaafa4900e18f0afb3a6af6583eb6330668`，本回执保留该不一致告警，未覆盖任何基线文件。

## 38 个新增候选状态

| Inventory ID | 域 | Primary backend | 最终板端状态 |
|---|---|---|---|
| F-KNW-01 | KNW | CPU | BOARD_REJECTED |
| F-KNW-02 | KNW | CPU | BOARD_REJECTED |
| F-KNW-03 | KNW | BPU | BOARD_EXPERIMENTAL |
| F-KNW-04 | KNW | CPU | BOARD_REJECTED |
| F-LLM-01 | LLM | CPU | X5_VALIDATED |
| F-LLM-02 | LLM | CPU | X5_VALIDATED |
| F-LLM-03 | LLM | BPU | BOARD_EXPERIMENTAL |
| F-LLM-04 | LLM | BPU | BOARD_EXPERIMENTAL |
| F-LLM-05 | LLM | BPU | BOARD_EXPERIMENTAL |
| F-MAT-01 | MAT | BPU | X5_VALIDATED |
| F-MAT-02 | MAT | BPU | X5_VALIDATED |
| F-MAT-03 | MAT | BPU | X5_VALIDATED |
| F-MAT-04 | MAT | BPU | X5_VALIDATED |
| F-MAT-05 | MAT | BPU | X5_VALIDATED |
| F-MAT-06 | MAT | CPU | X5_VALIDATED |
| F-MAT-07 | MAT | CPU | X5_VALIDATED |
| F-MAT-08 | MAT | CPU | X5_VALIDATED |
| F-PKG-01 | PKG | BPU | X5_VALIDATED |
| F-PKG-02 | PKG | BPU | X5_VALIDATED |
| F-PKG-03 | PKG | BPU | X5_VALIDATED |
| F-PKG-04 | PKG | CPU | X5_VALIDATED |
| F-PROC-01 | PROC | BPU | X5_VALIDATED |
| F-PROC-02 | PROC | BPU | X5_VALIDATED |
| F-PROC-03 | PROC | BPU | X5_VALIDATED |
| F-PROC-04 | PROC | BPU | X5_VALIDATED |
| F-PROC-05 | PROC | BPU | X5_VALIDATED |
| F-PROC-06 | PROC | BPU | X5_VALIDATED |
| F-PROC-07 | PROC | CPU | X5_VALIDATED |
| F-PROC-08 | PROC | CPU | X5_VALIDATED |
| F-PROC-09 | PROC | CPU | X5_VALIDATED |
| F-SEM-01 | SEM | BPU | X5_VALIDATED |
| F-SEM-02 | SEM | BPU | X5_VALIDATED |
| F-SEM-03 | SEM | BPU | X5_VALIDATED |
| F-SEM-04 | SEM | BPU | X5_VALIDATED |
| F-SEM-05 | SEM | CPU | X5_VALIDATED |
| F-SEM-06 | SEM | CPU | X5_VALIDATED |
| F-XRD-01 | XRD | BPU | X5_VALIDATED |
| F-XRD-02 | XRD | BPU | X5_VALIDATED |

## 哈希

- 本 JSON 回执 SHA-256：`b71b8f02957fb85db87c1fd80fc8ad38df90b369724573976b443e39318bcabf`
- registry SHA-256：`a2293bce08d6de380dbbbcf8876381e946d329692bc07dc98dec88199d2f7ef2`
- PC acceptance 实际/两份 release 内嵌 SHA-256：`128a9d14050af63882054cccd9c3b30e41f8acda190719e0c25af280cf47a9ce`
- 文档仍记录的 PC acceptance SHA-256：`aa03341e9bc44c5e47e63935035cebaafa4900e18f0afb3a6af6583eb6330668`（待后续单独核账，不在本次上板中改写）
- X5 staging SHA-256：`c5fa215a58168c0cb7274c2b1cf6d66bcd0f3c1e70d3f4cf13749e9b57dafb52`

下一步只允许对本回执执行一次只读 `FleetAudit PASSIVE_ONESHOT`，随后确认 `DEPLOYED_OFF`。
