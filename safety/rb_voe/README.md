# X5-RB-VoE

X5-RB-VoE（Risk-Bounded Value of Evidence）是面向材料实验闭环的**风险有界证据-动作编译器**。它不替代 XRD/PL、YOLO、LLM、材料预测模型，也不替代地瓜机器人的 BPU 模型转换工具链；它位于这些模型与真实执行器之间，把“当前结论缺少什么证据”编译为一个受约束的下一证据动作，并在证据、能力、安全和权限任一条件不足时返回 `HOLD/NOT_READY`。

批准的终局架构原文属于历史内部规划，未随发布树分发；公开架构与权限边界见[系统架构](../../docs/architecture/SYSTEM_ARCHITECTURE.md)。本文只说明 `rb_voe/` 当前已经实现的 R0/R1 软件能力，不代表终局系统已经完成。

等待硬件期间完成的 D 工作线证据启动包原文未随发布树分发；公开可核查材料以[证据索引](../../docs/evidence/EVIDENCE_INDEX.md)为准。该历史里程碑准备了来源目录、监督式 file-drop 合同、run-scoped replay/stale 负测、C0-C7 casebook 和只允许 catalog/E 离线回放的 claim gate，但不代表 `R2-PREP-D`、live assay mapper 或物理闭环已经完成。

2026-07-19 的 R2-PREP A/B 复赛预接线原文未随发布树分发；公开范围遵循[公开边界](../../docs/safety/PUBLICATION_BOUNDARY.md)。中央 runner、具身脑 finals 只读采集器、`scan_raw -> scan_self_filter -> scan` 自滤波代码、双臂 finals 只读探针、A0 顶视实际记录合同，以及 AI X5/Dashboard/边缘采集器的部署与回滚脚本曾按固定拓扑预写。该里程碑只代表离线集成候选；设备电源状态未观测、没有发起网络连接、没有运行 live shadow，也没有产生动作权限。当前首次真机运行仍预期 `HOLD`：表征站保持 `TARGET_ONLY`，具身脑仍须在同一启动周期采集 hash-bound 车体轮廓并完成自滤波、TF、Collision Monitor 和进程来源的真机验收。

## 当前结论

截至 2026-07-14，代码版本为 `0.4.0-r1-integration-prepared`，命令行报告 `R1_INTEGRATION_PREPARED_R2_LIVE_NOT_RUN`。这表示离线合同、策略、模拟、故障注入、历史只读回放、严格语义审计和外部固定发布包已通过本地 R1 门禁，并且四系统零权限 shadow 编排器、AI X5 严格只读运行身份适配器及网站公开只读证据视图已经实现；任何没有单独提供外部 pin 的候选都必须以 `UNPINNED_CANDIDATE` 拒绝。AI X5 代码尚未部署到本次未开机的板端，R2-R6 真机阶段仍未执行。

| 范围 | 当前状态 | 能够证明 | 不能证明 |
|---|---|---|---|
| R0 合同与边界 | 已实现 | 合同、注册表、规范化摘要、目标适配器和严格 AI X5 只读适配器可 fail-closed | 双 X5、双臂或仪器已经完成真机联合接入 |
| R1 策略与模拟 | 已通过离线门禁 | 固定输入下可产生确定性策略树，并对登记联合场景做反事实执行和故障注入 | 策略在真实材料和真实机器人上有效 |
| R1 策略对照 | 已通过离线门禁 | 同一 sealed world 下重算 H2、H1、严格固定二步、全证据成本参考和全局硬 HOLD；对照结果受发布清单约束 | 离线风险差异等同真实实验收益 |
| R1 回放、审计与发布 | 已通过离线门禁 | sealed 输入、历史工件只读封存、语义账本、终端清单和外部固定根可检测多类篡改及协同重写 | 外部固定根等同硬件可信根 |
| 物理许可核心 | 代码已实现，真机未接入 | 许可合同、签名域隔离和持久化防重放可以被离线测试 | 存在生产密钥、真实签名服务或物理启动权限 |
| Locked 统计协议 | 代码已实现 | 可对完整 matched-pair 记录执行冻结门禁计算 | 已经取得任何 locked 真机结论 |

R1 的证据来源固定为 `SIMULATED_COUNTERFACTUAL`：

- `hardware_touched=false`；
- `execution_authority=false`；
- `physical_closure_proven=false`；
- `physical_risk_denominator_increment=0`。

因此，R1 不能被写成真机验证、物理闭环完成、全系统完成，也不能支持“全球第一”“全球首创”或获奖保证。

## 本质与创新边界

### 它是什么

X5-RB-VoE 的算法对象是“证据是否独立、失败核是什么、下一项证据动作是否值得且可执行”，而不是“哪个模型分数最高”。核心链路是：

```text
ExperimentCase
-> Evidence DAG 与硬不变量
-> 登记扰动内的 best-found 反例和 failure core
-> 有限时域、联合根场景下的风险有界 option 策略
-> ExecutionChallenge / JointPermit
-> 新证据与 DecisionReceipt
-> 语义账本、终端清单、外部固定发布根
```

终局创新主张应建立在真实闭环结果上：已有模型不再只是分别运行，而是以可追溯证据、共享故障域、执行能力和本地安全共同约束下一实验动作。R0/R1 目前只完成了这条主张的软件骨架与可证伪协议。

### 它不是什么

- 不是新的材料分类器、视觉模型或 LLM；
- 不是自动替用户给已有模型分类或做模型排行榜；
- 不是 Horizon mapper、BPU 编译器或 CPU 到 BPU 的转换器；
- 不是端到端 VLA、关节控制器、Nav2 替代品或 `/cmd_vel` 发布者；
- 不是用哈希、签名或模拟器替代真实物理 evidence；
- 不是 Conformal、主动学习、POMDP 或安全许可机制的原创性声明。

能否形成有竞争力的算法创新，最终取决于后续 locked 真机实验是否证明：failure core 会改变动作选择，所选动作能以更低受限成本正确关闭不确定性，并且双 X5、双臂和实际 BPU 消融产生可重复任务差异。没有这些结果时，只能称为工程方法和候选算法系统。

### R1 可复验策略对照

当前 blind-sample sealed world 的对照结果不是手写展示数值，而是由同一个 `plan_policy`、场景集、option set、Evidence admission 和 closure predicate 重算：

| 对照 | 结果 | 可解释结论 |
|---|---:|---|
| H2 自适应策略 | robust risk `4.0`，全场景 modeled closure | 最多两项证据，最坏证据采集成本 `3.4` |
| H1 | robust risk `8.0`，不保证终局 closure | 证明第二步条件分支在当前模型中有任务差异 |
| 严格固定二步 | `12/12` 候选均无法完整闭合，转为 `HOLD` | 不把按 observation 改第二动作的树伪装成固定序列 |
| 全证据参考 | 4 项、采集成本 `5.7`，standalone atom union 覆盖 | 没有 N-step outcome model，因此 `plan_risk=null`，不伪造三/四步风险 |
| 全局硬门失败 | `NO_FEASIBLE_OPTION`，3 个场景均零 observation | 证明硬门失败时不会执行“最接近可行”的动作 |

单 atom failure-core 探针在本 fixture 中保持首动作 `E_VERIFY_IDENTITY`，但会改变条件分支树与风险；报告明确记录 `first_action_changes=false`，不制造首动作翻转。对照正文和独立重算结果见最终证据包的 `strategy_comparison.json`。

## 分层结构

| 层 | 目录 | 当前职责 | 当前权限 |
|---|---|---|---|
| 合同层 | `contracts/` | 10 类 JSON Schema、Python 合同、冻结注册表、规范化 JSON 与 SHA-256 内容寻址 | 只验证数据；不启动设备 |
| 证据与失败核 | `core/evidence_dag.py`、`invariants.py`、`counterexample.py` | 表示 acquisition/派生/谱系关系，识别相关证据和硬不变量失败，在登记扰动内搜索有害翻转 | 不宣称全局鲁棒性证明 |
| 任务与策略 | `core/task_automaton.py`、`options.py`、`scenarios.py`、`policy.py` | 约束状态转移、冻结 option/场景，生成绑定 `ExperimentCase`、failure core、closure predicate 和 release 的确定性策略树 | 只选择高层 evidence option；不规划关节或底盘速度 |
| 模拟与故障注入 | `sim/` | 直接执行已哈希策略；覆盖冻结联合根场景，拒绝 provenance 漂移、硬门失败和不合法重复 | `SIMULATED_COUNTERFACTUAL`，无物理 authority |
| Sealed 回放 | `replay/` | 校验外部给定摘要、离线输入、case 完整覆盖和输出断言；只读封存 XRD/PL、四线、MCAP/F407、机械臂历史与模拟工件 | 网络禁用、硬件 authority 为 false、物理分母为 0；旧数据不授予研磨资格 |
| 许可层 | `security/` | 分离 challenge/permit 签名域，绑定事务、角色、区域、命令包络、release 和有效期；持久化消费 nonce | 默认拒绝；测试/模拟 verifier 不得进入物理 admission |
| 审计与发布 | `audit/`、`release/` | canonical append-only 哈希链；严格验证 `CASE -> POLICY_PLAN -> OBSERVATION+ -> TERMINAL`；由语义报告派生终端清单，并用外部 pin 检查协同重写 | 证明记录一致性，不证明记录对应真实世界 |
| 统计层 | `stats.py` | Clopper-Pearson、paired bootstrap、受限正确关闭时间和 locked matched-pair 门禁 | 只计算调用者提供且满足冻结合同的数据；当前无物理样本结论 |
| 设备适配 | `adapters/` | AI X5 严格只读运行身份适配器；具身 X5 finals snapshot；双臂 finals probe/A0 actual；表征站目标接口 | 四类均未完成本轮真机接入；具身为 `REPLAY_VALIDATED`、双臂为 `SHADOW_VALIDATED` 预接线，表征站仍为 `TARGET_ONLY/NOT_READY` |
| Shadow 编排 | `shadow/`、`live_shadow.py` | 绑定四系统 capability manifest、固定拓扑、一次性 challenge、case/sample lineage 和采集器摘要 | `PlanOnly` 与离线夹具已验证；没有 prepare/execute/trigger 接口，命令数和物理风险分母固定为 0 |

## CPU 与 BPU 分工

当前 R0/R1 全部是 PC/CPU 上的离线 Python 软件，没有新增 BPU 模型、BPU 二进制或 X5 真机运行证据。

- **CPU/策略核：** 合同验证、Evidence DAG、反例诊断、option 规划、模拟、回放、许可校验、账本、发布清单和统计。
- **BPU/既有模型：** 终局中继续运行 AI 脑的 XRD/PL/LLM BPU 模型和具身脑感知/风险模型，并输出带 actual backend、bin、preprocess、runtime、输入和 release 哈希的 evidence/capability capsule。
- **分工原则：** BPU 负责高吞吐模型推理；CPU 负责稀疏控制流、证据关系、权限和审计。RB-VoE 的核心价值不依赖把策略 Python 强行迁移到 BPU。

未来只有在实测证明某个固定张量评分热点影响时延或资源门时，才考虑把该子模块转换到 BPU；这不是当前主线的完成条件。

## 合同

当前冻结的 schema 可由命令列出，包括：

- `ExperimentCase`：样品/证据上下文、允许 option、failure atom 与 closure predicate；
- `EvidenceIntent`：AI 脑提出的证据动作意图；
- `ExecutionChallenge`：具身脑基于 live 状态返回的执行挑战；
- `JointPermit`：绑定 challenge、case、option、角色、区域、包络、release 和有效期的一次性许可；
- `PhysicalEvidenceCapsule`：外部获取的物理证据及其谱系、来源与时间绑定；
- `DecisionReceipt`：终端判断和使用证据的可追溯回执；
- 四类 capability manifest：AI X5、具身 X5、双臂、表征站。

Python 构造器和 JSON Schema 都执行注册表约束。未知 option、failure atom、角色、区域、authority/key domain 或 evidence 状态必须拒绝，不能静默降级成自由字符串。

## 本地执行

以下命令均为离线 R0/R1 命令，不连接 X5、Pi、相机或仪器，也不产生物理权限。

```powershell
# 查看版本、成熟度和权限边界
python -m rb_voe info

# 列出冻结 schema
python -m rb_voe schemas

# 用四份 capability manifest 做零权限 shadow 预检
python -m rb_voe shadow-preflight `
  --ai-manifest path\to\ai.json `
  --embodied-manifest path\to\embodied.json `
  --dual-arm-manifest path\to\dual_arm.json `
  --assay-manifest path\to\assay.json `
  --release-id x5-rb-voe-r1-demo-release-v1 `
  --run-id readonly-shadow-001 `
  --now-ms 0 `
  --mode OFFLINE_REPLAY

# 校验一个合同 JSON；可用 --now-ms 固定时间语义
python -m rb_voe validate-contract path\to\contract.json --now-ms 0

# 在一个新的输出目录运行确定性 golden demo
python -m rb_voe demo `
  --output-dir evidence\rb_voe_r1_demo_local `
  --bootstrap-unpinned `
  --full
```

demo 输出包括：

```text
demo_audit.jsonl
demo_audit.jsonl.anchor.json
release_manifest.json
terminal_manifest.json
registry_inventory.json
policy_inventory.json
environment_inventory.json
strategy_comparison.json
demo_result.json
```

输出目录应为新的空目录，避免把旧产物误当成本轮结果。`--bootstrap-unpinned` 只用于生成候选外部根，不建立独立信任，也不能替代已脱离输出目录保存的 pin。

将 `candidate_release_root.json` 另存到工作区外后，必须在另一个全新目录使用 `--external-pin <外部路径> --full` 重跑；只有该次输出的 `acceptance_status=PASS` 和退出码 0 才是已固定结果。

历史冻结证据目录 `rb_voe_r1_integration_prepared_20260714` 未随发布树分发；公开证据范围见[证据索引](../../docs/evidence/EVIDENCE_INDEX.md)。历史回执记录 Python 3.10.20 与 3.12.13 各 `343 passed`，发布根为 `4c6e9c2d06634e6285fc9e5b18ca2002a6ba0474414c2b461a707600b451f7b0`。Dashboard、四线服务和网站胶水不混入算法发布根，而由当时的 `integration_inventory.json` 单独绑定。该数字只代表当时离线 release 的内容寻址身份；外部 pin 是另存于工作区外的本地一致性锚，不是签名、不可变介质或硬件可信根，也不能由当前公开树独立复验。

```powershell
# 校验账本的 canonical 哈希链和 terminal anchor
python -m rb_voe audit-ledger evidence\rb_voe_r1_demo_local\demo_audit.jsonl

# 校验严格 R1 事件语义及跨记录绑定
python -m rb_voe audit-r1-ledger evidence\rb_voe_r1_demo_local\demo_audit.jsonl

# 分别校验内容寻址清单
python -m rb_voe verify-manifest evidence\rb_voe_r1_demo_local\release_manifest.json
python -m rb_voe verify-manifest evidence\rb_voe_r1_demo_local\terminal_manifest.json

# 使用外部固定根校验 release、ledger 和 terminal 的整体绑定
python -m rb_voe verify-release-bundle `
  evidence\rb_voe_r1_demo_local\release_manifest.json `
  evidence\rb_voe_r1_demo_local\terminal_manifest.json `
  C:\path\outside\workspace\trusted_release_root.json `
  --ledger evidence\rb_voe_r1_demo_local\demo_audit.jsonl `
  --project-root . `
  --registry-inventory evidence\rb_voe_r1_demo_local\registry_inventory.json `
  --policy-inventory evidence\rb_voe_r1_demo_local\policy_inventory.json `
  --environment-inventory evidence\rb_voe_r1_demo_local\environment_inventory.json
```

安装项目后，也可用等价入口 `x5-rb-voe`。JSON 输出中的 `ok`、`reason_code` 和进程退出码才是自动化判定依据，不能只看日志文本。

## 真机未完成边界

下列工作尚未由 R0/R1 完成：

1. AI X5、具身 X5 和双臂预接 adapter 的上板回放、真实 capability capsule 与剩余 runtime identity 字段；表征站、holder-loader 和仪器 adapter 的 live 字段映射与能力证书；
2. 两台 X5 的独立生产密钥、签名服务、boot/session 绑定和物理 admission 部署；
3. BPU actual-backend、模型/bin、输入、preprocess、runtime 与 CMA/slot 的真机 capsule；
4. 真实 XRD/PL raw、样品 lineage、holder/custody 与外部观察的同一 episode 绑定；
5. Nav2/F407、双臂 zone lease、工具能力、命令包络和本地 VETO 的联合 shadow/physical 验收；
6. CAL、冻结 release、120 个 locked matched pairs 及零硬安全违规门禁；
7. 双 X5、双臂和 BPU 的预注册消融；
8. 未剪辑物理 episode、失败注入和恢复证据。

在这些条件完成前：

- `TargetOnlyAdapter` 返回 `NOT_READY` 是正确行为，不是服务故障；
- 模拟 closure 只能写成 `SIMULATED_MODELED_ONLY`；
- sealed replay 不能增加物理风险分母；
- 测试签名、哈希链和外部 pin 不能被描述为生产硬件可信根；
- 不得声称“全自主材料实验”“风险已被实测约束”“算法系统已全部完成”。

## 安全与威胁

完整威胁模型见 [THREAT_MODEL.md](THREAT_MODEL.md)。实现遵循 fail-closed：未知注册值、摘要不一致、证据相关性未处理、provenance 漂移、challenge/permit 不匹配、过期或重放、语义事件缺失、终端权限字段抬升等情况都应拒绝，而不是以模型置信度补偿。
