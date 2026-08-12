# X5-RB-VoE 威胁模型

本文描述 `rb_voe/` R0/R1 软件和终局真机架构的安全边界。它不是安全认证、密码学审计报告或真机验收记录。

## 1. 范围与资产

需要保护的资产包括：

- `ExperimentCase`、样品 lineage、raw acquisition 与 Evidence DAG 的真实性和关联关系；
- failure core、closure predicate、option 集、联合根场景和策略 release 的不可替换性；
- AI X5 与具身 X5 的 capability/challenge、双臂工具与 zone 状态；
- `JointPermit` 的一次性、短时、最小权限语义；
- observation、`DecisionReceipt`、账本、terminal manifest 和外部发布根的完整性；
- locked 统计的 parent block、随机化、ITT 完整性、时间上限和硬安全事件。

安全目标是：伪造、缺失、过期、重复、相关或越权信息不能让系统获得更大的物理权限，也不能被计入独立物理 evidence 或 locked 风险分母。

## 2. 信任边界

| 边界 | 可以信任什么 | 当前不能信任什么 |
|---|---|---|
| R0/R1 进程 | 当前代码对规范化、合同、策略和账本执行的一致性 | 主机、文件系统、系统时钟未被攻破 |
| 外部 pin | 脱离本轮输出目录保存的预期发布根 | 与产物一起被重写的“pin” |
| AI X5 | 终局中经独立密钥签名的科学 intent/permit 部分 | 当前不存在的生产签名或 live capability |
| 具身 X5 | 终局中经独立密钥签名的 challenge、本地 VETO 和执行状态 | 历史快照、fixture 或仅由 AI X5 转述的状态 |
| Pi/双臂/工位 | 终局中经资格验证的本地宏、工具、zone 与外部 outcome | 控制器自己的目标位姿作为动作成功 truth |
| 仪器/外部观察 | 新 acquisition、raw、holder/custody 和独立过程 evidence | 同一 raw 的多个派生模型输出被当成多份独立证据 |

R1 没有生产密钥、硬件可信根、外部单调计数器、live adapter 或物理 authority。`TestOnly*Verifier`、`SimulationSignatureVerifier`、本地 `DurableReplayStore` 和 golden fixture 只能用于测试。R1 物理入口无条件返回 `R1_PRODUCTION_PHYSICAL_AUTHORITY_NOT_IMPLEMENTED`，测试 verifier 或任意应用层自定义 verifier 都不能把它变成执行权限。

## 3. 攻击者与失效假设

考虑以下攻击者或工程失效：

- 能编辑本地 JSON、账本、清单、fixture 或运行参数的人；
- 能重放旧 challenge、permit、capability 或 observation 的进程；
- 被错误配置为代理其他节点启动动作的服务；
- 两个软件节点或两个证据派生链共享错误却被当成独立确认；
- 能同时重写 ledger、anchor、terminal 和 release 的攻击者；
- 模型/bin/preprocess、boot/session、地图、工具、zone 或样品身份在计划后漂移；
- 试图把 simulated/replay/shadow 成熟度提升为 physical/locked 权限的调用者。

R0/R1 不假设能抵抗已完全控制主机、密钥和外部 pin 保管位置的攻击者。终局也不能仅靠应用层 SHA-256 声称硬件级安全。

## 4. 核心威胁与控制

### 4.1 Tamper：单文件或字段篡改

**威胁：** 修改 case、策略、observation、receipt、schema、registry、fixture 或清单中的一个字段，仍让流程显示通过。

**当前控制：**

- canonical JSON 和内容 SHA-256；
- JSON Schema、Python 构造器和冻结注册表双层校验；
- append-only 前序哈希链与 terminal anchor；
- release manifest 绑定 source/schema/registry/fixture/policy/environment 清单；
- terminal manifest 构建时重新读取并验证实际 ledger，不接受调用者传入的可替换报告对象；
- 严格 release 验收重算实际 ledger、源码、schema、fixture 和逻辑 inventory，而不只比较三个 JSON payload。

**拒绝条件：** 未知字段/枚举、摘要不一致、前序哈希断裂、清单项目缺失、case/plan/episode/receipt 绑定不一致。

**剩余风险：** 攻击者若能控制整个主机和外部根，仍可重建一套自洽伪记录；内容完整性不等于物理真实性。

### 4.2 Replay：许可或证据重放

**威胁：** 重用旧 challenge/permit、跨 boot/session 使用 token、重复启动同一子宏，或用旧 evidence 冒充新 acquisition。

**当前控制：**

- permit 绑定 challenge、事务、case、option、角色、zone、命令包络、release、boot/session 和时间边界；
- challenge 与 permit 使用不同签名域和 key domain；
- `DurableReplayStore` 以 canonical JSONL、哈希链、anchor、文件锁和 `fsync` 提供本地测试诊断，但明确标记为 `LOCAL_TEST_ONLY`；
- freshness、nonce 和单次消费检查；
- sealed replay 明确 `hardware_authority=false`、物理风险分母增量为 0。

**拒绝条件：** nonce 已消费、boot/session 不符、过期、尚未生效、deadline 超限、签名域或 authority domain 不符。

**剩余风险：** 当前没有真机持久化介质、可信时钟、外部单调计数器或断电恢复实测；复制或替换整个本地 store 目录不能靠目录内哈希解决。因此 R1 不存在任何正向物理 admission，终局必须引入独立权限域中的单调消费后端后重新评审。

### 4.3 Confused Deputy：代理人越权

**威胁：** AI 脑、Dashboard、LLM、测试 verifier 或某个 Pi 被误当成最终物理授权者；一个宽泛 token 被用于另一角色、区域或动作。

**当前控制：**

- 科学 intent、具身 challenge、联合 permit 与本地执行 gate 分层；
- 独立 challenge/permit verifier 接口和 authority/key/signature domain；
- permit 绑定最小角色、工具、zone、option、命令包络、release 与有效期；
- R1 正向物理 admission 不存在；测试 HMAC、模拟 verifier、任意自定义 verifier 和伪造 `PermitConsumeResult` 都不能产生授权结果；
- AI 脑不能放宽 Nav2/F407、双臂控制器或本地急停。

**拒绝条件：** 角色/zone/envelope 不匹配、operator 未 armed、本地 gate 或 capability 失效、测试/模拟 verifier 被装入物理入口。

**剩余风险：** 真机服务身份、密钥保管、进程隔离与调用链尚未部署和验收。

### 4.4 Collusion：多节点串谋或共同失效

**威胁：** AI X5 与具身 X5 同时伪造 challenge/permit，或执行臂用自己的目标状态证明自己成功。

**当前控制：**

- 终局要求两台 X5 分离密钥、职责和输入状态；
- 执行回执不能独立构成科学 truth；
- 物理 closure 必须由新 raw 或独立外部 acquisition/observer 支持；
- operator、station、holder-loader 和仪器 evidence 使用不同角色和来源合同。

**拒绝条件：** 外部 outcome 缺失、证据来源与执行者同源却声明独立、所需 capability/角色未登记。

**剩余风险：** R1 只建模这些边界，没有真实独立密钥、传感器和组织流程；多个设备共享管理员、时钟、网络或供电时仍可能形成共同故障域。

### 4.5 Shared Failure：共享故障被重复计票

**威胁：** 同一 XRD raw 的视觉、数值、CPU、BPU 输出被当成四份独立证据；同一相机、标定、holder、炉次或参考库的错误被多数投票掩盖。

**当前控制：**

- Evidence DAG 记录 acquisition root、派生边和 lineage；
- independence 以物理 acquisition root 和共享 failure domain 计算，不按模型数量计算；
- `evaluate_physical_evidence_invariants` 要求新鲜、独立的物理 root；
- 联合根场景表示跨步骤持续的相关故障；ROBUST 评估使用封存根场景中的最大单场景损失，而不是加权平均掩盖尾部。

**拒绝条件：** 最小独立 acquisition 数不足、证据过期、未来时间戳、重复 root 或未知共享域。

**剩余风险：** 未登记的共同原因无法由软件自动发现；真实 failure-domain 注册需要材料、仪器和机器人专家共同冻结。

### 4.6 Provenance Drift：来源和运行身份漂移

**威胁：** 规划后更换模型/bin/preprocess、切换 CPU/BPU backend、设备重启、地图/标定变化、工具或 zone lease 变化、样品/holder 交换，仍执行旧策略或许可。

**当前控制：**

- plan 绑定 `ExperimentCase`、failure core、Evidence DAG root、closure predicate、option set 和 release；
- simulator 对 provenance drift fail-closed；
- 合同为 backend、release、boot/session、角色、zone、有效期和样品 lineage 预留严格绑定；
- adapter 当前 `TARGET_ONLY/NOT_READY`，不会用历史快照填补 live 状态。

**拒绝条件：** 任一摘要、身份、时间、capability、工具、地图或 zone 绑定不一致。

**剩余风险：** actual-backend capsule 和 live 设备身份尚未在 X5/Pi 上接入，R1 只能验证结构，不能证明传感器报告真实。

### 4.7 Authority Escalation：成熟度或权限抬升

**威胁：** 把 simulated、sealed replay 或 shadow 输出改写成 physical；把 `modeled closure` 写成真实 failure core 已关闭；把零物理样本计入风险证书。

**当前控制：**

- R1 语义事件强制 `SIMULATED_COUNTERFACTUAL`、`SIMULATED_MODELED_ONLY`；
- 强制 `hardware_touched=false`、`execution_authority=false`、`physical_closure_proven=false`；
- 强制 `physical_risk_denominator_increment=0`；
- terminal manifest 从重新读取的实际语义 ledger 派生，调用者不能自行替换 authority 字段；
- sealed replay 永远不具备物理 authority。

**拒绝条件：** 任一模拟事件声称硬件接触、执行权限、物理 closure 或非零物理分母。

**剩余风险：** 文档、视频或人工答辩仍可能脱离机器输出夸大。发布材料必须同时展示 maturity、provenance 和 denominator。

### 4.8 协同重写：ledger、anchor、terminal、release 一起改

**威胁：** 攻击者不是改单文件，而是重新计算 ledger 链、anchor、terminal 和 release，使整包内部一致。

**当前控制：**

- `ReleaseRootPin` 绑定 release manifest、ledger 文件、ledger terminal、decision receipt 和 terminal manifest；
- `verify-release-bundle` 必须读取独立保存的外部 pin、实际 ledger、项目根和三组重算逻辑 inventory；
- 无外部 pin 的 bootstrap 只生成 `UNPINNED_CANDIDATE` 并返回非零，候选根使用 `candidate_release_root.json` 命名；
- terminal manifest 构建时重新验证 ledger，减少调用者自报字段。

**拒绝条件：** 外部 pin 中任一摘要与本轮 bundle 不一致。

**剩余风险：** pin 若与输出放在同一可写目录、由同一攻击者重写，就失去独立性。终局应把固定根保存到不同权限域或不可变介质，并记录建立时间和保管人。

## 5. 统计完整性威胁

统计层重点防止“选择性计入”而不是只防文件篡改：

- matched pair 必须共享 parent block 和同一冻结 release；
- 必须完整记录 RB-VoE 与固定基线两臂，保持 ITT；
- 超过 `T_cap` 的任务按受限结果处理，不能静默删除；
- hard-safety violation 容忍度为 0；
- test 不能反向驱动同轮策略、阈值或场景修改；
- replay、simulation 和 shadow 样本不能进入物理风险分母。

当前只有统计计算合同，没有 locked 数据，因此没有风险上界、覆盖率或优于基线的实证结论。

## 6. Fail-Closed 矩阵

| 条件 | 系统结果 |
|---|---|
| 未知 schema/registry/option/failure atom | 拒绝合同 |
| Evidence DAG 循环、缺父、重复 ID 或独立性不足 | 拒绝 evidence |
| hard gate 失败或 option 不可行 | 不进入策略可行集 |
| case/plan/release/provenance 摘要漂移 | 拒绝模拟、回放或许可 |
| capability 缺失或 adapter 为 `TARGET_ONLY` | `NOT_READY/HOLD` |
| challenge/permit 签名域、角色、zone、包络不匹配 | 拒绝物理启动 |
| permit 过期、跨 session 或已消费 | 拒绝物理启动 |
| R1 语义事件缺失、乱序或 authority 抬升 | 拒绝 terminal manifest |
| bundle 与外部 pin 不一致 | 拒绝发布验证 |
| locked pair 不完整或存在硬安全违规 | 统计门禁失败 |

任何模型高置信度都不能覆盖上述拒绝。

## 7. 真机前必须补齐

1. AI/具身 X5 独立生产密钥、密钥轮换和撤销流程；
2. 可信 boot/session、时钟或单调计数器，以及 replay store 断电/回滚测试；
3. actual BPU backend、bin、preprocess、输入、runtime、CMA/slot 的不可混淆 capsule；
4. Nav2/F407、双臂工具/zone/速度/时长包络和本地 VETO 的物理 admission 接入；
5. 外部相机、holder-loader、仪器 raw 和样品 custody 的独立 evidence 链；
6. 外部 pin 的独立保管、发布签署和回滚策略；
7. 故障注入：断网、重启、power drop、旧 permit、样品交换、BPU/backend 漂移、observer 缺失；
8. locked matched-pair、零硬安全门和消融验收。

在这些项目完成之前，R1 的正确安全姿态是：**可以离线证明“软件拒绝了什么”，不能证明“真实系统已经安全执行了什么”。**
