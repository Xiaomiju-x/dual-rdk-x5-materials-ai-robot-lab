# X5-ICMat Foundry X5 真机阶段手册

状态：`PC_ACCEPTED_BOARD_PENDING`

本手册只在用户明确确认 AI 脑 X5 已上电后执行。它不修改 PC 网络，不扫描
设备，不覆盖 Dashboard/五端口/原模型槽，不创建 systemd 服务。

## 1. 权威输入

- staging 包：`x5-icmat-foundry-50model-x5-staging-c5fa215a58168c0c.zip`
- SHA-256：`c5fa215a58168c0cb7274c2b1cf6d66bcd0f3c1e70d3f4cf13749e9b57dafb52`
- 安装模式：`MANUAL_STAGING_ONLY`
- 自动启动：`false`
- 生产覆盖：`false`
- RB-VoE：`DEPLOYED_OFF`

## 2. 上板前只读核验

在 X5 终端只读执行并保存输出：

```bash
set -u
hostname
id
date -Is
uname -a
cat /etc/os-release
grep -E '^(MemAvailable|SwapTotal|SwapFree|CmaTotal|CmaFree):' /proc/meminfo
ps -eo pid,ppid,rss,stat,comm,args --sort=-rss | head -n 40
ss -ltnp | grep -E ':(8888|8080|8081|5000|5001)\b' || true
(command -v hrut_somstatus >/dev/null && hrut_somstatus) || true
python3 - <<'PY'
try:
    from hobot_dnn import pyeasy_dnn
    print('HOBOT_DNN_IMPORT=PASS', pyeasy_dnn)
except Exception as exc:
    print('HOBOT_DNN_IMPORT=FAIL', type(exc).__name__, str(exc))
PY
```

随后仅以 GET 检查既有服务。失败只记录，不重启服务：

```bash
for url in \
  http://127.0.0.1:8888/api/health \
  http://127.0.0.1:8080/api/camera/status \
  http://127.0.0.1:8081/api/camera/status \
  http://127.0.0.1:5000/api/health_check \
  http://127.0.0.1:5001/api/health_check
do
  printf '%s\n' "$url"
  curl -fsS --max-time 3 "$url" || true
  printf '\n'
done
```

身份、生产哈希、端口或 runtime 不匹配时停止上板；不得通过重启冻结服务来
“修复”候选。

## 3. 隔离解包

先在 X5 上复核 staging ZIP：

```bash
sha256sum x5-icmat-foundry-50model-x5-staging-c5fa215a58168c0c.zip
```

仅解到普通用户目录：

```bash
release_id=x5-icmat-foundry-50model-c5fa215a58168c0c
install_root="$HOME/icmat_foundry_finals/releases/$release_id"
mkdir -p "$install_root"
unzip -q x5-icmat-foundry-50model-x5-staging-c5fa215a58168c0c.zip -d "$install_root"
python3 - "$install_root" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
policy = json.loads((root / 'release_policy.json').read_text())
manifest = json.loads((root / 'release_manifest.json').read_text())
assert policy['automatic_start'] is False
assert policy['production_overwrite'] is False
assert policy['rb_voe_state'] == 'DEPLOYED_OFF'
assert manifest['kind'] == 'x5-staging'
assert len(manifest['models']) == 38
print('STAGING_CONTRACT=PASS')
PY
```

禁止使用 `sudo cp` 写入现有模型目录，禁止修改 `start_x5.sh`、Dashboard、
五端口、systemd 或原 BPU slot。

## 4. 真机验证顺序

1. 先用一个小型 CPU 模型验证 X5 CPU loader、固定输入、输出和进程退出；
2. 再用一个小型 BPU 模型验证 `pyeasy_dnn`/actual backend、CMA 增量和退出恢复；
3. 按任务域逐项验证 14 个 CPU 主模型和 24 个 BPU 主模型；
4. 每次只启动一个 candidate worker；不得调用生产 `bpu_slot_manager.switch()`；
5. 每项保存模型 SHA、输入 SHA、输出、backend、加载/推理/退出时间、
   MemAvailable/CmaFree 前中后值和五端口 GET 结果；
6. 候选异常只标记该模型 `BOARD_REJECTED` 或 `BOARD_EXPERIMENTAL`，继续保留
   PC 编译证据，不影响生产服务。

`pyeasy_dnn` 的可靠 CMA 释放边界是候选进程退出。只允许结束本次候选自己
启动的 PID，不杀生产 worker，不删除共享锁。

## 5. 三套 BPU LLM

`F-LLM-03/04/05` 必须分别验证，不能混用不同 content hash 的两段 bin。
每个模型包含：

- part1：手写 Qwen2 第 0-11 层；
- part2：第 12-23 层；
- CPU embedding、final norm 和 tokenizer；
- 两段 PC OpenExplorer 成功日志与 SHA-256。

板端必须补做 HF/CPU 参考与 actual INT8 BPU 的固定任务语义差分。若出现类似
历史通用生成标点塌缩，只允许保留通过验证的固定结构化任务，不得宣传通用
自由生成。内存/CMA 不够时按模型切换，不允许三模型或 49 个 X5 模型常驻。

## 6. 成功判据

单模型达到 `X5_VALIDATED` 必须同时满足：

- 模型哈希与 staging manifest 一致；
- actual backend 与注册表一致；
- 固定输入输出有限且任务合同有效；
- BPU 模型实际加载并推理，不以 mapper 估算替代；
- candidate 进程退出后 CMA/内存回到允许范围；
- 8888/8080/8081/5000/5001 和相机所有权未被改变；
- 没有新增开机服务、生产依赖或控制权限。

全部模型逐项完成前，只能说“50 模型 PC 候选库已完成、X5 真机验收进行中”。

## 7. RB-VoE 与回滚

50 模型真机回执生成后，才允许执行一次 `FleetAudit PASSIVE_ONESHOT`。它只读
审计副本并退出，不阻断模型，不改变 Dashboard 或预测结果。执行后状态恢复
`DEPLOYED_OFF`，不实现 `ENFORCE`。

最短回滚是停止候选自身进程并停止使用该 release 目录。由于没有服务注册、
没有生产覆盖和自动启动，冻结系统不需要回滚代码或重启服务。
