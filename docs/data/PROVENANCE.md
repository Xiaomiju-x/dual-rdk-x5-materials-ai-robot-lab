# 数据来源、许可与本地复现

本仓库公开团队有权发布的源码、合同、脱敏证据和小型 fixture；它不把第三方数据或模型重新许可为 Apache-2.0。任何外部资源进入实验前，都应先完成来源、版本、许可、哈希和用途登记。

## 权威入口

- 机器可读资源清单：[`source_catalog.v1.json`](../../ai_brain/icmat_foundry/contracts/source_catalog.v1.json)
- 晶体结构标识与限制：[`crystal_public_cache/README.md`](../../public_evidence_data/crystal_public_cache/README.md)
- 第三方声明：[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)
- 公开发布边界：[`PUBLICATION_BOUNDARY.md`](../safety/PUBLICATION_BOUNDARY.md)

## 本地取得流程

1. 从清单中的权威 URL 取得资源，先阅读上游许可证和使用条款。
2. 固定数据版本、revision、DOI 或数据库条目标识，不使用无版本的临时镜像。
3. 在本地记录原始文件 SHA-256、下载日期、许可证和引用信息。
4. 将原始文件放入被 `.gitignore` 排除的本地数据目录；不要提交凭据、Cookie、受限 CIF、论文全文或未知许可工件。
5. 预处理生成物继承记录上游来源、脚本版本、参数和输出哈希。公开结果必须注明 `public`、`synthetic`、`replay`、`sim-only` 或团队实验来源。

Materials Project 下载脚本从环境变量 `MP_API_KEY` 读取凭据；仓库不提供默认密钥。ICDD 与 ICSD 条目只保留公开标识，不随源码分发原始或变换后的 CIF。

## 可复现清单模板

```yaml
resource_id: "provider-stable-id"
source_url: "https://provider.example/record"
version_or_revision: "fixed-version"
retrieved_at: "YYYY-MM-DD"
sha256: "64-lowercase-hex"
license: "SPDX identifier or exact upstream terms URL"
redistribution: "allowed | restricted | unknown"
purpose: "training | evaluation | visualization | reference"
claim_boundary: "What this resource does and does not prove"
```

`restricted` 或 `unknown` 的资源不得进入 Git、Release 或自动构建制品；只在本地、按上游授权使用。
