#!/bin/bash
# XRD数值线 → RDK X5 一键部署脚本
# 用法: bash deploy_to_x5.sh [X5_IP]
# 示例: bash deploy_to_x5.sh 192.0.2.103

set -e
X5_IP="${1:-192.0.2.103}"
X5_USER="sunrise"
X5_DIR="/home/rdk/xrd1"
PROJ="$(cd "$(dirname "$0")" && pwd)"

echo "=== XRD数值线部署 → $X5_USER@$X5_IP:$X5_DIR ==="

# 1. 创建远程目录
echo "[1/5] 创建目录..."
ssh $X5_USER@$X5_IP "mkdir -p $X5_DIR/{static,crystal_data,xrd_knowledge/embeddings,data/raw_files}"

# 2. 核心代码
echo "[2/5] 传输核心代码..."
scp "$PROJ/web_demo.py" \
    "$PROJ/bpu/infer_with_llm.py" \
    "$PROJ/rag_engine.py" \
    "$PROJ/knowledge_base.py" \
    $X5_USER@$X5_IP:$X5_DIR/

# 3. 数据文件
echo "[3/5] 传输数据..."
scp "$PROJ/simulated_patterns.json" \
    "$PROJ/outputs/conformal_params.json" \
    $X5_USER@$X5_IP:$X5_DIR/

# BPU模型(如果convert_workspace有.bin)
for f in "$PROJ/bpu/convert_workspace/model_output/"*.bin; do
    [ -f "$f" ] && scp "$f" $X5_USER@$X5_IP:$X5_DIR/
done
for f in "$PROJ/bpu/convert_workspace_fine/model_output_fine/"*.bin; do
    [ -f "$f" ] && scp "$f" $X5_USER@$X5_IP:$X5_DIR/
done

# 标准化参数
scp "$PROJ/outputs/features/norm_params.json" $X5_USER@$X5_IP:$X5_DIR/
scp "$PROJ/outputs/features/label_map.json" $X5_USER@$X5_IP:$X5_DIR/
scp "$PROJ/outputs/features_fine/norm_params.json" $X5_USER@$X5_IP:$X5_DIR/norm_params_fine.json
scp "$PROJ/outputs/features_fine/label_map.json" $X5_USER@$X5_IP:$X5_DIR/label_map_fine.json

# RAG知识库
scp "$PROJ/xrd_knowledge/embeddings/chunks.json" $X5_USER@$X5_IP:$X5_DIR/xrd_knowledge/embeddings/
scp "$PROJ/xrd_knowledge/embeddings/vectors.npy" $X5_USER@$X5_IP:$X5_DIR/xrd_knowledge/embeddings/

# 4. 静态资源 + CIF + 部署配置
echo "[4/5] 传输资源..."
scp "$PROJ/static/manifest.json" "$PROJ/static/sw.js" $X5_USER@$X5_IP:$X5_DIR/static/ 2>/dev/null || true
# CIF文件
for f in "$PROJ/../crystal_data/"*.cif "d:/桌面/xrd/crystal_data/"*.cif; do
    [ -f "$f" ] && scp "$f" $X5_USER@$X5_IP:$X5_DIR/crystal_data/
done
# systemd
scp "$PROJ/xrd-demo.service" $X5_USER@$X5_IP:$X5_DIR/
scp "$PROJ/setup_x5.sh" $X5_USER@$X5_IP:$X5_DIR/

# 5. .raw样品文件
echo "[5/5] 传输样品..."
scp "$PROJ/data/raw_files/"*.raw $X5_USER@$X5_IP:$X5_DIR/data/raw_files/ 2>/dev/null || true

echo ""
echo "=== 部署完成 ==="
echo "启动: ssh $X5_USER@$X5_IP 'cd $X5_DIR && python3 web_demo.py --port 8080'"
echo "或一键: ssh $X5_USER@$X5_IP 'bash $X5_DIR/setup_x5.sh'"
