#!/bin/bash
# XRD AI系统 X5一键部署脚本
# 用法: scp setup_x5.sh rdk@x5:~/xrd1/ && ssh rdk@x5 "bash ~/xrd1/setup_x5.sh"

set -e
cd /home/rdk/xrd1

echo "=== XRD AI系统 X5部署 ==="

# 1. 安装systemd服务
echo "[1/4] 安装systemd服务..."
sudo cp xrd-demo.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable xrd-demo.service
echo "  ✅ 服务已安装并设为开机自启"

# 2. 创建必要目录
echo "[2/4] 检查目录..."
mkdir -p xrd_knowledge/embeddings crystal_data data/raw_files
echo "  ✅ 目录就绪"

# 3. 检查关键文件
echo "[3/4] 检查文件..."
for f in web_demo.py infer_with_llm.py; do
    if [ -f "$f" ]; then echo "  ✅ $f"; else echo "  ❌ 缺少 $f"; fi
done
for f in simulated_patterns.json conformal_params.json; do
    if [ -f "$f" ]; then echo "  ✅ $f"; else echo "  ⚠️ 可选: $f"; fi
done

# 4. 启动服务
echo "[4/4] 启动服务..."
sudo systemctl restart xrd-demo.service
sleep 3
if curl -s http://localhost:8080 > /dev/null 2>&1; then
    echo "  ✅ 系统已启动: http://$(hostname -I | awk '{print $1}'):8080"
else
    echo "  ❌ 启动失败, 查看日志: journalctl -u xrd-demo -f"
fi

echo ""
echo "=== 部署完成 ==="
echo "管理命令:"
echo "  启动: sudo systemctl start xrd-demo"
echo "  停止: sudo systemctl stop xrd-demo"
echo "  日志: journalctl -u xrd-demo -f"
echo "  状态: sudo systemctl status xrd-demo"
