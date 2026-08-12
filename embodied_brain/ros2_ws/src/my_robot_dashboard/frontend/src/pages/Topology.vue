<script setup lang="ts">
/**
 * Topology — full-page network topology visualization.
 *
 * Shows the three-system distributed cluster (AI Brain X5 + Embodied
 * Brain X5 + dual arms + dual furnaces + cloud LLM) as an animated 3D
 * node graph with live data-flow particles.
 *
 * Marquee answer-killer for the competition demo.
 */
import { ref } from 'vue'
import NetworkTopology3D from '@/components/premium/NetworkTopology3D.vue'
import KineticTitle from '@/components/premium/KineticTitle.vue'
import BorderBeam from '@/components/premium/BorderBeam.vue'
import CountUp from '@/components/premium/CountUp.vue'
import { useTelemetryStore } from '@/stores/telemetry'

const telemetry = useTelemetryStore()
const cinematic = ref(true)
</script>

<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1 class="page-title">
          <KineticTitle text="Topology · 系统拓扑" gradient="aurora" />
        </h1>
        <p class="page-subtitle">
          双 RDK X5 + 双 myCobot + 双烧结炉 + 云端 R1 异构集群活地图 · click 节点聚焦 ·
          实时数据流粒子 · bloom 后处理
        </p>
      </div>
      <div class="header-actions">
        <button class="btn" :class="{ 'btn-primary': cinematic }" @click="cinematic = !cinematic">
          {{ cinematic ? '✦' : '◌' }} Cinematic
        </button>
      </div>
    </header>

    <div class="topo-stage card-floating">
      <BorderBeam :duration="16" :size="280" :radius="22" :colorFrom="'rgba(124, 58, 237, 0.85)'" :colorTo="'rgba(34, 211, 238, 0.85)'" />
      <NetworkTopology3D :height="560" :cinematic="cinematic" />
    </div>

    <div class="topo-cards">
      <div class="topo-card card-elevated">
        <span class="card-tag" style="background: linear-gradient(135deg, #a78bfa, #7c3aed);">CLOUD</span>
        <div class="card-title">DeepSeek R1 reasoner</div>
        <div class="card-body">
          15-30s 深度推理云端兜底, 当 AI 脑本地 9 LLM 任意一条不确定时<br/>
          升级给 R1 二次验证.
        </div>
      </div>
      <div class="topo-card card-elevated">
        <span class="card-tag" style="background: linear-gradient(135deg, #2563eb, #06b6d4);">AI BRAIN</span>
        <div class="card-title">RDK X5 8G · 实验室</div>
        <div class="card-body">
          9 本地 LLM (5 BPU slot + 4 CPU llama-server) + 5 轻量 BPU 感知<br/>
          53 Flask routes · /api/predict /api/predict_stream /api/dispatch_task
        </div>
      </div>
      <div class="topo-card card-elevated">
        <span class="card-tag" style="background: linear-gradient(135deg, #06b6d4, #10b981);">EMBODIED</span>
        <div class="card-title">RDK X5 8G · 车载</div>
        <div class="card-body">
          ROS2 Humble + Nav2 + slam_toolbox + 8 BPU 感知节点<br/>
          实测 <CountUp :end="telemetry.observedHz" :decimals="1" suffix=" Hz" /> WebSocket · uptime <CountUp :end="telemetry.packet?.heartbeat.uptime_s ?? 0" suffix="s" />
        </div>
      </div>
      <div class="topo-card card-elevated">
        <span class="card-tag" style="background: linear-gradient(135deg, #059669, #10b981);">DUAL ARM</span>
        <div class="card-title">myCobot 280-Pi ×2</div>
        <div class="card-body">
          Pi 4B 2GB · 6 关节 · 工厂工位 (不装车顶)<br/>
          USB cam 1280×720 + AprilTag 6D pose · MG996R 夹爪
        </div>
      </div>
      <div class="topo-card card-elevated">
        <span class="card-tag" style="background: linear-gradient(135deg, #d97706, #e11d48);">FURNACE</span>
        <div class="card-title">烧结炉 ×2</div>
        <div class="card-body">
          PV/SV/MV 7段 OCR (OpenCV + Qwen-VL fallback)<br/>
          I1-I6 4 类报警 · TTS + 邮件 + 微信 dispatcher
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.page {
  max-width: 1680px;
  margin: 0 auto;
  display: flex; flex-direction: column;
  gap: 18px;
  animation: fadeUp 0.42s var(--ease-out-quint) both;
  height: 100%;
}
.page-header { display: flex; justify-content: space-between; align-items: flex-end; }
.header-actions { display: flex; gap: 8px; }

.topo-stage {
  position: relative;
  flex: 1;
  min-height: 0;
  overflow: hidden;
  border-radius: 22px;
  padding: 0;
}

.topo-cards {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
}
@media (max-width: 1280px) { .topo-cards { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 720px)  { .topo-cards { grid-template-columns: repeat(2, 1fr); } }

.topo-card {
  padding: 14px 16px;
  display: flex; flex-direction: column; gap: 6px;
  position: relative;
}
.card-tag {
  display: inline-flex; align-items: center; justify-content: center;
  align-self: flex-start;
  padding: 2px 10px;
  border-radius: 999px;
  color: white;
  font-size: 0.62rem; font-weight: 700;
  letter-spacing: 0.1em;
}
.card-title { font-size: 0.84rem; font-weight: 600; color: var(--ink-primary); }
.card-body { font-size: 0.7rem; color: var(--ink-tertiary); line-height: 1.5; }
</style>
