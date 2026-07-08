import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import type { ArmId } from '@/types/telemetry'

export interface Waypoint {
  id: string
  arm: ArmId
  name: string
  angles: number[]      // 6 deg
  gripper: number       // 0..100
  createdAt: number
}

const STORAGE_KEY = 'workstation.waypoints.v1'

function load(): Waypoint[] {
  try {
    const s = localStorage.getItem(STORAGE_KEY)
    if (!s) return []
    const arr = JSON.parse(s) as Waypoint[]
    if (!Array.isArray(arr)) return []
    return arr.filter((w) => w && Array.isArray(w.angles) && w.angles.length === 6)
  } catch (_e) { return [] }
}

function save(list: Waypoint[]): void {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(list)) } catch (_e) { /* quota */ }
}

let seq = Date.now()
function nextId(): string {
  seq += 1
  return `wp-${seq.toString(36)}`
}

export const useWaypointStore = defineStore('waypoints', () => {
  const list = ref<Waypoint[]>(load())

  watch(list, (v) => save(v), { deep: true })

  function add(arm: ArmId, angles: number[], gripper: number, name?: string): Waypoint {
    const wp: Waypoint = {
      id: nextId(),
      arm,
      name: name ?? `WP${list.value.filter((w) => w.arm === arm).length + 1}`,
      angles: angles.slice(0, 6),
      gripper,
      createdAt: Date.now(),
    }
    list.value = [...list.value, wp]
    return wp
  }
  function remove(id: string): void {
    list.value = list.value.filter((w) => w.id !== id)
  }
  function rename(id: string, name: string): void {
    list.value = list.value.map((w) => (w.id === id ? { ...w, name } : w))
  }
  function move(id: string, delta: -1 | 1): void {
    const i = list.value.findIndex((w) => w.id === id)
    if (i < 0) return
    const j = i + delta
    if (j < 0 || j >= list.value.length) return
    const next = list.value.slice()
    const [item] = next.splice(i, 1)
    next.splice(j, 0, item)
    list.value = next
  }
  function clear(arm?: ArmId): void {
    list.value = arm ? list.value.filter((w) => w.arm !== arm) : []
  }
  function forArm(arm: ArmId) { return computed(() => list.value.filter((w) => w.arm === arm)) }

  return { list, add, remove, rename, move, clear, forArm }
})
