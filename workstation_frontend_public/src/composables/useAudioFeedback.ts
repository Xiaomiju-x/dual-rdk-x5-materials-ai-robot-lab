/**
 * Tiny Web Audio synth — produces short crystalline blips for UI events.
 * No samples, no library: just oscillator + gain envelope. Cheap.
 *
 * AudioContext is created lazily on first sound (browsers gate autoplay
 * until a user gesture).
 */
import { useSettingsStore } from '@/stores/settings'

type SoundKind = 'click' | 'navigate' | 'warn' | 'err' | 'success' | 'ping'

let ctx: AudioContext | null = null

function getCtx(): AudioContext | null {
  if (typeof window === 'undefined') return null
  if (!ctx) {
    try { ctx = new (window.AudioContext || (window as any).webkitAudioContext)() } catch { return null }
  }
  // resume if suspended (browser autoplay policy)
  if (ctx.state === 'suspended') ctx.resume().catch(() => {})
  return ctx
}

interface ToneSpec {
  freqs: number[]    // chord frequencies (Hz)
  attack: number     // s
  decay: number      // s
  peak: number       // 0..1
  wave: OscillatorType
}

const PALETTE: Record<SoundKind, ToneSpec> = {
  click:    { freqs: [880],          attack: 0.002, decay: 0.05, peak: 0.04, wave: 'sine' },
  navigate: { freqs: [523.25, 783.99], attack: 0.005, decay: 0.10, peak: 0.05, wave: 'sine' },
  warn:     { freqs: [349.23, 440.00], attack: 0.005, decay: 0.18, peak: 0.07, wave: 'triangle' },
  err:      { freqs: [220, 196],     attack: 0.005, decay: 0.30, peak: 0.09, wave: 'sawtooth' },
  success:  { freqs: [523.25, 659.25, 783.99], attack: 0.003, decay: 0.18, peak: 0.06, wave: 'sine' },
  ping:     { freqs: [1318.51],      attack: 0.002, decay: 0.06, peak: 0.03, wave: 'sine' },
}

export function useAudioFeedback() {
  const settings = useSettingsStore()

  function play(kind: SoundKind) {
    if (!settings.sound) return
    const ac = getCtx()
    if (!ac) return
    const spec = PALETTE[kind]
    const now = ac.currentTime
    for (const f of spec.freqs) {
      const osc = ac.createOscillator()
      const gain = ac.createGain()
      osc.type = spec.wave
      osc.frequency.setValueAtTime(f, now)
      gain.gain.setValueAtTime(0, now)
      gain.gain.linearRampToValueAtTime(spec.peak, now + spec.attack)
      gain.gain.exponentialRampToValueAtTime(0.0001, now + spec.attack + spec.decay)
      osc.connect(gain).connect(ac.destination)
      osc.start(now)
      osc.stop(now + spec.attack + spec.decay + 0.02)
    }
  }

  return { play }
}
