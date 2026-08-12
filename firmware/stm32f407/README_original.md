# STM32F407 Lift Stage Bench Firmware

Lift serial diagnostics are selected by `TEST_MODE 3` in `App/main.c`.
The current default is `TEST_MODE 0`: normal ROS2/F407 0xAA55 chassis firmware.

## Wiring Used

- Host USB-TTL command link: `USART2`, `PD5=TX`, `PD6=RX`, `115200 8N1`
- Lift stepper driver UART: `UART5`, `PC12=TX`, `PD2=RX`, `115200 8N1`
- Lift stepper enable: `PD10`
- Lift stepper pulse: `PC9 = TIM8_CH4 AF3`
- Lift stepper direction: `PD7`
- Electric pushrod relay IN1: `PC13`, active high
- Electric pushrod relay IN2: `PC0`, active high
- Servo PWM for pushrod swing: `PB8 = TIM4_CH3 AF2`, 50 Hz
- Electromagnet driver: `PE0`, active high

Electric pushrod relay logic from `STM32F407引脚接线(2).docx`:

- Extend: `IN1=0`, `IN2=1`
- Retract: `IN2=1`, delay, then `IN1=1`
- Stop/off: disconnect `IN2` first, then set `IN1=0`; no guard delay is used
  on the stop path

## Build And Flash

Open `embodied_brain/stm32_f407/a.uvprojx` in Keil.

Build target `Target 1`, then flash the generated firmware to STM32F407.

The project file already includes the new `App/bsp_lift.c`.

Current mode:

```c
#define TEST_MODE       0
```

`TEST_MODE 5` is retained only as the frozen first-round video fallback. It
bypasses X5 serial commands and auto-runs the degraded sequence. Use
`TEST_MODE 3` only for attended ASCII lift diagnostics, then return to
`TEST_MODE 0` for the normal X5/ROS2 stack.

## Normal-Mode Safety Protocol

`TEST_MODE 0` now has a firmware-level estop latch shared with the ROS2 serial
bridge:

- down `0x10 EMERGENCY_STOP`: latches stop, zeros wheel targets, stops lift and
  pushrod, and switches the electromagnet off
- down `0x11 CLEAR_ESTOP`: the only command that clears the firmware latch
- up `0x03 SAFETY_STATE`: `estop_latched`, `emergency_active`, and the blocked
  command counter
- while latched, lift target/home and electromagnet ON return ACK status `3`;
  electromagnet OFF remains allowed
- before the first heartbeat or after the command link is stale, motion/actuator
  commands return ACK status `4`; clearing estop also requires a fresh heartbeat
- after the 1 s heartbeat timeout, an in-flight lift or fixture sequence is
  cancelled, the lift pulse is forced low, both pushrod relay inputs are made
  safe, and the electromagnet is switched off; motion does not resume when the
  heartbeat returns and must be commanded again

Finite lift pulses and the normal-mode pick/place/home fixture timings are
cooperatively serviced from the main loop. A received emergency frame is parsed
while a multi-minute 10 pps move is in progress instead of waiting for that move
to finish. `SET_LIFT_HEIGHT` and `LIFT_HOME` ACK now mean "accepted/started", not
physical completion; callers that require completion must continue to use lift
telemetry and a bounded arrival timeout.

Safety assumptions for this firmware:

- lift position remains an open-loop pulse count; there is still no home/top
  limit or encoder confirmation
- lift stop keeps the stepper enabled for static holding torque but emits no
  further pulses
- magnet OFF on estop/link loss follows the existing de-energize fail-safe
  policy, even though a held object can drop; the fixture needs a mechanical
  retention zone before unattended use
- pushrod direction changes retain the existing bounded `80ms` relay guard;
  pushrod stop itself is immediate, so emergency-command handling has no
  multi-second actuator wait
- `TEST_MODE 4/5` are standalone modes without the host heartbeat contract;
  their frozen continuous-spin/video timing remains unchanged

The ROS `/clear_estop` service waits for the F407 `CLEAR_ESTOP` ACK even when
ordinary actuator ACK waits are disabled. Run the no-motion interlock test only
after flashing the current `TEST_MODE 0` build:

```bash
python3 ~/tools/f407_link_test.py --verify-estop-interlock --require-ack \
  --report ~/f407_interlock_evidence/latest.json
```

This leaves estop latched by default. Add `--clear-estop` only after the operator
has confirmed the area is safe.

Standalone spin-test parameters in `App/main.c` are kept for `TEST_MODE 4`:

```c
#define LIFT_SPIN_TARGET_PPS        6400
#define LIFT_SPIN_RAMP_START_PPS    200
#define LIFT_SPIN_RAMP_STEP_PPS     200
#define LIFT_SPIN_RAMP_HOLD_MS      250U
```

Set `TEST_MODE` to `4` only when a no-X5 standalone spin test is needed. Set `TEST_MODE` back to `0` for normal chassis firmware.

## Serial Protocol

After boot, the board prints help text on the host USB-TTL serial port.

Commands are ASCII lines ending with `\n`:

```text
PING
STATUS
HELP
EN 0
EN 1
ZERO
SAFEZERO [bottom_margin_steps] [pps]
STOP
JOGUP [steps] [pps]
JOGDOWN [steps] [pps]
STEPUP [steps] [pps]
STEPDOWN [steps] [pps]
UP [steps] [pps]
DOWN [steps] [pps]
GOTO <target_steps> [pps]
SPEED <pps>
CYCLE [steps] [pps]
MAG ON|OFF
ACT EXT|RET|STOP
SERVO LEFT|HOME|RIGHT|US <us>
AUTO [steps] [pps] [extend_ms] [retract_ms] [hold_pps]
DEMO [lift_steps] [pps] [nav_ms]
RAW <en_hi> <dir_hi> <pul_hi>
BITPULSE <en_hi> <dir_hi> <pulses> <half_ms>
```

Defaults are now in lift-stage GPIO-safe segmented mode: `steps=25`, `pps=10` in firmware. Finite lift moves use a cooperative GPIO edge scheduler on `PC9`, matching the validated `BITPULSE half-ms=50` waveform instead of TIM8 continuous output. Longer finite moves are internally split into `25`-step segments with `500ms` dwell while keeping `EN=1`. `JOGUP/JOGDOWN/STEPUP/STEPDOWN` default to `25` steps at `10` pps and cap manual jogs to `500` steps and `10` pps. This is intentional: the loaded vertical axis held under slow GPIO pulses but blue-faulted/dropped under longer continuous moves.

`SAFEZERO` is used from the physical bottom stop only after short-step tests are stable: it moves upward by a small bottom margin, then defines that safe position as `pos=0`. `AUTO` should run between this safe bottom and the safe top, not between hard mechanical stops. The default SAFEZERO margin is `500` steps at `10` pps. The default AUTO speed is temporarily limited to `10` pps, and continuous top-hold pulses are forcibly disabled in firmware. Override these only after measuring the real travel and confirming the driver no longer enters blue protection.

While `AUTO`, `SAFEZERO`, `CYCLE`, `DEMO`, or `BITPULSE` is running, the ASCII parser remains live and `STATUS` keeps reporting progress. `STOP` cancels the sequence and stops lift/pushrod motion. Manual motion commands are rejected while the lift is busy; do not stack commands. `BITPULSE` returns `OK bitpulse started`; completion is visible as `busy=0` in `STATUS`.

## First Bring-Up Sequence

Run these from the car X5 after flashing. The validated port on 2026-07-06 14:05 was `/dev/ttyUSB1`; `/dev/ttyUSB0` was a binary sensor stream in that boot.

```bash
# Free the STM32 serial port from the normal chassis bridge first.
pkill -TERM -f 'ros2 launch my_robot_bringup full.launch.py' || true
pkill -TERM -f 'serial_f407_node' || true

python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB1 ping
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB1 status
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB1 en 1
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB1 stepup --steps 25 --pps 10
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB1 status

# Repeat STEPUP only after the platform holds for at least 10 seconds.
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB1 up --steps 100 --pps 10

# Aux outputs, test one at a time with the bottle clear of danger.
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB1 mag on
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB1 mag off
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB1 act ext --seconds 2
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB1 act ret --seconds 2
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB1 servo home
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB1 servo left
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB1 servo home

# Full bottle-transfer flow only after the lift can move and hold repeatedly.
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB1 auto --steps 500 --pps 10 --hold-pps 0

# Graceful degradation for the first-round video:
# left home -> right pickup -> magnet on -> left carry -> nav delay -> right release
# -> small lift up/down proof -> magnet off -> left home.
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB1 demo --steps 50 --pps 10 --nav-ms 3000
```

USB order can swap between boots. The STM32F407 lift-stage port is the one that returns `PING -> PONG`; the other port emits binary sensor data and should not be used for lift-stage ASCII commands.

Expected first-test output:

```text
PING -> PONG
STATUS -> STATUS en=0 busy=0 pos=0 rem=0
EN 1 -> static hold with no blue-driver protection
STEPUP 25 10 -> short upward move using GPIO-safe pulses, then hold for 10 seconds without drop
UP 100 10 -> internally segmented as 25+25+25+25 with dwell
```

2026-07-06 full mechanism result: user confirmed correct direction, bottom-to-top travel, and return to bottom succeeded with X5/F407 control.

2026-07-06 AUTO top-hold note: early tests tried continuous upward `top-hold` pulses (`250` then `1600` pps) to prevent sliding, but this is not safe near the top because it keeps moving upward during the pushrod/servo dwell. This approach is superseded: current default `hold_pps=0`, relying on stepper enable/static holding plus conservative travel. Electric pushrod extend/retract and servo left/home directions were confirmed correct by the user. Electromagnet release is still a hardware wiring/control issue: F407 reports `mag=0`, but the bottle remains attached while system power is on and drops only when the electromagnet power is cut.

2026-07-06 top-collision safety note: a full `36000` step AUTO run reached the mechanical top stop, the stepper driver showed a blue fault/limit indication, then holding torque was lost and the lift dropped. Treat this as driver protection after hard stall, not a normal homing method. Default AUTO travel was reduced to `30000` steps and speed to `1600pps`; do not run `36000` again until a top limit switch or a calibrated soft top with 2-5mm margin is in place.

2026-07-06 soft-bottom safety note: do not use the hard bottom stop as software zero for unattended AUTO. From the physical bottom, run `SAFEZERO 500 300` first; this moves upward by a small bottom margin and then defines that safe position as `pos=0`. AUTO travel must be measured from safe bottom upward after the long-stroke stall is solved. Tune `SAFEZERO` margin and `AUTO --steps` on the real fixture so both ends keep mechanical clearance.

2026-07-06 hold/driver-protection note: `EN 1` locks the motor at rest, so enable polarity and basic wiring are correct. However, a short `UP 1000 600` still caused drop/blue-driver protection without hitting the top stop. Finite lift moves now use trapezoidal acceleration/deceleration in the TIM8 ISR (`LIFT_RAMP_MIN_PPS=120`, `LIFT_RAMP_MAX_STEPS=1200`) to avoid hard stopping the vertical load. First post-flash test should be `UP 500 200`, not AUTO.

2026-07-06 calibrated bottom margin: after reflashing the ramped firmware, `ZERO` at the physical bottom followed by `UP 500 200` stopped cleanly with `STATUS en=1 busy=0 pos=500 rem=0`; the user confirmed the platform held position and the height is suitable. Treat `500` steps as the current bottom safety margin.

2026-07-06 long-stroke stall note: from the safe bottom zero, `UP 3000 300` reached `STATUS en=1 busy=0 pos=3000 rem=0` in firmware, but the user clarified the physical lift did not move and remained about 2 cm above the bottom. The stepper driver showed blue protection/lock. Treat this as a missed-motion/driver-stall condition, not a top calibration result. The top stroke is still uncalibrated. AUTO default is temporarily limited to `500` steps at `200` pps until segmented jog tests find the reliable range. Continuous `top-hold` remains disabled (`0`) because it is upward motion and can push into the top stop during dwell.

2026-07-07 bottom-start stall note: after power-cycling at the physical bottom, `UP 100 100` completed in firmware but produced no visible lift motion; the next `UP 400 150` produced no visible lift motion and put the driver into blue protection at the bottom. This points to a loaded hard-bottom start/stall condition rather than a communication fault. Finite-motion start speed was reduced from `120` pps to `20` pps, default finite speed from `800` to `120` pps, SAFEZERO from `300` to `60` pps, and `JOGUP/JOGDOWN` commands were added. After reflashing, test only `JOGUP 200 60` from the bottom; do not run long `UP`, `SAFEZERO`, or `AUTO` until the jog can visibly leave the bottom without blue protection.

2026-07-08 loaded-hold regression note: static `EN 1` at the 2cm soft-bottom holds, but loaded TIM8 finite moves can still end with a drop and blue driver protection. GPIO bit-bang diagnostics were stable at `BITPULSE 0 1 1/5/20/50 50`, so finite lift movement is now routed through blocking GPIO-safe pulses. Single `STEPUP 50 10` was normal; single `STEPUP 75 10` dropped/blue-faulted; four separate `STEPUP 25 10` segments reached `pos=100` and held normally. The firmware therefore segments all finite moves into 25-step chunks with 500ms dwell. Global TEST_MODE=3 debug motion cap remains `10` pps, busy-motion commands are rejected, continuous `top-hold` pulses and nonzero `SPEED` are forcibly disabled. If segmented `UP 100 10` still drops or blue-faults after this firmware, stop software testing and adjust hardware first: disable driver half-current/idle-current reduction, raise driver current one step, lower microstepping to 1/8 or 1/4, and re-check A+/A-/B+/B- phase wiring.

2026-07-08 first-round video degradation: because the vertical lift still drops/blue-faults under practical lift movement, the contest-video path is explicitly degraded. Servo home is now the left transport position (`LIFT_SERVO_HOME_US=LIFT_SERVO_LEFT_US`). The calibrated left/home pulse is `2300us`; calibrated right pickup/release is `1400us`. `MAG ON` successfully holds the bottle-top iron sheet, and `MAG OFF` releases/drops the bottle. `TEST_MODE 5` now auto-runs the video fallback on boot: magnet off, actuator full retract/home (`11s`) to recover from interrupted runs before any lateral servo move, home-left, servo right to pickup, actuator full extend (`10s`), magnet on, actuator full retract (`10s`), servo left to carry, 2.5s nav placeholder, segmented slow servo move to the right release side (`2100->1900->1700->1550->1400us`, then 5s settle), ramped lift-fault/top attempt (`120->240->480->800->1200pps`, then `10s` at 1200pps, stop), actuator full extend (`10s`), magnet off/drop, actuator full retract/stop (`10s`), then servo home-left. This avoids relying on the lift to carry the bottle upward. Treat this as a video fallback, not the final lab workflow. If the actuator does not have built-in end-limit switches, replace these timed full-stroke waits with limit or current feedback before long unattended operation.

Only after direction and motion are confirmed, increase `--steps` gradually until the platform reaches the top.

## No-Motion Diagnostics

If serial status changes but the 42 stepper does not move, rebuild and flash the current firmware, then run GPIO bit-bang diagnostics:

```bash
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB0 bitpulse 0 1 --pulses 300 --half-ms 5
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB0 bitpulse 1 1 --pulses 300 --half-ms 5
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB0 bitpulse 0 0 --pulses 300 --half-ms 5
python3 ~/tools/lift_stage_test.py --port /dev/ttyUSB0 bitpulse 1 0 --pulses 300 --half-ms 5
```

This bypasses TIM8 alternate-function output and toggles `PC9` as a normal GPIO. If one of these four cases moves the motor, keep that `en_hi` level and direction polarity. If none moves, check driver power, common GND, PUL+/PUL- wiring, EN wiring, and whether the stepper driver accepts 3.3V pulse inputs.

## Safety Notes

- There are no limit switches in this first pass. Do not run `AUTO` unattended; use `SAFEZERO` from the hard bottom first, then tune `--steps` from that safe bottom position.
- Keep power accessible during first motion tests.
- If `UP` moves downward, change `LIFT_DIR_INVERT` in `App/bsp_lift.h` from `0` to `1`.
- If the stepper driver enable polarity is reversed, change `LIFT_EN_ACTIVE_LOW` in `App/bsp_lift.h`.
- `STOP` stops lift motion, cancels `AUTO`, and stops the pushrod. It intentionally does not force `MAG OFF`, to avoid dropping a held bottle unexpectedly.
