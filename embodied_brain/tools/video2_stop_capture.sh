#!/usr/bin/env bash
set -eo pipefail
set +u

base="$HOME/video2_sessions"
session="${1:-}"
if [ -z "$session" ]; then
  if [ -f "$base/current_session.txt" ]; then
    session="$(cat "$base/current_session.txt")"
  else
    echo "ERR no current session; pass session path explicitly" >&2
    exit 2
  fi
fi

touch "$session/STOP"
if [ -f "$session/recorder.pid" ]; then
  pid="$(cat "$session/recorder.pid")"
  for _ in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 1
  fi
fi

if [ -f "$session/fps.txt" ]; then
  fps="$(cat "$session/fps.txt")"
else
  fps="${VIDEO2_FPS:-2}"
fi
frames="$session/frames_grid/frame_%05d.png"

if command -v ffmpeg >/dev/null 2>&1 && ls "$session"/frames_grid/frame_*.png >/dev/null 2>&1; then
  ffmpeg -y -hide_banner -loglevel warning -framerate "$fps" -i "$frames" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 "$session/video2_data_grid.mp4" || true

  # Crop panels from the 1600x900 grid. Panel outer boxes are 500x380.
  ffmpeg -y -hide_banner -loglevel warning -i "$session/video2_data_grid.mp4" \
    -filter:v "crop=500:380:20:74" "$session/video2_slam_lidar.mp4" || true
  ffmpeg -y -hide_banner -loglevel warning -i "$session/video2_data_grid.mp4" \
    -filter:v "crop=500:380:550:74" "$session/video2_depth_scan.mp4" || true
  ffmpeg -y -hide_banner -loglevel warning -i "$session/video2_data_grid.mp4" \
    -filter:v "crop=500:380:1080:74" "$session/video2_lab_fsd_shadow.mp4" || true
  ffmpeg -y -hide_banner -loglevel warning -i "$session/video2_data_grid.mp4" \
    -filter:v "crop=500:380:20:490" "$session/video2_ai_brain.mp4" || true
  ffmpeg -y -hide_banner -loglevel warning -i "$session/video2_data_grid.mp4" \
    -filter:v "crop=500:380:550:490" "$session/video2_vision_bev.mp4" || true
  echo "VIDEO2_EXPORT_DONE"
else
  echo "VIDEO2_FRAMES_ONLY: ffmpeg missing or no frames found"
fi

echo "$session"
ls -lh "$session" 2>/dev/null || true
