# Motion-First Bidirectional Tracker (4K Source, Person Anywhere) — Spec & ToDo (single-storypoint granularity)

## Overview
- Goal: Track a single target across an entire 4K (3840×2160) pool surveillance video, seeded at an arbitrary frame `f0`, with forward and backward passes fused via RTS smoothing. The person may be anywhere: water, deck, bleachers, walkways.
- Method: Motion-centric (dense optical flow → Kalman filter), optional detector only for scale/anchor within motion gates. No face ID.
- Output: Annotated 4K video, trajectory CSV/JSONL, audit log, QC artifacts.
- New: **Zone Editor Utility** that loads a frame and lets you draw/edit **zone polygons** interactively, saved to JSON for the pipeline.

## Assumptions
- Fixed wall-mounted camera; static scene geometry.
- Constant FPS indexing; compute at “work resolution” (720p/1080p) but store/render at 4K.
- Zones are named polygons in **source pixel coordinates**; each zone can have **multiple polygons**.

## Architecture Summary
- Optical flow precompute/stream at work res.
- Forward/backward KF passes; RTS fusion; EMA post-smoothing; gap handling.
- **Zone Editor Utility** produces `zones.json` consumed at runtime.
- Renderer overlays boxes/labels at 4K; original audio/timebase preserved.
- Reproducibility logging and audit artifacts.

---

## Project Structure (single-file target initially + a small helper)
- `tracker.py` — main pipeline (can be split later).
- `zone_editor.py` — **helper utility** to draw zone polygons interactively.
- `config.yaml` — runtime configuration (includes path to zones file).
- Outputs per run under `runs/<timestamp>/`.

---

## Acceptance Criteria
- Deterministic outputs for fixed inputs/config.
- End-to-end run produces: annotated 4K video, `track.csv`, `track.jsonl`, `audit.json`, optional QC clips.
- Zones respected: per-zone motion bounds and occlusion tolerances applied; stationary spans preserved in non-water zones.
- Work↔source mapping accurate; overlays align at native 4K.
- **Zone Editor Utility**:
  - Opens a specific frame (by index or timestamp) or external still image.
  - Supports create/edit/delete points and polygons, multiple polygons per zone, undo/redo, and save/load of `zones.json`.
  - Saves schema with `source_width/height`, `frame_index_used`, and per-zone polygons in **source pixel coordinates**.
  - Visual cues (active zone name/color), snap-to-vertex within a tolerance, and help overlay.
  - Round-trip: load → edit → save without losing vertex order or precision.

---

## ToDo (each item is 1 story point)

### Repo, Config, Run Management
- [ ] [1pt] Create `config.yaml` with defaults including **zones file path** and per-zone motion bounds.
- [ ] [1pt] Implement config loader with multi-file override merge via CLI `--config`.
- [ ] [1pt] Add `RUN_DIR` bootstrap (timestamped) and path expansion.

### Probe and I/O
- [ ] [1pt] Probe FPS, frame count, width/height; write `ffprobe.json`.
- [ ] [1pt] Implement frame reader with constant-FPS guard and work-res downscale; record exact scale/pad transform.
- [ ] [1pt] Detect GPU decode (NVDEC) vs CPU; log choice and codec info.

### Zones and Masks (Pipeline Consumption)
- [ ] [1pt] Load `zones.json` (if provided) and build binary masks per zone in source and work spaces.
- [ ] [1pt] Implement helper to compute **zone occupancy** for a box (fraction overlap) and choose the **active zone** per frame with hysteresis.
- [ ] [1pt] Implement **zone-aware motion bounds** (speed, accel, scale change) and missing-measurement tolerance.

### **Zone Editor Utility** (Interactive Polygon Tool)
- [ ] [1pt] CLI `zone-edit` to open a frame by `--frame-index` or `--timestamp 00:MM:SS.ff`; fall back to first frame if unspecified.
- [ ] [1pt] Support alternative input `--image path.png` to draw on a still (bypasses video probe).
- [ ] [1pt] Render base frame scaled to fit window; maintain **exact source→screen** mapping to write back **source pixel** coordinates.
- [ ] [1pt] Keyboard/UI controls:
  - `1..9` select zone by index/name cycle; `Tab` next zone; on first use, prompt for zone name.
  - `Left-click` add vertex; `Enter`/`c` close polygon; `n` start new polygon in active zone.
  - `Drag` on vertex to move (snap if within `snap_px`); `Backspace/z` undo last vertex; `Shift+z` redo.
  - `d` delete vertex under cursor; `r` remove last polygon in active zone; `x` delete polygon under cursor (confirm).
  - `s` save `zones.json`; `l` load existing; `h` toggle help; `=`/`-` change overlay alpha; `Esc/q` quit (confirm on unsaved changes).
- [ ] [1pt] Visual cues: distinct color per zone, active zone highlighted, semi-transparent polygon fills (`overlay_alpha`), vertex markers, index labels.
- [ ] [1pt] Precision: store float vertex coords in source pixels; no rounding; preserve order and multiple polygons per zone.
- [ ] [1pt] Schema save:
  - Header: `{ "source_width": W, "source_height": H, "frame_index_used": f, "timestamp_used": "HH:MM:SS.ff" | null, "version": "1.0" }`
  - Zones: `[ { "name": "water", "polygons": [ [ [x,y], ... ], ... ] }, ... ]`
- [ ] [1pt] Validation: on save, ensure all polygons are ≥3 vertices and within `[0..W-1, 0..H-1]`.
- [ ] [1pt] Round-trip tests: load → edit → save preserves geometry; unit tests for JSON schema.
- [ ] [1pt] Export a quick PNG overlay preview `zones_preview.png` (optional) with alpha composited on the frame for visual confirmation.

### Optical Flow
- [ ] [1pt] Implement Farnebäck flow at work res (CPU baseline) with tunables.
- [ ] [1pt] Add RAFT stub (GPU optional) feature-flagged via config.
- [ ] [1pt] Implement forward flow iterator with chunking and optional disk caching.
- [ ] [1pt] Implement reverse-time access (recompute or inverse-sample) for backward pass.
- [ ] [1pt] Implement robust flow sampling inside a **dilated** box using Tukey biweight; compute center shift and rim divergence.
- [ ] [1pt] Implement **flow coherence** metrics (agreement ratio, max angle).

### Seed Handling
- [ ] [1pt] Parse seed from CLI (`--seed-frame`, `--seed-box`) or JSON; validate bounds; store canonical **source** coords.
- [ ] [1pt] Map seed to work space when needed; preserve reversible mapping records.

### Kalman Filter (KF) + Zone-Aware Constraints
- [ ] [1pt] Define state `[cx,cy,w,h,vx,vy,vw,vh]` and CV transition with `Δt=1/FPS`.
- [ ] [1pt] Adaptive `Q`: scale by recent flow variance and **zone multiplier**.
- [ ] [1pt] Measurement from flow (center shift, rim divergence), mapped back to **source** coordinates.
- [ ] [1pt] **Zone-aware gating**: Mahalanobis + IoU with per-zone thresholds.
- [ ] [1pt] **Stationary hypothesis**: when coherence < threshold and zone ∈ {deck, bleachers}, bias to zero-velocity; prevent drift.
- [ ] [1pt] Missing-measurement mode (prediction-only) with per-zone max durations and covariance inflation.

### Optional Detector (Scale/Anchor Only, Zone-Gated)
- [ ] [1pt] Integrate optional person detector at downscaled res; map detections to source.
- [ ] [1pt] Accept at most one detection inside the KF gate and above IoU τ; use to re-anchor `w,h` (small `cx,cy` correction) respecting zone bounds.

### Forward Pass
- [ ] [1pt] Implement forward loop `f0..F-1`: predict; derive motion measurement; **zone-aware** gating; update or skip; emit per-frame record with zone label.
- [ ] [1pt] Diagnostics: `maha_sq`, `flow_agree`, `iou_meas_pred`, zone, flags (`MEAS|PRED|GAP_*|LOW_CONF|STATIONARY`).

### Backward Pass
- [ ] [1pt] Implement backward loop `f0..0` mirroring forward logic with reverse-time flow.
- [ ] [1pt] Convert backward outputs to forward temporal indexing for fusion.

### RTS Smoothing, EMA, and Gaps
- [ ] [1pt] Implement RTS smoother to fuse forward/backward estimates.
- [ ] [1pt] Apply EMA to `[cx,cy]` and `[w,h]`.
- [ ] [1pt] Implement gap interpolation (linear or monotone cubic) with flags and **confidence decay** during prediction-only spans.

### Coordinate Mapping
- [ ] [1pt] Implement precise work↔source mapping (with/without letterbox); unit-test round-trip error.
- [ ] [1pt] Ensure CSV/JSONL store **source-space floats**; round only when drawing.

### Renderer and Mux (4K)
- [ ] [1pt] Draw labeled rectangle at 4K with zone and confidence; configurable thickness/font.
- [ ] [1pt] Write annotated video preserving original FPS/timebase; mux original audio; NVENC when available.
- [ ] [1pt] Export low-bitrate review MP4 (CRF 20–23).

### Outputs and Schemas
- [ ] [1pt] Emit `track.csv` columns: `frame,time_s,cx,cy,w,h,x1,y1,x2,y2,confidence,flags,zone,maha_sq,flow_agree,iou_meas_pred`.
- [ ] [1pt] Emit `track.jsonl` mirroring CSV.
- [ ] [1pt] Emit `audit.json` with hashes, config snapshot, software versions, model hashes, GPU/driver info, seed.

### QC Artifacts
- [ ] [1pt] Confidence timeline; mark low-confidence spans with zone context.
- [ ] [1pt] Export short QC clips around bottom-k confidence windows (3 s each).
- [ ] [1pt] CSV-driven plots: confidence vs time, speed vs time, zone occupancy vs time.

### Performance and Chunking
- [ ] [1pt] Chunked processing (e.g., 1500 frames, 60-frame overlap) with persisted partial KF states.
- [ ] [1pt] CLI knobs for chunk size/overlap; log throughput and VRAM/CPU usage.

### Validation Tests
- [ ] [1pt] Unit: KF predict/update with zone-bound clamping.
- [ ] [1pt] Unit: work↔source mapping error < 0.5 px at 4K.
- [ ] [1pt] Synthetic integration: moving rectangle across **multiple zones** with occlusions; IoU ≥ 0.9 after RTS; no zone-bound violations.
- [ ] [1pt] Realistic clip: seated (bleachers) remains stationary ≥30 s; deck walking respects speed bounds; swim respects water bounds.

### Reproducibility and Forensics
- [ ] [1pt] Store SHA-256 of input MKV container and raw stream.
- [ ] [1pt] Persist merged config and environment (Python/OpenCV/torch; CUDA; GPU/driver).
- [ ] [1pt] Log all thresholds (including per-zone), rationale notes, and seed details.

---

## Zone Editor JSON Schema (v1.0)
```json
{
  "version": "1.0",
  "source_width": 3840,
  "source_height": 2160,
  "frame_index_used": 43200,
  "timestamp_used": "00:24:00.00",
  "zones": [
    { "name": "water", "polygons": [ [ [x,y], [x,y], [x,y] ], ... ] },
    { "name": "deck", "polygons": [ [ [x,y], ... ] ] },
    { "name": "bleachers", "polygons": [] },
    { "name": "walkway", "polygons": [] }
  ]
}
```

---

## CLI Summary
- `python tracker.py full-run --video pool.mkv --config config.yaml --seed-frame 43200 --seed-box 512,360,680,540 --run-dir runs/2025-08-20_caseA`
- `python zone_editor.py zone-edit --video pool.mkv --frame-index 43200 --zones-out zones.json`
- `python zone_editor.py zone-edit --image frame.png --zones-in zones.json --zones-out zones.json`
