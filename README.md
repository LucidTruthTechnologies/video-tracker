# Motion-First Bidirectional Tracker (4K Source, Person Anywhere)

A high-performance video tracking system designed for surveillance applications, featuring zone-aware motion tracking with bidirectional Kalman filtering and RTS smoothing.

## Features

- **Zone-Aware Tracking**: Define custom zones (water, deck, bleachers, walkways) with different motion parameters
- **Interactive Zone Editor**: Visual polygon drawing tool for defining tracking zones
- **Bidirectional Tracking**: Forward and backward passes with RTS smoothing for robust trajectory estimation
- **Motion-First Approach**: Dense optical flow analysis for reliable tracking without requiring person detection
- **4K Support**: Native 4K video processing with work-resolution computation for performance
- **Reproducible Results**: Deterministic outputs with comprehensive audit logging

## Installation

### System Dependencies

**FFmpeg is required** for video processing and metadata extraction:

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg -y

# macOS (using Homebrew)
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
# Or use: winget install ffmpeg
```

**Verify FFmpeg installation:**
```bash
ffprobe -version
```

### Python Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd video-tracker
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify complete installation:**
   ```bash
   ffprobe -version  # Should show FFmpeg version
   python -c "import cv2, numpy, yaml; print('All dependencies installed!')"
   ```

## Quick Start

### 1. Define Tracking Zones

Use the interactive zone editor to define tracking zones:

```bash
# Edit zones on a specific video frame
python zone_editor.py zone-edit --video pool.mkv --frame-index 43200 --zones-out zones.json

# Or edit zones on a still image
python zone_editor.py zone-edit --image frame.png --zones-out zones.json
```

**Zone Editor Controls:**
- `1-9`: Select zone by index
- `Tab`: Next zone
- `Left-click`: Add vertex / Select vertex
- `Enter/c`: Close polygon
- `n`: New polygon in active zone
- `Backspace/z`: Undo
- `Shift+z`: Redo
- `h`: Toggle help overlay
- `Esc/q`: Quit

### 2. Run Tracking

Execute the full tracking pipeline:

```bash
python tracker.py full-run \
  --video pool.mkv \
  --config config.yaml \
  --seed-frame 43200 \
  --seed-box 512,360,680,540 \
  --run-dir runs/2025-08-20_caseA
```

### 3. Review Results

Results are saved in the specified run directory:
- `track.csv`: Trajectory data in CSV format
- `track.jsonl`: Trajectory data in JSONL format
- `annotated_4k.mp4`: Annotated video with tracking overlays
- `audit.json`: Comprehensive audit log
- `ffprobe.json`: Video metadata

## Configuration

The system uses `config.yaml` for configuration. Key settings include:

- **Zone Parameters**: Motion bounds, occlusion tolerances, and flow parameters per zone
- **Tracking Parameters**: Kalman filter settings, gating thresholds, and smoothing options
- **Flow Engine**: Farneback (CPU) or RAFT (GPU) optical flow
- **Output Settings**: Video quality, annotation style, and QC options

### Zone Configuration Example

```yaml
zone_params:
  water:
    speed_px_s_max: 2400
    accel_px_s2_max: 10000
    scale_change_per_frame_max: 0.10
    missing_frames_max: 120
    process_noise_scale: 1.5
  deck:
    speed_px_s_max: 1200
    accel_px_s2_max: 5000
    scale_change_per_frame_max: 0.06
    missing_frames_max: 180
    process_noise_scale: 1.0
```

## Architecture

### Core Components

1. **VideoProbe**: Video metadata extraction and GPU decode detection
2. **ZoneManager**: Zone loading, validation, and mask generation
3. **OpticalFlow**: Dense optical flow computation with zone-aware sampling
4. **KalmanTracker**: Bidirectional tracking with zone constraints
5. **RTSSmoother**: Forward/backward pass fusion and smoothing
6. **Renderer**: 4K video annotation and output generation

### Processing Pipeline

1. **Initialization**: Load config, probe video, initialize zones
2. **Forward Pass**: Track forward from seed frame with zone-aware constraints
3. **Backward Pass**: Track backward from seed frame
4. **Fusion**: RTS smoothing to combine forward/backward estimates
5. **Post-processing**: EMA smoothing, gap interpolation, confidence scoring
6. **Output**: Generate annotated video and trajectory data

## Performance Considerations

- **Work Resolution**: Processing at 720p/1080p for performance, rendering at 4K
- **Chunked Processing**: Configurable frame chunks with overlap for memory efficiency
- **GPU Acceleration**: Optional GPU decode (NVDEC) and optical flow (RAFT)
- **Caching**: Disk caching for optical flow results

## Output Formats

### Trajectory CSV

```csv
frame,time_s,cx,cy,w,h,x1,y1,x2,y2,confidence,flags,zone,maha_sq,flow_agree,iou_meas_pred
43200,1440.0,596.5,450.2,168.0,180.0,512.5,360.1,680.5,540.1,0.95,MEAS,water,2.1,0.85,0.92
43201,1440.03,598.2,451.8,168.5,180.2,514.0,361.7,682.5,542.0,0.94,MEAS,water,2.3,0.83,0.91
```

### Audit Log

```json
{
  "input_hash": "sha256:abc123...",
  "config_snapshot": {...},
  "software_versions": {...},
  "gpu_info": {...},
  "seed_details": {...},
  "processing_metadata": {...}
}
```

## Troubleshooting

### Common Issues

**"No such file or directory: 'ffprobe'"**
- Install FFmpeg: `sudo apt install ffmpeg` (Ubuntu/Debian)
- Verify with: `ffprobe -version`

**"ModuleNotFoundError: No module named 'cv2'"**
- Activate virtual environment: `source venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`

**Video file not found or invalid**
- Check file path and permissions
- Ensure video format is supported (MP4, AVI, MKV, MOV)
- Verify video file integrity

**Zone file errors**
- Validate JSON syntax: `python -m json.tool test_zones.json`
- Check zone dimensions match video dimensions
- Ensure all polygons have ≥3 vertices

**Memory issues during processing**
- Reduce work resolution in config.yaml
- Process shorter video segments
- Check available system memory

### Debug Mode

Enable verbose logging:

```bash
export PYTHONPATH=.
python -u tracker.py full-run --video pool.mkv --config config.yaml 2>&1 | tee run.log
```

## Development

### Project Structure

```
video-tracker/
├── tracker.py          # Main tracking pipeline
├── zone_editor.py      # Interactive zone editor
├── config.yaml         # Configuration file
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── runs/              # Output directory
    └── <timestamp>/   # Per-run results
```

### Adding New Features

1. Follow the existing code structure and patterns
2. Add comprehensive error handling and validation
3. Include unit tests for new functionality
4. Update configuration schema as needed
5. Document new features in this README

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
