#!/usr/bin/env python3
"""
Motion-First Bidirectional Tracker (4K Source, Person Anywhere)
Main pipeline implementation with zone-aware tracking and RTS smoothing.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
from dataclasses import dataclass, field
import hashlib
import subprocess
import platform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration container with validation."""
    io: Dict[str, Any]
    zones_file: str
    zone_params: Dict[str, Dict[str, Any]]
    flow: Dict[str, Any]
    tracker: Dict[str, Any]
    detector: Dict[str, Any]
    render: Dict[str, Any]
    qc: Dict[str, Any]
    zone_editor: Dict[str, Any]
    outputs: Dict[str, str]
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.io.get('work_height') or self.io['work_height'] <= 0:
            raise ValueError("work_height must be positive")
        if self.tracker.get('dt') is not None and self.tracker['dt'] <= 0:
            raise ValueError("dt must be positive or null")


@dataclass
class Zone:
    """Zone representation with polygons and parameters."""
    name: str
    polygons: List[List[Tuple[float, float]]]
    params: Dict[str, Any]
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside any polygon of this zone."""
        for polygon in self.polygons:
            if self._point_in_polygon(x, y, polygon):
                return True
        return False
    
    def _point_in_polygon(self, x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        """Point-in-polygon test using ray casting."""
        if len(polygon) < 3:
            return False
        
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            if ((polygon[i][1] > y) != (polygon[j][1] > y) and
                x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / 
                    (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
                inside = not inside
            j = i
        return inside


@dataclass
class TrackingState:
    """Kalman filter state and metadata."""
    frame_idx: int
    time_s: float
    cx: float  # center x (source coordinates)
    cy: float  # center y (source coordinates)
    w: float   # width (source coordinates)
    h: float   # height (source coordinates)
    vx: float  # velocity x
    vy: float  # velocity y
    vw: float  # velocity width
    vh: float  # velocity height
    confidence: float
    zone: str
    flags: List[str] = field(default_factory=list)
    maha_sq: Optional[float] = None
    flow_agree: Optional[float] = None
    iou_meas_pred: Optional[float] = None
    x1: Optional[float] = None  # bounding box x1 (source coordinates)
    y1: Optional[float] = None  # bounding box y1 (source coordinates)
    x2: Optional[float] = None  # bounding box x2 (source coordinates)
    y2: Optional[float] = None  # bounding box y2 (source coordinates)


@dataclass
class Seed:
    """Seed information for tracking initialization."""
    frame_idx: int
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 in source coordinates
    time_s: float
    
    def validate(self, source_width: int, source_height: int) -> bool:
        """Validate seed bounds."""
        x1, y1, x2, y2 = self.box
        return (0 <= x1 < x2 < source_width and 
                0 <= y1 < y2 < source_height)


class ConfigLoader:
    """Configuration loader with multi-file override support."""
    
    @staticmethod
    def load_config(config_paths: List[str]) -> Config:
        """Load and merge multiple configuration files."""
        if not config_paths:
            raise ValueError("At least one config file must be provided")
        
        # Load base config
        base_config = ConfigLoader._load_yaml(config_paths[0])
        
        # Apply overrides
        for override_path in config_paths[1:]:
            override_config = ConfigLoader._load_yaml(override_path)
            ConfigLoader._merge_configs(base_config, override_config)
        
        # Expand environment variables in output paths
        ConfigLoader._expand_output_paths(base_config)
        
        return Config(**base_config)
    
    @staticmethod
    def _load_yaml(file_path: str) -> Dict[str, Any]:
        """Load YAML file."""
        try:
            import yaml
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.error("PyYAML not installed. Install with: pip install PyYAML")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load config file {file_path}: {e}")
            sys.exit(1)
    
    @staticmethod
    def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge override config into base config."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigLoader._merge_configs(base[key], value)
            else:
                base[key] = value
    
    @staticmethod
    def _expand_output_paths(config: Dict[str, Any]):
        """Expand environment variables in output paths."""
        for key, value in config.get('outputs', {}).items():
            if isinstance(value, str):
                config['outputs'][key] = os.path.expandvars(value)


class VideoProbe:
    """Video metadata probing and validation."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.metadata = {}
        
    def probe(self) -> Dict[str, Any]:
        """Probe video file and return metadata."""
        try:
            # Use ffprobe to get video information
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', self.video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = None
            for stream in data['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            # Extract metadata
            self.metadata = {
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': self._parse_fps(video_stream.get('r_frame_rate', '0/1')),
                'frame_count': int(video_stream.get('nb_frames', 0)),
                'duration': float(data['format'].get('duration', 0)),
                'codec': video_stream.get('codec_name', 'unknown'),
                'pixel_format': video_stream.get('pix_fmt', 'unknown'),
                'bitrate': int(data['format'].get('bit_rate', 0))
            }
            
            # Detect GPU decode capability
            self.metadata['gpu_decode'] = self._detect_gpu_decode()
            
            logger.info(f"Video probed: {self.metadata['width']}x{self.metadata['height']} "
                       f"@{self.metadata['fps']:.2f}fps, {self.metadata['frame_count']} frames")
            
            return self.metadata
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Video probing failed: {e}")
            raise
    
    def _parse_fps(self, fps_str: str) -> float:
        """Parse FPS string (e.g., '30/1' -> 30.0)."""
        try:
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                return num / den if den != 0 else 0
            else:
                return float(fps_str)
        except:
            return 0.0
    
    def _detect_gpu_decode(self) -> bool:
        """Detect if GPU decode (NVDEC) is available."""
        try:
            # Check for NVIDIA GPU and drivers
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("NVIDIA GPU detected - NVDEC available")
                return True
        except:
            pass
        
        logger.info("GPU decode not available - using CPU")
        return False


class FrameReader:
    """Frame reader with work resolution downscaling."""
    
    def __init__(self, video_path: str, work_height: int):
        self.video_path = video_path
        self.work_height = work_height
        self.cap = None
        self.source_width = 0
        self.source_height = 0
        self.work_width = 0
        self.scale_factor = 1.0
        
    def open(self) -> bool:
        """Open video capture."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            return False
        
        self.source_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.source_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate work resolution
        self.scale_factor = self.work_height / self.source_height
        self.work_width = int(self.source_width * self.scale_factor)
        
        logger.info(f"Frame reader: {self.source_width}x{self.source_height} -> {self.work_width}x{self.work_height}")
        return True
    
    def seek_frame(self, frame_idx: int) -> bool:
        """Seek to specific frame."""
        if self.cap is None:
            return False
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Read next frame, returns (success, source_frame, work_frame)."""
        if self.cap is None:
            return False, None, None
        
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None
        
        # Downscale to work resolution
        work_frame = cv2.resize(frame, (self.work_width, self.work_height))
        
        return True, frame, work_frame
    
    def close(self):
        """Close video capture."""
        if self.cap:
            self.cap.release()


class ZoneManager:
    """Zone management and mask generation."""
    
    def __init__(self, config: Config, source_width: int, source_height: int):
        self.config = config
        self.source_width = source_width
        self.source_height = source_height
        self.zones: Dict[str, Zone] = {}
        self.source_masks: Dict[str, np.ndarray] = {}
        self.work_masks: Dict[str, np.ndarray] = {}
        
    def load_zones(self, zones_file: str) -> bool:
        """Load zones from JSON file."""
        if not zones_file or not os.path.exists(zones_file):
            logger.warning(f"Zones file not found: {zones_file}")
            return False
        
        try:
            with open(zones_file, 'r') as f:
                zones_data = json.load(f)
            
            # Validate schema
            if not self._validate_zones_schema(zones_data):
                return False
            
            # Load zones
            self.zones = {}
            for zone_data in zones_data['zones']:
                zone_name = zone_data['name']
                polygons = zone_data['polygons']
                
                # Get zone parameters (with defaults)
                zone_params = self.config.zone_params.get(zone_name, {})
                
                zone = Zone(name=zone_name, polygons=polygons, params=zone_params)
                self.zones[zone_name] = zone
            
            logger.info(f"Loaded {len(self.zones)} zones from {zones_file}")
            
            # Generate masks
            self._generate_masks()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load zones: {e}")
            return False
    
    def _validate_zones_schema(self, zones_data: Dict[str, Any]) -> bool:
        """Validate zones JSON schema."""
        required_fields = ['version', 'source_width', 'source_height', 'zones']
        for field in required_fields:
            if field not in zones_data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Check dimensions match
        if (zones_data['source_width'] != self.source_width or 
            zones_data['source_height'] != self.source_height):
            logger.error("Zone dimensions don't match video dimensions")
            return False
        
        # Validate zone polygons
        for zone in zones_data['zones']:
            if 'name' not in zone or 'polygons' not in zone:
                logger.error("Invalid zone structure")
                return False
            
            for polygon in zone['polygons']:
                if len(polygon) < 3:
                    logger.error(f"Polygon in zone {zone['name']} has <3 vertices")
                    return False
                
                for point in polygon:
                    if len(point) != 2:
                        logger.error(f"Invalid point in zone {zone['name']}")
                        return False
                    
                    x, y = point
                    if not (0 <= x < self.source_width and 0 <= y < self.source_height):
                        logger.error(f"Point {point} outside bounds in zone {zone['name']}")
                        return False
        
        return True
    
    def _generate_masks(self):
        """Generate binary masks for each zone in source and work resolutions."""
        work_height = self.config.io['work_height']
        work_width = int(self.source_width * work_height / self.source_height)
        
        for zone_name, zone in self.zones.items():
            # Source resolution mask
            source_mask = np.zeros((self.source_height, self.source_width), dtype=np.uint8)
            for polygon in zone.polygons:
                polygon_array = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(source_mask, [polygon_array], 255)
            self.source_masks[zone_name] = source_mask
            
            # Work resolution mask
            work_mask = np.zeros((work_height, work_width), dtype=np.uint8)
            for polygon in zone.polygons:
                # Scale polygon to work resolution
                work_polygon = []
                for x, y in polygon:
                    work_x = int(x * work_width / self.source_width)
                    work_y = int(y * work_height / self.source_height)
                    work_polygon.append([work_x, work_y])
                
                polygon_array = np.array(work_polygon, dtype=np.int32)
                cv2.fillPoly(work_mask, [polygon_array], 255)
            
            self.work_masks[zone_name] = work_mask
    
    def get_active_zone(self, box: Tuple[float, float, float, float], 
                        work_res: bool = False) -> Tuple[str, float]:
        """Determine active zone for a bounding box with overlap fraction."""
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        best_zone = "unknown"
        best_overlap = 0.0
        
        masks = self.work_masks if work_res else self.source_masks
        
        for zone_name, mask in masks.items():
            # Calculate overlap fraction
            box_mask = np.zeros_like(mask)
            x1_int, y1_int = int(x1), int(y1)
            x2_int, y2_int = int(x2), int(y2)
            
            # Ensure bounds
            x1_int = max(0, min(x1_int, mask.shape[1] - 1))
            y1_int = max(0, min(y1_int, mask.shape[0] - 1))
            x2_int = max(0, min(x2_int, mask.shape[1] - 1))
            y2_int = max(0, min(y2_int, mask.shape[0] - 1))
            
            if x2_int > x1_int and y2_int > y1_int:
                box_mask[y1_int:y2_int, x1_int:x2_int] = 255
                overlap = np.sum((mask > 0) & (box_mask > 0)) / np.sum(box_mask > 0)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_zone = zone_name
        
        return best_zone, best_overlap


class OpticalFlow:
    """Optical flow computation with zone-aware sampling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = config.get('engine', 'farneback')
        self.dilate_box_px = config.get('dilate_box_px_work', 12)
        self.tukey_c = config.get('tukey_c', 4.5)
        
    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Compute optical flow between two frames."""
        if self.engine == 'farneback':
            return self._compute_farneback_flow(frame1, frame2)
        elif self.engine == 'raft':
            return self._compute_raft_flow(frame1, frame2)
        else:
            logger.warning(f"Unknown flow engine: {self.engine}, falling back to Farneback")
            return self._compute_farneback_flow(frame1, frame2)
    
    def _compute_farneback_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Compute Farneback optical flow."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Compute flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, 
            flags=0
        )
        
        return flow
    
    def _compute_raft_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Compute RAFT optical flow (GPU-accelerated)."""
        # TODO: Implement RAFT flow computation
        logger.warning("RAFT flow not yet implemented, falling back to Farneback")
        return self._compute_farneback_flow(frame1, frame2)
    
    def sample_flow_in_box(self, flow: np.ndarray, box: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """Sample flow within a dilated bounding box using Tukey biweight."""
        x1, y1, x2, y2 = map(int, box)
        
        # Dilate box
        x1_d = max(0, x1 - self.dilate_box_px)
        y1_d = max(0, y1 - self.dilate_box_px)
        x2_d = min(flow.shape[1], x2 + self.dilate_box_px)
        y2_d = min(flow.shape[0], y2 + self.dilate_box_px)
        
        # Extract flow in dilated region
        flow_region = flow[y1_d:y2_d, x1_d:x2_d]
        
        if flow_region.size == 0:
            return {'center_shift': (0, 0), 'rim_divergence': 0, 'flow_agree': 0}
        
        # Compute center shift (mean flow)
        center_shift = np.mean(flow_region, axis=(0, 1))
        
        # Compute rim divergence (flow variance)
        rim_divergence = np.std(flow_region, axis=(0, 1))
        
        # Compute flow agreement (coherence)
        flow_magnitudes = np.linalg.norm(flow_region, axis=2)
        flow_angles = np.arctan2(flow_region[:, :, 1], flow_region[:, :, 0])
        
        # Calculate agreement ratio
        mean_angle = np.mean(flow_angles)
        angle_diff = np.abs(flow_angles - mean_angle)
        agreement_ratio = np.sum(angle_diff < np.pi/4) / angle_diff.size
        
        return {
            'center_shift': center_shift,
            'rim_divergence': rim_divergence,
            'flow_agree': agreement_ratio
        }


class KalmanTracker:
    """Kalman filter tracker with zone-aware constraints."""
    
    def __init__(self, config: Dict[str, Any], fps: float):
        self.config = config
        self.dt = 1.0 / fps if fps > 0 else 0.033  # Default to 30fps
        
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.state_dim = 8
        self.measurement_dim = 4  # [cx, cy, w, h]
        
        # Initialize Kalman filter
        self.kf = cv2.KalmanFilter(self.state_dim, self.measurement_dim)
        self._setup_kalman_filter()
        
        # Zone-aware parameters
        self.zone_params = config.get('zone_params', {})
        
    def _setup_kalman_filter(self):
        """Setup Kalman filter matrices."""
        # State transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.eye(self.state_dim, dtype=np.float32)
        self.kf.transitionMatrix[:4, 4:] = np.eye(4, dtype=np.float32) * self.dt
        
        # Measurement matrix
        self.kf.measurementMatrix = np.zeros((self.measurement_dim, self.state_dim), dtype=np.float32)
        self.kf.measurementMatrix[:4, :4] = np.eye(4, dtype=np.float32)
        
        # Process noise covariance
        process_noise = np.eye(self.state_dim, dtype=np.float32)
        process_noise[4:, 4:] *= 100  # Higher noise for velocities
        self.kf.processNoiseCov = process_noise * 0.1
        
        # Measurement noise covariance
        measurement_noise = np.eye(self.measurement_dim, dtype=np.float32)
        self.kf.measurementNoiseCov = measurement_noise * 10
        
        # Initial state covariance
        self.kf.errorCovPost = np.eye(self.state_dim, dtype=np.float32) * 100
        
    def predict(self) -> np.ndarray:
        """Predict next state."""
        prediction = self.kf.predict()
        return prediction.flatten()
    
    def update(self, measurement: np.ndarray, zone_name: str = "unknown") -> np.ndarray:
        """Update state with measurement."""
        # Adjust process noise based on zone
        if zone_name in self.zone_params:
            zone_scale = self.zone_params[zone_name].get('process_noise_scale', 1.0)
            self.kf.processNoiseCov *= zone_scale
        
        # Update with measurement
        self.kf.correct(measurement.astype(np.float32))
        return self.kf.statePost.flatten()
    
    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.kf.statePost.flatten()
    
    def set_state(self, state: np.ndarray):
        """Set current state."""
        self.kf.statePost = state.reshape(-1, 1).astype(np.float32)


class Tracker:
    """Main tracking pipeline."""
    
    def __init__(self, config: Config, video_path: str):
        self.config = config
        self.video_path = video_path
        self.video_probe = VideoProbe(video_path)
        self.frame_reader = None
        self.zone_manager = None
        self.optical_flow = None
        self.kalman_tracker = None
        self.seed = None
        
    def initialize(self) -> bool:
        """Initialize tracking components."""
        try:
            # Probe video
            metadata = self.video_probe.probe()
            
            # Initialize frame reader
            self.frame_reader = FrameReader(self.video_path, self.config.io['work_height'])
            if not self.frame_reader.open():
                raise RuntimeError("Failed to open video file")
            
            # Initialize zone manager
            self.zone_manager = ZoneManager(self.config, metadata['width'], metadata['height'])
            if self.config.zones_file:
                self.zone_manager.load_zones(self.config.zones_file)
            
            # Initialize optical flow
            self.optical_flow = OpticalFlow(self.config.flow)
            
            # Initialize Kalman tracker
            self.kalman_tracker = KalmanTracker(self.config.tracker, metadata['fps'])
            
            logger.info("Tracker initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize tracker: {e}")
            return False
    
    def set_seed(self, frame_idx: int, box: Tuple[float, float, float, float]) -> bool:
        """Set tracking seed."""
        if self.video_probe.metadata:
            time_s = frame_idx / self.video_probe.metadata['fps']
            self.seed = Seed(frame_idx=frame_idx, box=box, time_s=time_s)
            
            # Validate seed
            if not self.seed.validate(self.video_probe.metadata['width'], 
                                    self.video_probe.metadata['height']):
                logger.error("Invalid seed bounds")
                return False
            
            logger.info(f"Seed set: frame {frame_idx}, box {box}")
            return True
        return False
    
    def run_tracking(self, run_dir: Path) -> bool:
        """Run the complete tracking pipeline."""
        if not self.seed:
            logger.error("No seed set")
            return False
        
        try:
            # Create output files
            track_csv = run_dir / "track.csv"
            track_jsonl = run_dir / "track.jsonl"
            audit_json = run_dir / "audit.json"
            
            # Initialize tracking
            self.frame_reader.seek_frame(self.seed.frame_idx)
            
            # Initialize Kalman filter with seed
            self._initialize_tracker_with_seed()
            
            # Run forward tracking
            forward_results = self._run_forward_tracking()
            
            # TODO: Implement backward tracking and RTS smoothing
            logger.info("Forward tracking completed, backward tracking not yet implemented")
            
            # Save results
            self._save_tracking_results(forward_results, track_csv, track_jsonl)
            
            # Render annotated video
            annotated_video_path = run_dir / "annotated_video.mp4"
            if self._render_annotated_video(forward_results, annotated_video_path):
                logger.info(f"Annotated video saved: {annotated_video_path}")
            else:
                logger.warning("Failed to render annotated video")
            
            # Save audit log
            self._save_audit_log(audit_json)
            
            return True
            
        except Exception as e:
            logger.error(f"Tracking failed: {e}")
            return False
    
    def _initialize_tracker_with_seed(self):
        """Initialize Kalman tracker with seed information in work coordinates."""
        x1, y1, x2, y2 = self.seed.box
        
        # Convert seed from source coordinates to work coordinates
        scale_factor = getattr(self.frame_reader, 'scale_factor', 1.0)
        work_x1 = x1 * scale_factor
        work_y1 = y1 * scale_factor
        work_x2 = x2 * scale_factor
        work_y2 = y2 * scale_factor
        
        # Calculate center and size in work coordinates
        work_cx = (work_x1 + work_x2) / 2
        work_cy = (work_y1 + work_y2) / 2
        work_w = work_x2 - work_x1
        work_h = work_y2 - work_y1
        
        # Set initial state: [cx, cy, w, h, vx, vy, vw, vh] in work coordinates
        initial_state = np.array([work_cx, work_cy, work_w, work_h, 0.0, 0.0, 0.0, 0.0])
        self.kalman_tracker.set_state(initial_state)
        
        logger.info(f"Initialized tracker with seed: source center=({x1:.1f}, {y1:.1f}), work center=({work_cx:.1f}, {work_cy:.1f})")
    
    def _run_forward_tracking(self) -> List[TrackingState]:
        """Run forward tracking from seed frame to end of video."""
        results = []
        current_frame_idx = self.seed.frame_idx
        fps = self.video_probe.metadata['fps']
        
        # Read first frame
        ret, source_frame, work_frame = self.frame_reader.read_frame()
        if not ret:
            logger.error("Failed to read seed frame")
            return results
        
        prev_work_frame = work_frame
        prev_source_frame = source_frame
        
        logger.info(f"Starting forward tracking from frame {current_frame_idx}")
        
        while True:
            # Read next frame
            ret, source_frame, work_frame = self.frame_reader.read_frame()
            if not ret:
                logger.info(f"Reached end of video at frame {current_frame_idx}")
                break
            
            current_frame_idx += 1
            time_s = current_frame_idx / fps
            
            # Compute optical flow
            flow = self.optical_flow.compute_flow(prev_work_frame, work_frame)
            
            # Get current Kalman prediction
            predicted_state = self.kalman_tracker.predict()
            
            # Sample flow in predicted region
            pred_box = self._state_to_box(predicted_state)
            flow_info = self.optical_flow.sample_flow_in_box(flow, pred_box)
            
            # Create measurement from flow
            measurement = self._create_measurement_from_flow(predicted_state, flow_info)
            
            # Determine active zone
            zone_name, zone_overlap = self.zone_manager.get_active_zone(pred_box, work_res=True)
            
            # Apply zone-aware constraints
            if self._validate_measurement(measurement, zone_name, predicted_state):
                # Update Kalman filter
                updated_state = self.kalman_tracker.update(measurement, zone_name)
                confidence = self._calculate_confidence(flow_info, zone_name)
                flags = ["MEAS"]
            else:
                # Use prediction only
                updated_state = predicted_state
                confidence = self._calculate_confidence(flow_info, zone_name) * 0.8  # Reduce confidence
                flags = ["PRED"]
            
            # Create tracking state record
            tracking_state = self._create_tracking_state(
                current_frame_idx, time_s, updated_state, zone_name, 
                confidence, flags, flow_info
            )
            
            results.append(tracking_state)
            
            # Update previous frames
            prev_work_frame = work_frame
            prev_source_frame = source_frame
            
            # Log progress every 100 frames
            if current_frame_idx % 100 == 0:
                logger.info(f"Forward tracking: frame {current_frame_idx}, zone: {zone_name}, confidence: {confidence:.3f}")
        
        logger.info(f"Forward tracking completed: {len(results)} frames processed")
        return results
    
    def _state_to_box(self, state: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert Kalman state to bounding box (x1, y1, x2, y2) in work coordinates."""
        cx, cy, w, h = state[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return (x1, y1, x2, y2)
    
    def _work_to_source_coords(self, work_coords: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Convert work coordinates to source coordinates."""
        x1, y1, x2, y2 = work_coords
        scale_factor = getattr(self.frame_reader, 'scale_factor', 1.0)
        
        source_x1 = x1 / scale_factor
        source_y1 = y1 / scale_factor
        source_x2 = x2 / scale_factor
        source_y2 = y2 / scale_factor
        
        return (source_x1, source_y1, source_x2, source_y2)
    
    def _create_measurement_from_flow(self, predicted_state: np.ndarray, 
                                    flow_info: Dict[str, Any]) -> np.ndarray:
        """Create measurement from optical flow information."""
        cx, cy, w, h = predicted_state[:4]
        center_shift = flow_info['center_shift']
        
        # Apply flow shift to center
        new_cx = cx + center_shift[0]
        new_cy = cy + center_shift[1]
        
        # Keep size unchanged for now (could be refined with rim divergence)
        new_w = w
        new_h = h
        
        return np.array([new_cx, new_cy, new_w, new_h])
    
    def _validate_measurement(self, measurement: np.ndarray, zone_name: str, 
                            predicted_state: np.ndarray) -> bool:
        """Validate measurement against zone constraints."""
        if zone_name == "unknown":
            return True  # Accept measurement if no zone constraints
        
        # Get zone parameters
        zone_params = self.config.zone_params.get(zone_name, {})
        
        # Check speed constraints
        if 'speed_px_s_max' in zone_params:
            max_speed = zone_params['speed_px_s_max']
            current_speed = np.linalg.norm(measurement[:2] - predicted_state[:2])
            if current_speed > max_speed:
                logger.debug(f"Measurement rejected: speed {current_speed:.1f} exceeds zone limit {max_speed}")
                return False
        
        # Check size constraints
        if 'scale_change_per_frame_max' in zone_params:
            max_scale_change = zone_params['scale_change_per_frame_max']
            scale_change_w = abs(measurement[2] - predicted_state[2]) / predicted_state[2]
            scale_change_h = abs(measurement[3] - predicted_state[3]) / predicted_state[3]
            if scale_change_w > max_scale_change or scale_change_h > max_scale_change:
                logger.debug(f"Measurement rejected: scale change exceeds zone limit {max_scale_change}")
                return False
        
        return True
    
    def _calculate_confidence(self, flow_info: Dict[str, Any], zone_name: str) -> float:
        """Calculate confidence score based on flow coherence and zone."""
        # Base confidence from flow agreement
        flow_agree = flow_info.get('flow_agree', 0.0)
        
        # Zone-specific confidence adjustments
        zone_params = self.config.zone_params.get(zone_name, {})
        zone_confidence_boost = zone_params.get('confidence_boost', 1.0)
        
        # Calculate confidence (0.0 to 1.0)
        confidence = flow_agree * zone_confidence_boost
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def _create_tracking_state(self, frame_idx: int, time_s: float, state: np.ndarray,
                              zone_name: str, confidence: float, flags: List[str],
                              flow_info: Dict[str, Any]) -> TrackingState:
        """Create TrackingState record from current tracking information."""
        cx, cy, w, h, vx, vy, vw, vh = state
        
        # Convert work coordinates to source coordinates
        scale_factor = getattr(self.frame_reader, 'scale_factor', 1.0)
        source_cx = cx / scale_factor
        source_cy = cy / scale_factor
        source_w = w / scale_factor
        source_h = h / scale_factor
        
        # Make bounding box larger by percentage (configurable)
        box_expansion = self.config.render.get('box_expansion_percent', 20)  # 20% larger
        expansion_x = source_w * (box_expansion / 100.0)
        expansion_y = source_h * (box_expansion / 100.0)
        
        # Calculate expanded bounding box in source coordinates
        x1 = source_cx - (source_w / 2) - expansion_x
        y1 = source_cy - (source_h / 2) - expansion_y
        x2 = source_cx + (source_w / 2) + expansion_x
        y2 = source_cy + (source_h / 2) + expansion_y
        
        # Ensure bounding box is within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.video_probe.metadata['width'], x2)
        y2 = min(self.video_probe.metadata['height'], y2)
        
        # Convert velocities to source coordinates as well
        source_vx = vx / scale_factor
        source_vy = vy / scale_factor
        source_vw = vw / scale_factor
        source_vh = vh / scale_factor
        
        return TrackingState(
            frame_idx=frame_idx,
            time_s=time_s,
            cx=source_cx,
            cy=source_cy,
            w=source_w,
            h=source_h,
            vx=source_vx,
            vy=source_vy,
            vw=source_vw,
            vh=source_vh,
            confidence=confidence,
            zone=zone_name,
            flags=flags,
            flow_agree=flow_info.get('flow_agree', 0.0),
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2
        )
    
    def _save_tracking_results(self, results: List[TrackingState], 
                             csv_file: Path, jsonl_file: Path):
        """Save tracking results to CSV and JSONL files."""
        # Save CSV
        with open(csv_file, 'w', newline='') as f:
            import csv
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'frame', 'time_s', 'cx', 'cy', 'w', 'h', 'x1', 'y1', 'x2', 'y2',
                'confidence', 'flags', 'zone', 'maha_sq', 'flow_agree', 'iou_meas_pred'
            ])
            
            # Write data
            for result in results:
                writer.writerow([
                    result.frame_idx, result.time_s, result.cx, result.cy, 
                    result.w, result.h, result.x1, result.y1, result.x2, result.y2,
                    result.confidence, ','.join(result.flags), result.zone,
                    result.maha_sq, result.flow_agree, result.iou_meas_pred
                ])
        
        # Save JSONL
        with open(jsonl_file, 'w') as f:
            for result in results:
                # Convert to serializable format
                result_dict = self._convert_config_to_serializable(result.__dict__)
                json.dump(result_dict, f)
                f.write('\n')
        
        logger.info(f"Saved tracking results: {len(results)} frames to {csv_file} and {jsonl_file}")
    
    def _render_annotated_video(self, results: List[TrackingState], 
                               output_video_path: Path) -> bool:
        """Render annotated video with tracking boxes and metadata."""
        try:
            logger.info(f"Starting video rendering to: {output_video_path}")
            
            # Get video properties
            fps = self.video_probe.metadata['fps']
            width = self.video_probe.metadata['width']
            height = self.video_probe.metadata['height']
            
            logger.info(f"Video properties: {width}x{height} @ {fps}fps")
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error("Failed to create video writer")
                return False
            
            logger.info("Video writer created successfully")
            
            # Create results lookup by frame
            results_by_frame = {r.frame_idx: r for r in results}
            logger.info(f"Processing {len(results)} tracking results for {len(results_by_frame)} frames")
            
            # Seek to start
            self.frame_reader.seek_frame(0)
            logger.info("Seeked to frame 0")
            
            frame_idx = 0
            while True:
                # Read frame
                ret, source_frame, _ = self.frame_reader.read_frame()
                if not ret:
                    logger.info(f"End of video reached at frame {frame_idx}")
                    break
                
                # Get tracking result for this frame
                result = results_by_frame.get(frame_idx)
                
                if result:
                    # Draw bounding box with bounds checking
                    x1, y1, x2, y2 = map(int, [result.x1, result.y1, result.x2, result.y2])
                    
                    # Ensure bounds are within frame
                    x1 = max(0, min(x1, source_frame.shape[1] - 1))
                    y1 = max(0, min(y1, source_frame.shape[0] - 1))
                    x2 = max(0, min(x2, source_frame.shape[1] - 1))
                    y2 = max(0, min(y2, source_frame.shape[0] - 1))
                    
                    # Box color based on confidence
                    if result.confidence > 0.7:
                        color = (0, 255, 0)  # Green for high confidence
                    elif result.confidence > 0.4:
                        color = (0, 255, 255)  # Yellow for medium confidence
                    else:
                        color = (0, 0, 255)  # Red for low confidence
                    
                    # Draw rectangle
                    thickness = self.config.render.get('draw_thickness', 4)
                    cv2.rectangle(source_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw center point
                    center_x, center_y = int(result.cx), int(result.cy)
                    center_x = max(0, min(center_x, source_frame.shape[1] - 1))
                    center_y = max(0, min(center_y, source_frame.shape[0] - 1))
                    cv2.circle(source_frame, (center_x, center_y), 5, color, -1)
                    
                    # Draw velocity vector
                    if abs(result.vx) > 0.1 or abs(result.vy) > 0.1:
                        # Velocity is already in source coordinates, just scale for visibility
                        vel_scale = 10.0  # Make velocity arrows more visible
                        end_x = center_x + int(result.vx * vel_scale)
                        end_y = center_y + int(result.vy * vel_scale)
                        cv2.arrowedLine(source_frame, (center_x, center_y), (end_x, end_y), color, 2)
                    
                    # Draw text overlay with configurable positioning and background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = self.config.render.get('font_scale', 0.6)
                    thickness = 2
                    line_height = 25
                    padding = 10
                    
                    # Get annotation corner configuration
                    corner = self.config.render.get('annotation_corner', 'top-left')
                    show_background = self.config.render.get('annotation_background', True)
                    
                    # Calculate text position based on corner
                    if corner == 'top-left':
                        start_x = padding
                        start_y = padding + line_height
                    elif corner == 'top-right':
                        start_x = source_frame.shape[1] - 200  # Approximate text width
                        start_y = padding + line_height
                    elif corner == 'bottom-left':
                        start_x = padding
                        start_y = source_frame.shape[0] - padding - 6*line_height
                    elif corner == 'bottom-right':
                        start_x = source_frame.shape[1] - 200
                        start_y = source_frame.shape[0] - padding - 6*line_height
                    else:
                        # Default to top-left
                        start_x = padding
                        start_y = padding + line_height
                    
                    # Prepare text lines
                    text_lines = [
                        f"Frame: {frame_idx}",
                        f"Time: {result.time_s:.2f}s",
                        f"Conf: {result.confidence:.3f}",
                        f"Zone: {result.zone}",
                        f"Flags: {','.join(result.flags) if result.flags else 'None'}",
                        f"Pos: ({result.cx:.1f}, {result.cy:.1f})",
                        f"Vel: ({result.vx:.1f}, {result.vy:.1f})"
                    ]
                    
                    # Draw background rectangle if enabled
                    if show_background:
                        bg_width = 220  # Approximate width for all text
                        bg_height = len(text_lines) * line_height + padding
                        bg_x1 = start_x - padding
                        bg_y1 = start_y - line_height
                        bg_x2 = bg_x1 + bg_width
                        bg_y2 = bg_y1 + bg_height
                        
                        # Ensure background is within frame bounds
                        bg_x1 = max(0, bg_x1)
                        bg_y1 = max(0, bg_y1)
                        bg_x2 = min(source_frame.shape[1], bg_x2)
                        bg_y2 = min(source_frame.shape[0], bg_y2)
                        
                        # Draw semi-transparent dark background
                        overlay = source_frame.copy()
                        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.7, source_frame, 0.3, 0, source_frame)
                    
                    # Draw text lines
                    for i, text in enumerate(text_lines):
                        y_pos = start_y + i * line_height
                        cv2.putText(source_frame, text, (start_x, y_pos), 
                                   font, font_scale, (255, 255, 255), thickness)
                
                # Write frame
                out.write(source_frame)
                frame_idx += 1
                
                # Log progress
                if frame_idx % 30 == 0:
                    logger.info(f"Rendering frame {frame_idx}")
            
            out.release()
            logger.info(f"Annotated video saved to: {output_video_path}")
            logger.info(f"Successfully rendered {frame_idx} frames")
            return True
            
        except Exception as e:
            logger.error(f"Video rendering failed: {e}")
            return False
    
    def _save_audit_log(self, audit_file: Path):
        """Save audit log with system information."""
        audit_data = {
            "input_hash": self._compute_file_hash(self.video_path),
            "config_snapshot": self._convert_config_to_serializable(self.config.__dict__),
            "software_versions": self._get_software_versions(),
            "gpu_info": self._get_gpu_info(),
            "seed_details": {
                "frame_idx": self.seed.frame_idx,
                "box": [float(x) for x in self.seed.box],  # Convert tuple to list of floats
                "time_s": float(self.seed.time_s)
            } if self.seed else None,
            "processing_metadata": {
                "timestamp": time.time(),
                "platform": platform.platform(),
                "python_version": platform.python_version()
            }
        }
        
        with open(audit_file, 'w') as f:
            json.dump(audit_data, f, indent=2)
    
    def _convert_config_to_serializable(self, obj):
        """Convert config object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._convert_config_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_config_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except:
            return "unknown"
    
    def _get_software_versions(self) -> Dict[str, str]:
        """Get software version information."""
        versions = {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "python": platform.python_version()
        }
        return versions
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        gpu_info = {"available": False}
        
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info["available"] = True
                gpu_info["nvidia_info"] = result.stdout.strip()
        except:
            pass
        
        return gpu_info


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Motion-First Bidirectional Tracker")
    parser.add_argument('command', choices=['full-run'], help='Tracking command')
    parser.add_argument('--config', nargs='+', default=['config.yaml'], 
                       help='Configuration files (base + overrides)')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--seed-frame', type=int, help='Seed frame index')
    parser.add_argument('--seed-box', help='Seed bounding box (x1,y1,x2,y2)')
    parser.add_argument('--run-dir', help='Output directory for this run')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = ConfigLoader.load_config(args.config)
        logger.info("Configuration loaded successfully")
        
        # Create run directory
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_dir = Path(f"runs/{timestamp}")
        
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run directory: {run_dir}")
        
        # Initialize tracker
        tracker = Tracker(config, args.video)
        if not tracker.initialize():
            sys.exit(1)
        
        # Set seed if provided
        if args.seed_frame is not None and args.seed_box:
            try:
                seed_box = tuple(map(float, args.seed_box.split(',')))
                if len(seed_box) != 4:
                    raise ValueError("Seed box must have 4 values: x1,y1,x2,y2")
                
                if not tracker.set_seed(args.seed_frame, seed_box):
                    sys.exit(1)
            except ValueError as e:
                logger.error(f"Invalid seed box format: {e}")
                sys.exit(1)
        else:
            logger.error("Both --seed-frame and --seed-box are required")
            sys.exit(1)
        
        # Run tracking
        if not tracker.run_tracking(run_dir):
            sys.exit(1)
        
        logger.info("Tracking completed successfully")
        
    except Exception as e:
        logger.error(f"Tracker failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
