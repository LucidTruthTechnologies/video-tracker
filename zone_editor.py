#!/usr/bin/env python3
"""
Zone Editor Utility for Motion-First Bidirectional Tracker
Interactive polygon drawing tool for defining tracking zones.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Deque
import numpy as np
import cv2
from dataclasses import dataclass, field
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """Zone representation with polygons."""
    name: str
    polygons: List[List[Tuple[float, float]]] = field(default_factory=list)
    
    def add_polygon(self, polygon: List[Tuple[float, float]]):
        """Add a new polygon to this zone."""
        if len(polygon) >= 3:
            self.polygons.append(polygon)
    
    def remove_last_polygon(self) -> bool:
        """Remove the last polygon from this zone."""
        if self.polygons:
            self.polygons.pop()
            return True
        return False
    
    def remove_polygon_at(self, index: int) -> bool:
        """Remove polygon at specific index."""
        if 0 <= index < len(self.polygons):
            self.polygons.pop(index)
            return True
        return False


class ZoneEditor:
    """Interactive zone editor with polygon drawing capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.zones: Dict[str, Zone] = {}
        self.active_zone = "water"
        self.current_polygon: List[Tuple[float, float]] = []
        self.dragging_vertex = False
        self.dragged_vertex_idx = -1
        self.dragged_polygon_idx = -1
        self.undo_stack: Deque[Dict[str, Any]] = deque(maxlen=50)
        self.redo_stack: Deque[Dict[str, Any]] = deque(maxlen=50)
        self.show_help = True
        self.overlay_alpha = config.get('overlay_alpha', 0.35)
        self.vertex_radius = config.get('vertex_radius_px', 6)
        self.snap_px = config.get('snap_px', 8)
        
        # Initialize default zones
        self._init_default_zones()
        
        # Color palette
        self.colors = config.get('color_palette', [
            (0, 160, 255, 128),    # water
            (0, 200, 120, 128),    # deck
            (240, 180, 0, 128),    # bleachers
            (200, 0, 200, 128),    # walkway
        ])
        
        # Window state
        self.window_name = "Zone Editor"
        self.image = None
        self.source_width = 0
        self.source_height = 0
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
    def _init_default_zones(self):
        """Initialize default zones if none exist."""
        default_zones = ["water", "deck", "bleachers", "walkway"]
        for zone_name in default_zones:
            if zone_name not in self.zones:
                self.zones[zone_name] = Zone(name=zone_name)
    
    def load_image_from_video(self, video_path: str, frame_index: int) -> bool:
        """Load a specific frame from video file."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return False
            
            # Get video properties
            self.source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_index >= total_frames:
                logger.error(f"Frame index {frame_index} exceeds video length ({total_frames})")
                return False
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error(f"Failed to read frame {frame_index}")
                return False
            
            self.image = frame
            logger.info(f"Loaded frame {frame_index} from {video_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load frame from video: {e}")
            return False
    
    def load_image_from_file(self, image_path: str) -> bool:
        """Load image from file."""
        try:
            self.image = cv2.imread(image_path)
            if self.image is None:
                logger.error(f"Failed to load image: {image_path}")
                return False
            
            self.source_height, self.source_width = self.image.shape[:2]
            logger.info(f"Loaded image: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return False
    
    def load_zones(self, zones_file: str) -> bool:
        """Load existing zones from JSON file."""
        if not os.path.exists(zones_file):
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
                self.zones[zone_name] = Zone(name=zone_name, polygons=polygons)
            
            # Set active zone to first available
            if self.zones:
                self.active_zone = list(self.zones.keys())[0]
            
            logger.info(f"Loaded {len(self.zones)} zones from {zones_file}")
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
            logger.error("Zone dimensions don't match image dimensions")
            return False
        
        return True
    
    def save_zones(self, zones_file: str, frame_index: int = 0, timestamp: Optional[str] = None) -> bool:
        """Save zones to JSON file."""
        try:
            zones_data = {
                "version": "1.0",
                "source_width": self.source_width,
                "source_height": self.source_height,
                "frame_index_used": frame_index,
                "timestamp_used": timestamp,
                "zones": []
            }
            
            for zone_name, zone in self.zones.items():
                zone_data = {
                    "name": zone_name,
                    "polygons": zone.polygons
                }
                zones_data["zones"].append(zone_data)
            
            # Validate before saving
            if not self._validate_zones_for_save(zones_data):
                return False
            
            with open(zones_file, 'w') as f:
                json.dump(zones_data, f, indent=2)
            
            logger.info(f"Saved {len(self.zones)} zones to {zones_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save zones: {e}")
            return False
    
    def _validate_zones_for_save(self, zones_data: Dict[str, Any]) -> bool:
        """Validate zones before saving."""
        for zone in zones_data['zones']:
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
    
    def _save_state_for_undo(self):
        """Save current state for undo operation."""
        state = {
            'zones': {name: Zone(name=name, polygons=[p[:] for p in zone.polygons]) 
                     for name, zone in self.zones.items()},
            'active_zone': self.active_zone,
            'current_polygon': self.current_polygon[:]
        }
        self.undo_stack.append(state)
        self.redo_stack.clear()  # Clear redo when new action is performed
    
    def undo(self):
        """Undo last action."""
        if self.undo_stack:
            # Save current state to redo
            current_state = {
                'zones': {name: Zone(name=name, polygons=[p[:] for p in zone.polygons]) 
                         for name, zone in self.zones.items()},
                'active_zone': self.active_zone,
                'current_polygon': self.current_polygon[:]
            }
            self.redo_stack.append(current_state)
            
            # Restore previous state
            state = self.undo_stack.pop()
            self.zones = state['zones']
            self.active_zone = state['active_zone']
            self.current_polygon = state['current_polygon'][:]
            logger.info("Undo performed")
    
    def redo(self):
        """Redo last undone action."""
        if self.redo_stack:
            # Save current state to undo
            current_state = {
                'zones': {name: Zone(name=name, polygons=[p[:] for p in zone.polygon]) 
                         for name, zone in self.zones.items()},
                'active_zone': self.active_zone,
                'current_polygon': self.current_polygon[:]
            }
            self.undo_stack.append(current_state)
            
            # Restore redo state
            state = self.redo_stack.pop()
            self.zones = state['zones']
            self.active_zone = state['active_zone']
            self.current_polygon = state['current_polygon'][:]
            logger.info("Redo performed")
    
    def _screen_to_source_coords(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to source image coordinates."""
        source_x = (screen_x - self.offset_x) / self.scale_factor
        source_y = (screen_y - self.offset_y) / self.scale_factor
        return source_x, source_y
    
    def _source_to_screen_coords(self, source_x: float, source_y: float) -> Tuple[int, int]:
        """Convert source image coordinates to screen coordinates."""
        screen_x = int(source_x * self.scale_factor + self.offset_x)
        screen_y = int(source_y * self.scale_factor + self.offset_y)
        return screen_x, screen_y
    
    def _find_vertex_at_position(self, screen_x: int, screen_y: int) -> Tuple[int, int]:
        """Find vertex at screen position, returns (polygon_idx, vertex_idx)."""
        for zone_name, zone in self.zones.items():
            for poly_idx, polygon in enumerate(zone.polygons):
                for vert_idx, (sx, sy) in enumerate(polygon):
                    px, py = self._source_to_screen_coords(sx, sy)
                    if abs(px - screen_x) <= self.vertex_radius and abs(py - screen_y) <= self.vertex_radius:
                        return poly_idx, vert_idx
        return -1, -1
    
    def _find_polygon_at_position(self, screen_x: int, screen_y: int) -> Tuple[str, int]:
        """Find polygon at screen position, returns (zone_name, polygon_idx)."""
        source_x, source_y = self._screen_to_source_coords(screen_x, screen_y)
        
        for zone_name, zone in self.zones.items():
            for poly_idx, polygon in enumerate(zone.polygons):
                if self._point_in_polygon(source_x, source_y, polygon):
                    return zone_name, poly_idx
        
        return "", -1
    
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
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse event callback for OpenCV window."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_polygon:
                # Add vertex to current polygon
                source_x, source_y = self._screen_to_source_coords(x, y)
                self.current_polygon.append((source_x, source_y))
                self._save_state_for_undo()
                logger.info(f"Added vertex at ({source_x:.1f}, {source_y:.1f})")
            else:
                # Check if clicking on existing vertex
                poly_idx, vert_idx = self._find_vertex_at_position(x, y)
                if poly_idx >= 0:
                    # Start dragging vertex
                    self.dragging_vertex = True
                    self.dragged_polygon_idx = poly_idx
                    self.dragged_vertex_idx = vert_idx
                    self._save_state_for_undo()
                else:
                    # Check if clicking on polygon
                    zone_name, poly_idx = self._find_polygon_at_position(x, y)
                    if zone_name:
                        # Start dragging polygon
                        self.dragging_vertex = True
                        self.dragged_polygon_idx = poly_idx
                        self.dragged_vertex_idx = -1
                        self._save_state_for_undo()
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_vertex:
                if self.dragged_vertex_idx >= 0:
                    # Dragging specific vertex
                    source_x, source_y = self._screen_to_source_coords(x, y)
                    zone = list(self.zones.values())[0]  # Get first zone for now
                    if 0 <= self.dragged_polygon_idx < len(zone.polygons):
                        polygon = zone.polygons[self.dragged_polygon_idx]
                        if 0 <= self.dragged_vertex_idx < len(polygon):
                            polygon[self.dragged_vertex_idx] = (source_x, source_y)
                else:
                    # Dragging entire polygon
                    pass  # TODO: Implement polygon dragging
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_vertex:
                self.dragging_vertex = False
                self.dragged_polygon_idx = -1
                self.dragged_vertex_idx = -1
    
    def _draw_interface(self):
        """Draw the zone editor interface."""
        if self.image is None:
            return
        
        # Create display image
        display_img = self.image.copy()
        
        # Calculate scale to fit window
        window_width = 1200
        window_height = 800
        
        h, w = self.image.shape[:2]
        scale_x = window_width / w
        scale_y = window_height / h
        self.scale_factor = min(scale_x, scale_y)
        
        new_width = int(w * self.scale_factor)
        new_height = int(h * self.scale_factor)
        
        # Center the image
        self.offset_x = (window_width - new_width) // 2
        self.offset_y = (window_height - new_height) // 2
        
        # Scale image
        scaled_img = cv2.resize(self.image, (new_width, new_height))
        
        # Create full display image
        display_img = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        display_img[self.offset_y:self.offset_y + new_height, 
                   self.offset_x:self.offset_x + new_width] = scaled_img
        
        # Draw zones
        for zone_name, zone in self.zones.items():
            color = self.colors[list(self.zones.keys()).index(zone_name) % len(self.colors)]
            color_bgr = (color[2], color[1], color[0])  # Convert RGBA to BGR
            
            for polygon in zone.polygons:
                # Convert to screen coordinates
                screen_polygon = []
                for sx, sy in polygon:
                    px, py = self._source_to_screen_coords(sx, sy)
                    screen_polygon.append([px, py])
                
                if len(screen_polygon) >= 3:
                    # Draw filled polygon
                    overlay = display_img.copy()
                    cv2.fillPoly(overlay, [np.array(screen_polygon, dtype=np.int32)], color_bgr)
                    cv2.addWeighted(display_img, 1 - self.overlay_alpha, overlay, self.overlay_alpha, 0, display_img)
                    
                    # Draw polygon outline
                    cv2.polylines(display_img, [np.array(screen_polygon, dtype=np.int32)], 
                                True, color_bgr, 2)
                    
                    # Draw vertices
                    for i, (px, py) in enumerate(screen_polygon):
                        cv2.circle(display_img, (px, py), self.vertex_radius, color_bgr, -1)
                        cv2.circle(display_img, (px, py), self.vertex_radius, (255, 255, 255), 1)
                        
                        # Draw vertex index
                        cv2.putText(display_img, str(i), (px + 5, py - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw current polygon
        if self.current_polygon:
            color = self.colors[list(self.zones.keys()).index(self.active_zone) % len(self.colors)]
            color_bgr = (color[2], color[1], color[0])
            
            # Draw current polygon points
            for i, (sx, sy) in enumerate(self.current_polygon):
                px, py = self._source_to_screen_coords(sx, sy)
                cv2.circle(display_img, (px, py), self.vertex_radius, color_bgr, -1)
                cv2.circle(display_img, (px, py), self.vertex_radius, (255, 255, 255), 1)
                cv2.putText(display_img, str(i), (px + 5, py - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw lines between points
            if len(self.current_polygon) > 1:
                screen_polygon = []
                for sx, sy in self.current_polygon:
                    px, py = self._source_to_screen_coords(sx, sy)
                    screen_polygon.append([px, py])
                
                for i in range(len(screen_polygon) - 1):
                    cv2.line(display_img, tuple(screen_polygon[i]), tuple(screen_polygon[i + 1]), 
                            color_bgr, 2)
        
        # Draw active zone indicator
        zone_text = f"Active Zone: {self.active_zone}"
        cv2.putText(display_img, zone_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        cv2.putText(display_img, zone_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 0), 1)
        
        # Draw help overlay
        if self.show_help:
            self._draw_help_overlay(display_img)
        
        # Show image
        cv2.imshow(self.window_name, display_img)
    
    def _draw_help_overlay(self, img):
        """Draw help overlay on the image."""
        help_text = [
            "Zone Editor Controls:",
            "1-9: Select zone by index",
            "Tab: Next zone",
            "Left-click: Add vertex / Select vertex",
            "Enter/c: Close polygon",
            "n: New polygon in active zone",
            "Backspace/z: Undo",
            "Shift+z: Redo",
            "d: Delete vertex under cursor",
            "r: Remove last polygon in active zone",
            "x: Delete polygon under cursor",
            "s: Save zones.json",
            "l: Load zones.json",
            "h: Toggle help",
            "=/-: Change overlay alpha",
            "Esc/q: Quit"
        ]
        
        y_offset = 60
        for line in help_text:
            cv2.putText(img, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 0), 2)
            cv2.putText(img, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            y_offset += 20
    
    def run(self):
        """Run the zone editor main loop."""
        if self.image is None:
            logger.error("No image loaded")
            return
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        logger.info("Zone Editor started. Press 'h' for help.")
        
        while True:
            self._draw_interface()
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or Esc
                break
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord('1') or key == ord('2') or key == ord('3') or key == ord('4'):
                zone_idx = key - ord('1')
                zone_names = list(self.zones.keys())
                if zone_idx < len(zone_names):
                    self.active_zone = zone_names[zone_idx]
                    logger.info(f"Active zone: {self.active_zone}")
            elif key == ord('\t'):  # Tab
                zone_names = list(self.zones.keys())
                current_idx = zone_names.index(self.active_zone)
                next_idx = (current_idx + 1) % len(zone_names)
                self.active_zone = zone_names[next_idx]
                logger.info(f"Active zone: {self.active_zone}")
            elif key == ord('n'):
                # Start new polygon
                self.current_polygon = []
                logger.info(f"Started new polygon in zone: {self.active_zone}")
            elif key == ord('c') or key == 13:  # c or Enter
                if len(self.current_polygon) >= 3:
                    # Close current polygon
                    self.current_polygon.append(self.current_polygon[0])  # Close loop
                    self.zones[self.active_zone].add_polygon(self.current_polygon[:-1])  # Don't duplicate first point
                    self.current_polygon = []
                    self._save_state_for_undo()
                    logger.info(f"Closed polygon in zone: {self.active_zone}")
                else:
                    logger.warning("Need at least 3 vertices to close polygon")
            elif key == ord('z') and cv2.waitKey(1) & 0xFF == ord('Z'):  # Shift+z
                self.redo()
            elif key == ord('z'):
                self.undo()
            elif key == ord('r'):
                if self.zones[self.active_zone].remove_last_polygon():
                    self._save_state_for_undo()
                    logger.info(f"Removed last polygon from zone: {self.active_zone}")
            elif key == ord('='):
                self.overlay_alpha = min(1.0, self.overlay_alpha + 0.05)
                logger.info(f"Overlay alpha: {self.overlay_alpha:.2f}")
            elif key == ord('-'):
                self.overlay_alpha = max(0.0, self.overlay_alpha - 0.05)
                logger.info(f"Overlay alpha: {self.overlay_alpha:.2f}")
        
        cv2.destroyAllWindows()


def main():
    """Main entry point for zone editor."""
    parser = argparse.ArgumentParser(description="Zone Editor for Motion Tracker")
    parser.add_argument('zone-edit', help='Zone editing command')
    parser.add_argument('--video', help='Input video file')
    parser.add_argument('--frame-index', type=int, default=0, help='Frame index to load')
    parser.add_argument('--timestamp', help='Timestamp to load (HH:MM:SS.ff)')
    parser.add_argument('--image', help='Input image file (alternative to video)')
    parser.add_argument('--zones-in', help='Input zones file to load')
    parser.add_argument('--zones-out', default='zones.json', help='Output zones file')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if os.path.exists(args.config):
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # Initialize zone editor
        editor = ZoneEditor(config.get('zone_editor', {}))
        
        # Load image or video frame
        if args.image:
            if not editor.load_image_from_file(args.image):
                sys.exit(1)
        elif args.video:
            if not editor.load_image_from_video(args.video, args.frame_index):
                sys.exit(1)
        else:
            logger.error("Must specify either --video or --image")
            sys.exit(1)
        
        # Load existing zones if specified
        if args.zones_in:
            editor.load_zones(args.zones_in)
        
        # Run editor
        editor.run()
        
        # Save zones
        if editor.save_zones(args.zones_out, args.frame_index):
            logger.info(f"Zones saved to {args.zones_out}")
        else:
            logger.error("Failed to save zones")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Zone editor failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
