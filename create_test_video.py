#!/usr/bin/env python3
"""
Create a simple test video for testing the forward tracking pipeline.
Generates a video with a moving rectangle that crosses different zones.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

def create_test_video(output_path: str, duration: float = 10.0, fps: float = 30.0):
    """Create a test video with a moving object."""
    
    # Video parameters
    width, height = 1920, 1080  # 1080p for testing
    total_frames = int(duration * fps)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Object parameters
    object_width = 100
    object_height = 80
    
    # Motion parameters - object moves from left to right
    start_x = 100
    end_x = width - 200
    start_y = height // 2
    end_y = height // 2
    
    print(f"Creating test video: {width}x{height} @ {fps}fps for {duration}s")
    print(f"Object moves from ({start_x}, {start_y}) to ({end_x}, {end_y})")
    
    for frame_idx in range(total_frames):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 128
        
        # Calculate object position (linear motion)
        progress = frame_idx / (total_frames - 1)
        current_x = int(start_x + progress * (end_x - start_x))
        current_y = int(start_y + progress * (end_y - start_y))
        
        # Add some sinusoidal motion for more interesting tracking
        current_y += int(50 * np.sin(progress * 4 * np.pi))
        
        # Draw object (red rectangle)
        x1 = max(0, current_x - object_width // 2)
        y1 = max(0, current_y - object_height // 2)
        x2 = min(width, current_x + object_width // 2)
        y2 = min(height, current_y + object_height // 2)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = frame_idx / fps
        cv2.putText(frame, f"Time: {timestamp:.2f}s", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add object position
        cv2.putText(frame, f"Pos: ({current_x}, {current_y})", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        if frame_idx % 30 == 0:
            print(f"Generated frame {frame_idx}/{total_frames}")
    
    out.release()
    print(f"Test video saved to: {output_path}")
    print(f"Object trajectory: starts at ({start_x}, {start_y}), ends at ({end_x}, {end_y})")
    print(f"Seed frame 0: object at ({start_x}, {start_y}) with size {object_width}x{object_height}")

def main():
    parser = argparse.ArgumentParser(description="Create test video for tracking")
    parser.add_argument('--output', default='test_video.mp4', help='Output video file')
    parser.add_argument('--duration', type=float, default=10.0, help='Video duration in seconds')
    parser.add_argument('--fps', type=float, default=30.0, help='Video FPS')
    
    args = parser.parse_args()
    
    create_test_video(args.output, args.duration, args.fps)

if __name__ == "__main__":
    main()
