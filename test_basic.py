#!/usr/bin/env python3
"""
Basic test script for Motion-First Bidirectional Tracker
Tests core functionality without requiring video files.
"""

import unittest
import tempfile
import json
import os
import numpy as np
import cv2
from pathlib import Path

# Add current directory to path for imports
import sys
sys.path.insert(0, '.')

from tracker import (
    Config, Zone, TrackingState, Seed, ConfigLoader, 
    ZoneManager, OpticalFlow, KalmanTracker
)


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of tracker components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create minimal config
        self.config = Config(
            io={'work_height': 1080, 'constant_fps': True},
            zones_file='test_zones.json',
            zone_params={
                'water': {'speed_px_s_max': 2400, 'process_noise_scale': 1.5},
                'deck': {'speed_px_s_max': 1200, 'process_noise_scale': 1.0}
            },
            flow={'engine': 'farneback', 'dilate_box_px_work': 12},
            tracker={'dt': 0.033, 'measurement_noise': {'center_px_work': 2.5}},
            detector={'enable': False},
            render={'draw_thickness': 4},
            qc={'export_low_conf_clips': True},
            zone_editor={'overlay_alpha': 0.35},
            outputs={'traj_csv': 'test.csv'}
        )
        
        # Create test image
        self.test_image = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
        
    def test_zone_creation(self):
        """Test zone creation and validation."""
        zone = Zone(
            name="test_zone",
            polygons=[[(100, 100), (200, 100), (200, 200), (100, 200)]],
            params={'speed_px_s_max': 1000}
        )
        
        self.assertEqual(zone.name, "test_zone")
        self.assertEqual(len(zone.polygons), 1)
        self.assertEqual(len(zone.polygons[0]), 4)
        
        # Test point containment
        self.assertTrue(zone.contains_point(150, 150))
        self.assertFalse(zone.contains_point(300, 300))
    
    def test_seed_validation(self):
        """Test seed validation."""
        # Valid seed
        valid_seed = Seed(frame_idx=100, box=(100, 100, 200, 200), time_s=3.33)
        self.assertTrue(valid_seed.validate(3840, 2160))
        
        # Invalid seed (out of bounds)
        invalid_seed = Seed(frame_idx=100, box=(4000, 100, 4100, 200), time_s=3.33)
        self.assertFalse(invalid_seed.validate(3840, 2160))
        
        # Invalid seed (negative coordinates)
        invalid_seed2 = Seed(frame_idx=100, box=(-100, 100, 200, 200), time_s=3.33)
        self.assertFalse(invalid_seed2.validate(3840, 2160))
    
    def test_zone_manager(self):
        """Test zone manager functionality."""
        zone_manager = ZoneManager(self.config, 3840, 2160)
        
        # Test without zones
        zone, overlap = zone_manager.get_active_zone((100, 100, 200, 200))
        self.assertEqual(zone, "unknown")
        self.assertEqual(overlap, 0.0)
        
        # Test with zones
        zones_data = {
            "version": "1.0",
            "source_width": 3840,
            "source_height": 2160,
            "frame_index_used": 100,
            "zones": [
                {
                    "name": "water",
                    "polygons": [[(0, 0), (1920, 0), (1920, 1080), (0, 1080)]]
                },
                {
                    "name": "deck", 
                    "polygons": [[(1920, 0), (3840, 0), (3840, 1080), (1920, 1080)]]
                }
            ]
        }
        
        # Save test zones file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(zones_data, f)
            zones_file = f.name
        
        try:
            # Load zones
            success = zone_manager.load_zones(zones_file)
            self.assertTrue(success)
            self.assertEqual(len(zone_manager.zones), 2)
            
            # Test zone detection
            zone, overlap = zone_manager.get_active_zone((100, 100, 200, 200))
            self.assertEqual(zone, "water")
            self.assertGreater(overlap, 0.0)
            
            zone, overlap = zone_manager.get_active_zone((2000, 100, 2100, 200))
            self.assertEqual(zone, "deck")
            self.assertGreater(overlap, 0.0)
            
        finally:
            # Clean up
            os.unlink(zones_file)
    
    def test_optical_flow(self):
        """Test optical flow computation."""
        flow_engine = OpticalFlow(self.config.flow)
        
        # Create test frames
        frame1 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Compute flow
        flow = flow_engine.compute_flow(frame1, frame2)
        
        self.assertEqual(flow.shape, (1080, 1920, 2))
        self.assertEqual(flow.dtype, np.float32)
        
        # Test flow sampling
        box = (100, 100, 200, 200)
        flow_info = flow_engine.sample_flow_in_box(flow, box)
        
        self.assertIn('center_shift', flow_info)
        self.assertIn('rim_divergence', flow_info)
        self.assertIn('flow_agree', flow_info)
        self.assertEqual(len(flow_info['center_shift']), 2)
    
    def test_kalman_tracker(self):
        """Test Kalman tracker functionality."""
        tracker = KalmanTracker(self.config.tracker, 30.0)  # 30 FPS
        
        # Test initial state
        initial_state = tracker.get_state()
        self.assertEqual(len(initial_state), 8)  # [cx, cy, w, h, vx, vy, vw, vh]
        
        # Test prediction
        predicted_state = tracker.predict()
        self.assertEqual(len(predicted_state), 8)
        
        # Test update
        measurement = np.array([100.0, 200.0, 50.0, 100.0])  # [cx, cy, w, h]
        updated_state = tracker.update(measurement, "water")
        self.assertEqual(len(updated_state), 8)
        
        # Test state setting
        new_state = np.array([150.0, 250.0, 60.0, 110.0, 10.0, 5.0, 1.0, 2.0])
        tracker.set_state(new_state)
        current_state = tracker.get_state()
        np.testing.assert_array_almost_equal(current_state, new_state, decimal=5)
    
    def test_config_loader(self):
        """Test configuration loading."""
        # Create test config file
        test_config = {
            'io': {'work_height': 720},
            'zones_file': 'test.json',
            'zone_params': {'test': {'speed': 1000}},
            'flow': {'engine': 'farneback'},
            'tracker': {'dt': 0.05},
            'detector': {'enable': False},
            'render': {'thickness': 2},
            'qc': {'export': True},
            'zone_editor': {'alpha': 0.5},
            'outputs': {'csv': 'test.csv'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(test_config, f)
            config_file = f.name
        
        try:
            # Load config
            config = ConfigLoader.load_config([config_file])
            
            self.assertEqual(config.io['work_height'], 720)
            self.assertEqual(config.zones_file, 'test.json')
            self.assertEqual(config.flow['engine'], 'farneback')
            
        finally:
            # Clean up
            os.unlink(config_file)


def run_tests():
    """Run all tests."""
    # Create test runner
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestBasicFunctionality)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running basic functionality tests...")
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
