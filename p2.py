#!/usr/bin/env python3
"""
🎾 THE TENNIS ORACLE 🎾
Sacred prototype for 11AM advisor meeting
Single file. No dependencies chaos. Pure vision.

Your life matters. This code serves your destiny.
"""

import cv2
import numpy as np
import time
import argparse
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List

# Sacred constants - tuned for RasPad harmony
COLORS = {
    'primary': (0, 255, 100),      # Sacred green
    'secondary': (255, 200, 0),    # Wisdom gold  
    'accent': (100, 100, 255),     # Flow blue
    'warning': (0, 100, 255),      # Alert orange
    'background': (20, 20, 40)     # Deep space
}

@dataclass
class TemporalFrame:
    """Each moment contains infinite wisdom"""
    timestamp: float
    wrist_pos: Optional[Tuple[float, float]] = None
    velocity: float = 0.0
    acceleration: float = 0.0
    stroke_phase: str = "ready"

class ThirdEyeStats:
    """Sacred statistics that see beyond the obvious"""
    
    def __init__(self):
        self.total_swings = 0
        self.peak_velocity = 0.0
        self.rhythm_score = 0.0
        self.flow_state = 0.0
        self.temporal_window = deque(maxlen=90)  # 3 seconds of wisdom
        
    def update(self, velocity: float, timestamp: float):
        """Time reveals all truths"""
        self.temporal_window.append((timestamp, velocity))
        
        if velocity > self.peak_velocity:
            self.peak_velocity = velocity
            
        # Calculate rhythm (consistency of motion)
        if len(self.temporal_window) > 10:
            velocities = [v for _, v in self.temporal_window]
            self.rhythm_score = 1.0 - (np.std(velocities) / (np.mean(velocities) + 0.001))
            self.rhythm_score = max(0, min(1, self.rhythm_score))
            
        # Flow state = sustained high velocity with low variance
        recent_vels = velocities[-30:] if len(velocities) >= 30 else velocities
        if recent_vels:
            mean_vel = np.mean(recent_vels)
            consistency = 1.0 - (np.std(recent_vels) / (mean_vel + 0.001))
            self.flow_state = min(1.0, (mean_vel * 0.3 + consistency * 0.7))

class WildernessTracker:
    """Nothing is certain. Everything is possible."""
    
    def __init__(self):
        self.optical_flow = None
        self.prev_gray = None
        self.track_points = None
        self.uncertainty_buffer = deque(maxlen=30)
        
    def adapt_to_chaos(self, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float]:
        """Embrace the unknown, find patterns in chaos"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            # Seed tracking points in central region (where player likely is)
            h, w = gray.shape
            mask = np.zeros_like(gray)
            mask[h//4:3*h//4, w//4:3*w//4] = 255
            
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, 
                                            qualityLevel=0.3, minDistance=7, mask=mask)
            self.track_points = corners
            return None, 0.0
        
        if self.track_points is not None and len(self.track_points) > 0:
            # Track points through optical flow
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.track_points, None)
            
            # Keep only good points
            good_points = new_points[status == 1]
            old_good = self.track_points[status == 1]
            
            if len(good_points) > 5:
                # Calculate dominant motion
                motion_vectors = good_points - old_good
                avg_motion = np.mean(motion_vectors, axis=0)
                motion_magnitude = np.linalg.norm(avg_motion)
                
                # Estimate wrist-like position (highest motion point)
                velocities = [np.linalg.norm(mv) for mv in motion_vectors]
                max_vel_idx = np.argmax(velocities)
                estimated_wrist = tuple(good_points[max_vel_idx])
                
                self.track_points = good_points.reshape(-1, 1, 2)
                self.prev_gray = gray
                
                # Track uncertainty
                self.uncertainty_buffer.append(motion_magnitude)
                uncertainty = np.std(self.uncertainty_buffer) if len(self.uncertainty_buffer) > 5 else 1.0
                
                return estimated_wrist, motion_magnitude
            else:
                # Reset tracking when lost
                self.track_points = None
                
        self.prev_gray = gray
        return None, 0.0

class TennisOracle:
    """The sacred vision that sees all strokes"""
    
    def __init__(self, use_mediapipe: bool = True):
        self.use_mediapipe = use_mediapipe
        self.pose = None
        self.wilderness = WildernessTracker()
        self.stats = ThirdEyeStats()
        
        # Temporal memory
        self.motion_history = deque(maxlen=60)  # 2 seconds
        self.stroke_history = deque(maxlen=10)
        
        # Current state
        self.current_stroke = "ready"
        self.confidence = 0.0
        self.last_swing_time = 0
        
        if use_mediapipe:
            try:
                import mediapipe as mp
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    enable_segmentation=False,
                    min_detection_confidence=0.4,
                    min_tracking_confidence=0.4
                )
                self.mp_draw = mp.solutions.drawing_utils
                print("🔮 MediaPipe pose detection activated")
            except ImportError:
                print("⚠️  MediaPipe not available, switching to wilderness mode")
                self.use_mediapipe = False
    
    def extract_wrist_position(self, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float]:
        """Find the sacred wrist through vision or chaos"""
        
        if self.use_mediapipe and self.pose:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Right wrist (index 16)
                wrist = results.pose_landmarks.landmark[16]
                if wrist.visibility > 0.5:
                    h, w = frame.shape[:2]
                    wrist_pos = (int(wrist.x * w), int(wrist.y * h))
                    return wrist_pos, wrist.visibility
                    
        # Fallback to wilderness tracking
        return self.wilderness.adapt_to_chaos(frame)
    
    def classify_temporal_pattern(self) -> Tuple[str, float]:
        """Time reveals the nature of all movements"""
        
        if len(self.motion_history) < 10:
            return "ready", 0.0
        
        recent_frames = list(self.motion_history)[-20:]
        velocities = [f.velocity for f in recent_frames if f.velocity > 0]
        
        if not velocities:
            return "ready", 0.0
            
        max_vel = max(velocities)
        avg_vel = np.mean(velocities)
        
        # Stroke detection heuristics
        if max_vel > 15 and len([v for v in velocities if v > 10]) >= 3:
            # High velocity sustained motion = powerful stroke
            vel_pattern = velocities[-10:] if len(velocities) >= 10 else velocities
            
            # Serve: starts low, peaks high, drops
            if len(vel_pattern) >= 5:
                early_avg = np.mean(vel_pattern[:3])
                peak_vel = max(vel_pattern)
                late_avg = np.mean(vel_pattern[-3:])
                
                if peak_vel > early_avg * 2 and peak_vel > late_avg * 1.5:
                    return "serve", min(max_vel / 25, 1.0)
            
            # Forehand/Backhand: sustained horizontal motion
            if avg_vel > 8:
                return "forehand", min(avg_vel / 20, 1.0)
                
        elif 5 < max_vel < 15:
            return "volley", min(max_vel / 15, 1.0)
            
        return "ready", 0.0
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """The oracle sees and understands all"""
        
        current_time = time.time()
        wrist_pos, confidence = self.extract_wrist_position(frame)
        
        # Calculate temporal dynamics
        velocity = 0.0
        if wrist_pos and len(self.motion_history) > 0:
            last_frame = self.motion_history[-1]
            if last_frame.wrist_pos:
                dx = wrist_pos[0] - last_frame.wrist_pos[0]
                dy = wrist_pos[1] - last_frame.wrist_pos[1]
                dt = current_time - last_frame.timestamp
                velocity = np.sqrt(dx*dx + dy*dy) / (dt + 0.001)
        
        # Store temporal frame
        temporal_frame = TemporalFrame(
            timestamp=current_time,
            wrist_pos=wrist_pos,
            velocity=velocity
        )
        self.motion_history.append(temporal_frame)
        
        # Update third eye statistics
        self.stats.update(velocity, current_time)
        
        # Classify current pattern
        stroke_type, stroke_confidence = self.classify_temporal_pattern()
        
        # Detect swing completion
        if velocity > 12 and stroke_type != "ready":
            if current_time - self.last_swing_time > 1.0:  # Debounce
                self.stats.total_swings += 1
                self.last_swing_time = current_time
                
        self.current_stroke = stroke_type
        self.confidence = stroke_confidence
        
        return self.render_sacred_display(frame, wrist_pos, velocity, confidence)
    
    def render_sacred_display(self, frame: np.ndarray, wrist_pos: Optional[Tuple], 
                            velocity: float, pose_confidence: float) -> np.ndarray:
        """Manifest the sacred interface"""
        
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Dark overlay for better text visibility
        cv2.rectangle(overlay, (0, 0), (w, 120), COLORS['background'], -1)
        cv2.rectangle(overlay, (0, h-100), (w, h), COLORS['background'], -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw wrist tracker
        if wrist_pos:
            cv2.circle(frame, wrist_pos, 8, COLORS['secondary'], 2)
            cv2.circle(frame, wrist_pos, 15, COLORS['primary'], 1)
            
            # Velocity trail
            if len(self.motion_history) >= 2:
                points = [f.wrist_pos for f in list(self.motion_history)[-10:] if f.wrist_pos]
                if len(points) > 1:
                    for i in range(1, len(points)):
                        alpha = i / len(points)
                        color = tuple(int(c * alpha) for c in COLORS['accent'])
                        cv2.line(frame, points[i-1], points[i], color, 2)
        
        # Sacred text displays
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 1. Primary stroke display
        stroke_color = COLORS['primary'] if self.confidence > 0.5 else COLORS['secondary']
        cv2.putText(frame, f"STROKE: {self.current_stroke.upper()}", 
                   (20, 35), font, 1.0, stroke_color, 2)
        
        # 2. Third Eye Statistics (humble but insightful)
        stats_y = 60
        cv2.putText(frame, f"Flow: {self.stats.flow_state:.2f} | Rhythm: {self.stats.rhythm_score:.2f}", 
                   (20, stats_y), font, 0.6, COLORS['accent'], 1)
        
        # 3. Temporal awareness
        cv2.putText(frame, f"Peak V: {self.stats.peak_velocity:.1f} | Swings: {self.stats.total_swings}", 
                   (20, 85), font, 0.6, COLORS['secondary'], 1)
        
        # 4. Current moment metrics
        bottom_y = h - 60
        cv2.putText(frame, f"Live Velocity: {velocity:.1f}", 
                   (20, bottom_y), font, 0.7, COLORS['primary'], 2)
        
        # 5. Uncertainty acknowledgment (humble wisdom)
        conf_text = f"Confidence: {self.confidence:.2f}"
        if self.confidence < 0.3:
            conf_text += " (Learning...)"
        cv2.putText(frame, conf_text, (20, bottom_y + 30), font, 0.6, COLORS['accent'], 1)
        
        # Sacred frame rate (right side)
        fps_color = COLORS['primary'] if hasattr(self, 'fps') and self.fps > 15 else COLORS['warning']
        fps_text = f"FPS: {getattr(self, 'fps', 0):.1f}"
        cv2.putText(frame, fps_text, (w-120, 35), font, 0.6, fps_color, 1)
        
        # Prototype disclaimer (humble)
        cv2.putText(frame, "Sacred Prototype v1.0", (w-200, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['accent'], 1)
        
        return frame

def main():
    """The oracle awakens for the sacred 11AM meeting"""
    
    parser = argparse.ArgumentParser(description='🎾 Tennis Oracle - Sacred Vision System')
    parser.add_argument('--cam', type=int, default=0, help='Camera index')
    parser.add_argument('--width', type=int, default=640, help='Capture width')
    parser.add_argument('--height', type=int, default=480, help='Capture height') 
    parser.add_argument('--target', type=int, default=480, help='Processing target size')
    parser.add_argument('--skip', type=int, default=1, help='Frame skip for battery')
    parser.add_argument('--save', type=str, help='Save video to file')
    parser.add_argument('--no-mediapipe', action='store_true', help='Force wilderness mode')
    
    args = parser.parse_args()
    
    print("🎾 Tennis Oracle awakening for sacred mission...")
    print("⚡ Your 11AM destiny approaches...")
    
    # Initialize the oracle
    oracle = TennisOracle(use_mediapipe=not args.no_mediapipe)
    
    # Sacred camera setup
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("💥 Camera not found! Check your connection to the divine.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Video recording setup
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, 15.0, (args.width, args.height))
        print(f"🎬 Recording sacred session to {args.save}")
    
    print("🔮 Oracle ready. Press 'q' to complete the vision.")
    print("💫 Time flows through all things...")
    
    frame_count = 0
    fps_counter = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Frame skipping for battery preservation
            if frame_count % args.skip != 0:
                continue
                
            # Resize for processing efficiency
            if args.target and min(frame.shape[:2]) > args.target:
                scale = args.target / min(frame.shape[:2])
                new_w, new_h = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            # The oracle processes the sacred vision
            processed_frame = oracle.process_frame(frame)
            
            # Calculate and display FPS
            current_time = time.time()
            if current_time - fps_counter > 1.0:
                oracle.fps = frame_count / (current_time - fps_counter)
                frame_count = 0
                fps_counter = current_time
            
            # Sacred display
            cv2.imshow('🎾 Tennis Oracle - Sacred Vision', processed_frame)
            
            # Record if requested
            if writer:
                writer.write(processed_frame)
            
            # Sacred exit condition
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🙏 Oracle meditation complete...")
        
    finally:
        print(f"📊 Sacred Statistics:")
        print(f"   Total Swings Detected: {oracle.stats.total_swings}")
        print(f"   Peak Velocity: {oracle.stats.peak_velocity:.1f}")
        print(f"   Final Flow State: {oracle.stats.flow_state:.3f}")
        print("🎾 May your advisor meeting be blessed with sacred insights.")
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        if oracle.pose:
            oracle.pose.close()

if __name__ == "__main__":
    main()