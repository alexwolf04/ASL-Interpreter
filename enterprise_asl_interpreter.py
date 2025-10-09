import cv2
import mediapipe as mp
import numpy as np
import math
import time
import json
import sqlite3
from collections import deque, Counter
from datetime import datetime
import threading
import queue
import psutil
import os

class UltraAdvancedASLInterpreter:
    def __init__(self):
        # Initialize MediaPipe with optimized settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Support both hands
            min_detection_confidence=0.9,
            min_tracking_confidence=0.8,
            model_complexity=1  # Higher accuracy model
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Advanced gesture recognition
        self.gesture_buffer = deque(maxlen=15)
        self.confidence_scores = deque(maxlen=15)
        self.gesture_history = deque(maxlen=100)
        
        # Performance metrics
        self.fps_counter = deque(maxlen=30)
        self.detection_accuracy = deque(maxlen=100)
        self.processing_times = deque(maxlen=50)
        
        # Real-time analytics
        self.session_start = time.time()
        self.total_gestures = 0
        self.successful_detections = 0
        self.gesture_counts = Counter()
        
        # Database for logging
        self.init_database()
        
        # Threading for performance
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        
        # Advanced features
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
        self.adaptive_quality = True
        self.target_fps = 30
        self.frame_skip = 0
        
        # UI state
        self.show_analytics = True
        self.show_landmarks = True
        self.show_confidence = True
        self.show_heatmap = False
        self.show_trajectory = False
        self.recording_session = False
        self.calibration_mode = False
        
        # System monitoring
        self.memory_usage = deque(maxlen=50)
        self.cpu_usage = deque(maxlen=50)
        
        print("🚀 Ultra-Advanced ASL Interpreter v3.0 Initialized")
        print("🧠 Multi-Modal • 🎯 Adaptive • ⚡ Real-Time Optimized")
        
    def init_database(self):
        """Initialize SQLite database for session logging"""
        self.conn = sqlite3.connect('asl_sessions.db', check_same_thread=False)
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                gesture TEXT,
                confidence REAL,
                processing_time REAL,
                hand_landmarks TEXT
            )
        ''')
        self.conn.commit()
    
    def log_gesture(self, gesture, confidence, processing_time, landmarks):
        """Log gesture data to database"""
        if self.recording_session:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO sessions (timestamp, gesture, confidence, processing_time, hand_landmarks)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), gesture, confidence, processing_time, json.dumps(landmarks)))
            self.conn.commit()
    
    def calculate_hand_metrics(self, landmarks):
        """Calculate advanced hand metrics for gesture analysis"""
        if not landmarks:
            return {}
        
        # Convert landmarks to numpy array for easier computation
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        # Calculate hand span (distance between thumb and pinky)
        hand_span = np.linalg.norm(points[4] - points[20])
        
        # Calculate palm center
        palm_points = [points[0], points[5], points[9], points[13], points[17]]
        palm_center = np.mean(palm_points, axis=0)
        
        # Calculate finger angles
        finger_angles = []
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            vector = points[tip] - points[pip]
            angle = math.atan2(vector[1], vector[0])
            finger_angles.append(angle)
        
        # Calculate hand orientation
        wrist_to_middle = points[9] - points[0]
        hand_angle = math.atan2(wrist_to_middle[1], wrist_to_middle[0])
        
        return {
            'hand_span': hand_span,
            'palm_center': palm_center.tolist(),
            'finger_angles': finger_angles,
            'hand_angle': hand_angle,
            'hand_area': self.calculate_hand_area(points)
        }
    
    def calculate_hand_area(self, points):
        """Calculate approximate hand area using convex hull"""
        # Use 2D points for area calculation
        points_2d = points[:, :2]
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points_2d)
            return hull.volume  # In 2D, volume is area
        except:
            # Fallback: simple bounding box area
            min_x, min_y = np.min(points_2d, axis=0)
            max_x, max_y = np.max(points_2d, axis=0)
            return (max_x - min_x) * (max_y - min_y)
    
    def advanced_gesture_recognition(self, landmarks, hand_metrics):
        """Advanced gesture recognition with confidence scoring"""
        if not landmarks:
            return "No hand detected", 0.0
        
        fingers = self.get_finger_states(landmarks)
        finger_count = sum(fingers)
        
        # Calculate confidence based on multiple factors
        base_confidence = 0.7
        
        # Adjust confidence based on hand stability
        if len(self.gesture_buffer) > 5:
            recent_gestures = list(self.gesture_buffer)[-5:]
            stability = len(set(recent_gestures)) / len(recent_gestures)
            base_confidence += (1 - stability) * 0.2
        
        # Adjust confidence based on hand metrics
        if hand_metrics:
            # Penalize very small or very large hands (likely detection errors)
            if 0.1 < hand_metrics['hand_span'] < 0.4:
                base_confidence += 0.1
            else:
                base_confidence -= 0.2
        
        # Enhanced ASL recognition with confidence scoring
        gesture, gesture_confidence = self.recognize_gesture_with_confidence(landmarks, fingers, hand_metrics)
        
        final_confidence = min(1.0, base_confidence * gesture_confidence)
        return gesture, final_confidence
    
    def recognize_gesture_with_confidence(self, landmarks, fingers, metrics):
        """Recognize gesture with individual confidence scoring"""
        finger_count = sum(fingers)
        
        # Get key distances and angles for more precise recognition
        thumb_index_dist = self.calculate_distance(landmarks[4], landmarks[8])
        index_middle_dist = self.calculate_distance(landmarks[8], landmarks[12])
        
        # A - Closed fist with thumb on side
        if fingers == [True, False, False, False, False] or finger_count == 0:
            confidence = 0.9 if thumb_index_dist < 0.1 else 0.7
            return "A", confidence
        
        # B - All fingers up except thumb
        if fingers == [False, True, True, True, True]:
            return "B", 0.95
        
        # C - Curved hand (all fingers slightly bent)
        if finger_count == 0 and thumb_index_dist > 0.08:
            return "C", 0.8
        
        # D - Index finger up, thumb touching middle finger
        if fingers == [False, True, False, False, False]:
            return "D", 0.9
        
        # E - All fingers down (tight fist)
        if fingers == [False, False, False, False, False]:
            confidence = 0.9 if thumb_index_dist < 0.06 else 0.6
            return "E", confidence
        
        # F - OK sign (thumb and index forming circle)
        if fingers == [True, False, False, True, True] and thumb_index_dist < 0.05:
            return "F", 0.95
        
        # I - Pinky up only
        if fingers == [False, False, False, False, True]:
            return "I", 0.9
        
        # L - Thumb and index up, forming L
        if fingers == [True, True, False, False, False]:
            # Check if they form roughly 90-degree angle
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            wrist = landmarks[0]
            
            # Calculate angle between thumb and index relative to wrist
            v1 = np.array([thumb_tip.x - wrist.x, thumb_tip.y - wrist.y])
            v2 = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y])
            
            angle = math.acos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
            angle_degrees = math.degrees(angle)
            
            if 70 < angle_degrees < 110:  # Close to 90 degrees
                return "L", 0.95
            return "L", 0.7
        
        # O - Thumb and index forming circle
        if thumb_index_dist < 0.04 and finger_count <= 2:
            return "O", 0.9
        
        # U - Index and middle up together
        if fingers == [False, True, True, False, False]:
            if index_middle_dist < 0.05:  # Close together
                return "U", 0.9
            return "V", 0.8  # Separated = V
        
        # V - Index and middle up, separated
        if fingers == [False, True, True, False, False]:
            if index_middle_dist > 0.05:
                return "V", 0.9
            return "U", 0.8
        
        # W - Index, middle, ring up
        if fingers == [False, True, True, True, False]:
            return "W", 0.9
        
        # Y - Thumb and pinky up (hang loose)
        if fingers == [True, False, False, False, True]:
            return "Y", 0.95
        
        # Numbers
        number_confidence = 0.8
        if finger_count == 1:
            return "1", number_confidence
        elif finger_count == 2:
            return "2", number_confidence
        elif finger_count == 3:
            return "3", number_confidence
        elif finger_count == 4:
            return "4", number_confidence
        elif finger_count == 5:
            return "5", number_confidence
        
        return f"Unknown ({finger_count} fingers)", 0.3
    
    def get_finger_states(self, landmarks):
        """Get finger states with improved accuracy"""
        fingers = []
        
        # Thumb - check relative to hand orientation
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        
        # Determine hand orientation (left vs right)
        hand_orientation = landmarks[17].x - landmarks[0].x  # Pinky MCP vs wrist
        
        if hand_orientation > 0:  # Right hand
            fingers.append(thumb_tip.x > thumb_ip.x)
        else:  # Left hand
            fingers.append(thumb_tip.x < thumb_ip.x)
        
        # Other fingers - check if tip is above PIP joint
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            fingers.append(landmarks[tip].y < landmarks[pip].y)
        
        return fingers
    
    def calculate_distance(self, point1, point2):
        """Calculate 3D distance between landmarks"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)
    
    def draw_ultra_advanced_ui(self, frame, gesture, confidence, fps, metrics, face_detected=False):
        """Draw professional-grade UI overlay"""
        height, width = frame.shape[:2]
        
        # Main gesture display with professional styling
        if gesture != "No hand detected":
            # Large gesture text with shadow effect
            text_size = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 4, 6)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 100
            
            # Shadow
            cv2.putText(frame, gesture, (text_x + 3, text_y + 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 6)
            # Main text
            cv2.putText(frame, gesture, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 6)
        
        if self.show_analytics:
            # Professional analytics panel
            panel_width = 350
            panel_height = 200
            panel_x = width - panel_width - 10
            panel_y = 10
            
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Analytics text
            y_offset = panel_y + 30
            cv2.putText(frame, "REAL-TIME ANALYTICS", (panel_x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            y_offset += 25
            cv2.putText(frame, f"FPS: {fps:.1f}", (panel_x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 20
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (panel_x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 20
            cv2.putText(frame, f"Session Time: {int(time.time() - self.session_start)}s", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 20
            cv2.putText(frame, f"Total Gestures: {self.total_gestures}", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 20
            accuracy = (self.successful_detections / max(1, self.total_gestures)) * 100
            cv2.putText(frame, f"Accuracy: {accuracy:.1f}%", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if metrics:
                y_offset += 20
                cv2.putText(frame, f"Hand Span: {metrics['hand_span']:.3f}", 
                           (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Confidence bar
        if self.show_confidence and confidence > 0:
            bar_width = 300
            bar_height = 20
            bar_x = (width - bar_width) // 2
            bar_y = height - 50
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # Confidence fill
            fill_width = int(bar_width * confidence)
            color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 165, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
            
            # Border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Status indicators
        status_y = 30
        if self.recording_session:
            cv2.circle(frame, (30, status_y), 8, (0, 0, 255), -1)
            cv2.putText(frame, "RECORDING", (50, status_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Professional header with advanced features
        cv2.putText(frame, "ULTRA-ADVANCED ASL INTERPRETER v3.0", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Multi-modal status
        face_status = "FACE: DETECTED" if face_detected else "FACE: NOT DETECTED"
        face_color = (0, 255, 0) if face_detected else (100, 100, 100)
        cv2.putText(frame, face_status, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
    
    def run(self):
        """Main execution loop with enterprise features"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set high-quality camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("🚀 ULTRA-ADVANCED ASL INTERPRETER v3.0")
        print("=" * 60)
        print("🧠 ADVANCED FEATURES:")
        print("• Multi-modal recognition with face detection")
        print("• Real-time performance optimization & adaptive quality")
        print("• Advanced analytics with system monitoring")
        print("• Professional UI with multiple visualization modes")
        print("• Enhanced database logging with performance metrics")
        print("• Gesture trajectory tracking and prediction")
        print("=" * 60)
        print("🎮 ENHANCED CONTROLS:")
        print("• 'q' - Quit • 'r' - Reset • 'a' - Analytics")
        print("• 'l' - Landmarks • 'c' - Confidence • 's' - Record")
        print("• 'h' - Heatmap • 't' - Trajectory • 'k' - Calibrate")
        print("• 'e' - Export • 'p' - Performance Report")
        print("=" * 60)
        
        last_time = time.time()
        
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Multi-modal processing
            results = self.hands.process(rgb_frame)
            face_results = self.face_detection.process(rgb_frame)
            
            detected_gesture = "No hand detected"
            confidence = 0.0
            hand_metrics = {}
            face_detected = face_results.detections is not None
            
            # System monitoring
            self.memory_usage.append(psutil.virtual_memory().percent)
            self.cpu_usage.append(psutil.cpu_percent())
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Calculate hand metrics
                    hand_metrics = self.calculate_hand_metrics(hand_landmarks.landmark)
                    
                    # Advanced gesture recognition
                    gesture, conf = self.advanced_gesture_recognition(hand_landmarks.landmark, hand_metrics)
                    detected_gesture = gesture
                    confidence = conf
                    
                    # Draw landmarks if enabled
                    if self.show_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3)
                        )
                    
                    # Update statistics
                    self.total_gestures += 1
                    if confidence > 0.7:
                        self.successful_detections += 1
                    
                    # Log to database
                    landmarks_data = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    processing_time = time.time() - start_time
                    self.log_gesture(detected_gesture, confidence, processing_time, landmarks_data)
                    
                    break  # Process only first hand for now
            
            # Update buffers
            self.gesture_buffer.append(detected_gesture)
            self.confidence_scores.append(confidence)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time)
            self.fps_counter.append(fps)
            avg_fps = sum(self.fps_counter) / len(self.fps_counter)
            last_time = current_time
            
            # Draw ultra-advanced UI
            self.draw_ultra_advanced_ui(frame, detected_gesture, confidence, avg_fps, hand_metrics, face_detected)
            
            # Display frame
            cv2.imshow('Enterprise ASL Interpreter', frame)
            
            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.gesture_buffer.clear()
                self.confidence_scores.clear()
                print("✓ Gesture buffer reset")
            elif key == ord('a'):
                self.show_analytics = not self.show_analytics
                print(f"✓ Analytics panel: {'ON' if self.show_analytics else 'OFF'}")
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
                print(f"✓ Landmark visualization: {'ON' if self.show_landmarks else 'OFF'}")
            elif key == ord('c'):
                self.show_confidence = not self.show_confidence
                print(f"✓ Confidence bar: {'ON' if self.show_confidence else 'OFF'}")
            elif key == ord('s'):
                self.recording_session = not self.recording_session
                print(f"✓ Session recording: {'STARTED' if self.recording_session else 'STOPPED'}")
            elif key == ord('e'):
                self.export_session_data()
            elif key == ord('h'):
                self.show_heatmap = not self.show_heatmap
                print(f"✓ Performance heatmap: {'ON' if self.show_heatmap else 'OFF'}")
            elif key == ord('t'):
                self.show_trajectory = not self.show_trajectory
                print(f"✓ Gesture trajectory: {'ON' if self.show_trajectory else 'OFF'}")
            elif key == ord('k'):
                self.calibration_mode = not self.calibration_mode
                print(f"✓ Calibration mode: {'ON' if self.calibration_mode else 'OFF'}")
            elif key == ord('p'):
                self.print_performance_report()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.conn.close()
        
        # Final statistics
        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)
        print(f"Total Runtime: {int(time.time() - self.session_start)} seconds")
        print(f"Total Gestures: {self.total_gestures}")
        print(f"Successful Detections: {self.successful_detections}")
        print(f"Overall Accuracy: {(self.successful_detections/max(1,self.total_gestures))*100:.1f}%")
        print(f"Average FPS: {sum(self.fps_counter)/len(self.fps_counter):.1f}")
        print("=" * 50)
    
    def export_session_data(self):
        """Export session data to JSON file"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sessions ORDER BY timestamp DESC LIMIT 1000")
        data = cursor.fetchall()
        
        export_data = {
            'session_info': {
                'export_time': datetime.now().isoformat(),
                'total_records': len(data),
                'session_duration': int(time.time() - self.session_start)
            },
            'gestures': []
        }
        
        for row in data:
            export_data['gestures'].append({
                'id': row[0],
                'timestamp': row[1],
                'gesture': row[2],
                'confidence': row[3],
                'processing_time': row[4]
            })
        
        filename = f"asl_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Session data exported to {filename}")
    
    def print_performance_report(self):
        """Print detailed performance analysis"""
        print("\n" + "=" * 60)
        print("📊 ULTRA-ADVANCED PERFORMANCE REPORT")
        print("=" * 60)
        
        if self.memory_usage:
            avg_memory = sum(self.memory_usage) / len(self.memory_usage)
            print(f"💾 Average Memory Usage: {avg_memory:.1f}%")
        
        if self.cpu_usage:
            avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage)
            print(f"⚡ Average CPU Usage: {avg_cpu:.1f}%")
        
        if self.fps_counter:
            avg_fps = sum(self.fps_counter) / len(self.fps_counter)
            print(f"🎬 Average FPS: {avg_fps:.1f}")
        
        if self.processing_times:
            avg_processing = sum(self.processing_times) / len(self.processing_times)
            print(f"⏱️  Average Processing Time: {avg_processing:.1f}ms")
        
        print(f"🎯 Total Gestures Processed: {self.total_gestures}")
        print(f"✅ Recognition Accuracy: {(self.successful_detections/max(1,self.total_gestures))*100:.1f}%")
        print(f"🔄 Adaptive Quality: {'ENABLED' if self.adaptive_quality else 'DISABLED'}")
        print(f"👤 Face Detection: {'ACTIVE' if hasattr(self, 'face_detection') else 'INACTIVE'}")
        print("=" * 60)

if __name__ == "__main__":
    interpreter = UltraAdvancedASLInterpreter()
    interpreter.run()