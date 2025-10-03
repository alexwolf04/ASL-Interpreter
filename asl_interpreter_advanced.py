import cv2
import numpy as np
import math
from collections import deque

class AdvancedASLInterpreter:
    def __init__(self):
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.kernel = np.ones((5,5), np.uint8)
        
        # Gesture recognition parameters
        self.gesture_buffer = deque(maxlen=10)  # Buffer for gesture stability
        self.current_gesture = "Show your hand"
        
        # Hand tracking parameters
        self.hand_cascade = None
        self.setup_hand_detection()
        
    def setup_hand_detection(self):
        """Setup hand detection using multiple methods"""
        # Try to load hand cascade (if available)
        try:
            self.hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')
        except:
            self.hand_cascade = None
    
    def preprocess_frame(self, frame):
        """Enhanced preprocessing for better hand detection"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Convert to different color spaces for better detection
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        
        # Enhanced skin detection using multiple color spaces
        # HSV skin detection
        lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
        upper_hsv = np.array([20, 150, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # YCrCb skin detection
        lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine masks
        mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
        
        # Advanced morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.medianBlur(mask, 5)
        
        return mask
    
    def find_hand_contour(self, mask, frame):
        """Find the best hand contour using multiple criteria"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by area and position
        valid_contours = []
        frame_height, frame_width = frame.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 8000 and area < frame_width * frame_height * 0.3:  # Reasonable hand size
                # Check if contour is roughly in the center area of frame
                x, y, w, h = cv2.boundingRect(contour)
                if w > 50 and h > 50:  # Minimum dimensions
                    valid_contours.append((contour, area))
        
        if not valid_contours:
            return None
        
        # Return the largest valid contour
        return max(valid_contours, key=lambda x: x[1])[0]
    
    def analyze_hand_shape(self, contour, frame):
        """Advanced hand shape analysis for gesture recognition"""
        if contour is None:
            return "No hand detected"
        
        # Calculate contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return "Unknown"
        
        # Get convex hull and defects
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) < 4:
            return "Unknown"
        
        defects = cv2.convexityDefects(contour, hull)
        
        # Calculate various shape metrics
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        extent = float(area) / (w * h)
        
        # Count significant defects (fingers)
        finger_count = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                # Calculate angle between vectors
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                
                if a > 0 and b > 0 and c > 0:
                    angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))
                    
                    # If angle is less than 90 degrees and defect is deep enough
                    if angle <= math.pi/2 and d > 10000:
                        finger_count += 1
        
        # Enhanced gesture recognition
        gesture = self.classify_gesture(finger_count, circularity, aspect_ratio, extent, area)
        
        # Add to buffer for stability
        self.gesture_buffer.append(gesture)
        
        # Return most common gesture in buffer
        if len(self.gesture_buffer) >= 5:
            gesture_counts = {}
            for g in self.gesture_buffer:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
            return max(gesture_counts, key=gesture_counts.get)
        
        return gesture
    
    def classify_gesture(self, finger_count, circularity, aspect_ratio, extent, area):
        """Classify gesture based on multiple features"""
        
        # Very circular shape
        if circularity > 0.8:
            return "O"
        
        # Closed fist (compact shape)
        if finger_count == 0 and extent > 0.6:
            if aspect_ratio < 1.3:
                return "A"
            else:
                return "S"
        
        # One finger
        if finger_count == 1:
            if aspect_ratio > 2.0:
                return "I"
            else:
                return "L"
        
        # Two fingers
        if finger_count == 2:
            if aspect_ratio > 1.5:
                return "V"
            else:
                return "U"
        
        # Three fingers
        if finger_count == 3:
            return "W"
        
        # Four fingers
        if finger_count == 4:
            return "4"
        
        # Open hand (5 fingers)
        if finger_count >= 5 or (extent < 0.5 and area > 15000):
            return "5"
        
        # Elongated shapes
        if aspect_ratio > 2.5:
            return "1"
        
        # Default based on finger count
        finger_map = {0: "A", 1: "1", 2: "2", 3: "3", 4: "4"}
        return finger_map.get(finger_count, "Unknown")
    
    def draw_hand_info(self, frame, contour, gesture):
        """Draw hand information on frame"""
        if contour is not None:
            # Draw contour
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            
            # Draw bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw convex hull
            hull = cv2.convexHull(contour)
            cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)
            
            # Draw center point
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)
    
    def run(self):
        """Main loop for advanced ASL interpretation"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Advanced ASL Interpreter started!")
        print("Instructions:")
        print("- Position your hand clearly in front of the camera")
        print("- Ensure good lighting")
        print("- Keep your hand steady for better recognition")
        print("- Press 'q' to quit, 'm' to show mask")
        
        show_mask = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Preprocess frame
            mask = self.preprocess_frame(frame)
            
            # Find hand contour
            contour = self.find_hand_contour(mask, frame)
            
            # Analyze gesture
            gesture = self.analyze_hand_shape(contour, frame)
            
            # Draw hand information
            self.draw_hand_info(frame, contour, gesture)
            
            # Display gesture text
            height, width = frame.shape[:2]
            text_size = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 60
            
            # Background for text
            cv2.rectangle(frame, (text_x - 15, text_y - text_size[1] - 15), 
                         (text_x + text_size[0] + 15, text_y + 15), (0, 0, 0), -1)
            
            # Main gesture text
            cv2.putText(frame, gesture, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            
            # Instructions
            cv2.putText(frame, "Advanced ASL Interpreter - 'q':quit 'm':mask", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frames
            cv2.imshow('Advanced ASL Interpreter', frame)
            
            if show_mask:
                cv2.imshow('Hand Mask', mask)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow('Hand Mask')
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    interpreter = AdvancedASLInterpreter()
    interpreter.run()