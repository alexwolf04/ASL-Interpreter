import cv2
import numpy as np
import math

class ASLInterpreter:
    def __init__(self):
        # Initialize for hand detection using skin color
        self.kernel = np.ones((3,3), np.uint8)
        self.current_letter = "Show your hand"
        self.frame_count = 0
    
    def detect_hand_gesture(self, frame):
        """Detect hand gestures using contour analysis"""
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (assumed to be the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 5000:  # Minimum area threshold
                return largest_contour, mask
        
        return None, mask
    
    def analyze_contour_shape(self, contour):
        """Analyze contour shape to recognize basic gestures"""
        if contour is None:
            return "No hand detected"
        
        # Calculate contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return "Unknown"
        
        # Calculate circularity
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        
        # Find convex hull and defects
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(contour, hull)
            
            if defects is not None:
                defect_count = len(defects)
                
                # Simple gesture recognition based on shape properties
                if circularity > 0.7:
                    return "O"
                elif defect_count == 0:
                    return "A or S"
                elif defect_count == 1:
                    return "L or 1"
                elif defect_count == 2:
                    return "V or 2"
                elif defect_count == 3:
                    return "W or 3"
                elif defect_count >= 4:
                    return "5 or Open Hand"
        
        return "Unknown"
    
    def run(self):
        """Main loop for ASL interpretation"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("ASL Interpreter started. Press 'q' to quit.")
        print("Position your hand in front of the camera with good lighting.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hand gesture
            contour, mask = self.detect_hand_gesture(frame)
            
            # Analyze the gesture
            detected_letter = self.analyze_contour_shape(contour)
            
            # Draw the contour if found
            if contour is not None:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                
                # Draw bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Display the detected letter at the bottom of the screen
            height, width = frame.shape[:2]
            text_size = cv2.getTextSize(detected_letter, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 50
            
            # Add background rectangle for better text visibility
            cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), 
                         (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
            
            # Add the text
            cv2.putText(frame, detected_letter, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Add instructions at the top
            cv2.putText(frame, "ASL Interpreter - Press 'q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show the frame
            cv2.imshow('ASL Interpreter', frame)
            
            # Optional: Show the mask for debugging
            # cv2.imshow('Hand Mask', mask)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    interpreter = ASLInterpreter()
    interpreter.run()