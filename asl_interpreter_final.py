import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

class MediaPipeASLInterpreter:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture recognition
        self.gesture_buffer = deque(maxlen=8)
        self.current_gesture = "Show your hand"
        
    def calculate_distance(self, point1, point2):
        """Calculate distance between two landmarks"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_finger_up(self, landmarks, finger_tip, finger_pip, finger_mcp=None):
        """Check if a finger is extended"""
        tip = landmarks[finger_tip]
        pip = landmarks[finger_pip]
        
        # For thumb (special case)
        if finger_tip == 4:
            # Check if thumb is extended horizontally
            return tip.x > landmarks[3].x if landmarks[0].x < landmarks[17].x else tip.x < landmarks[3].x
        
        # For other fingers, check if tip is above pip
        return tip.y < pip.y
    
    def get_finger_states(self, landmarks):
        """Get the state of all fingers (up/down)"""
        fingers = []
        
        # Thumb
        fingers.append(self.is_finger_up(landmarks, 4, 3))
        
        # Index finger
        fingers.append(self.is_finger_up(landmarks, 8, 6))
        
        # Middle finger  
        fingers.append(self.is_finger_up(landmarks, 12, 10))
        
        # Ring finger
        fingers.append(self.is_finger_up(landmarks, 16, 14))
        
        # Pinky
        fingers.append(self.is_finger_up(landmarks, 20, 18))
        
        return fingers
    
    def recognize_asl_letter(self, landmarks):
        """Recognize ASL letters based on hand landmarks"""
        if not landmarks:
            return "No hand detected"
        
        fingers = self.get_finger_states(landmarks)
        finger_count = sum(fingers)
        
        # Get some key landmark positions for additional checks
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        wrist = landmarks[0]
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        
        # Calculate some distances for gesture recognition
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
        
        # ASL Letter Recognition Logic
        
        # A - Closed fist with thumb on side
        if finger_count == 0 or (fingers == [True, False, False, False, False]):
            return "A"
        
        # B - All fingers up except thumb
        if fingers == [False, True, True, True, True]:
            return "B"
        
        # C - Curved hand shape
        if finger_count == 0 and thumb_index_dist > 0.1:
            return "C"
        
        # D - Index finger up, thumb touching middle finger
        if fingers == [False, True, False, False, False]:
            return "D"
        
        # E - All fingers down (closed fist)
        if fingers == [False, False, False, False, False]:
            return "E"
        
        # F - Index and middle down, others up, thumb touching index
        if fingers == [True, False, False, True, True]:
            return "F"
        
        # G - Index finger pointing horizontally
        if fingers == [True, True, False, False, False]:
            # Check if index is pointing sideways
            if abs(index_tip.y - index_mcp.y) < 0.05:
                return "G"
        
        # H - Index and middle fingers extended horizontally
        if fingers == [False, True, True, False, False]:
            return "H"
        
        # I - Pinky up only
        if fingers == [False, False, False, False, True]:
            return "I"
        
        # J - Pinky up, moving (we'll just detect pinky up for now)
        if fingers == [False, False, False, False, True]:
            return "I/J"
        
        # K - Index and middle up, thumb between them
        if fingers == [True, True, True, False, False]:
            return "K"
        
        # L - Thumb and index up, forming L shape
        if fingers == [True, True, False, False, False]:
            # Check if they form roughly 90 degree angle
            return "L"
        
        # M - Thumb under first three fingers
        if fingers == [False, False, False, False, False]:
            return "M"
        
        # N - Thumb under first two fingers  
        if fingers == [False, False, False, False, False]:
            return "N"
        
        # O - All fingers curved to form O
        if finger_count == 0 and thumb_index_dist < 0.08:
            return "O"
        
        # P - Index and middle down, others up
        if fingers == [True, False, False, True, True]:
            return "P"
        
        # Q - Thumb and index pointing down
        if fingers == [True, True, False, False, False]:
            if index_tip.y > index_mcp.y:  # Index pointing down
                return "Q"
        
        # R - Index and middle crossed
        if fingers == [False, True, True, False, False]:
            return "R"
        
        # S - Closed fist with thumb over fingers
        if fingers == [False, False, False, False, False]:
            return "S"
        
        # T - Thumb between index and middle
        if fingers == [True, False, False, False, False]:
            return "T"
        
        # U - Index and middle up together
        if fingers == [False, True, True, False, False]:
            return "U"
        
        # V - Index and middle up, separated
        if fingers == [False, True, True, False, False]:
            v_distance = self.calculate_distance(index_tip, middle_tip)
            if v_distance > 0.05:
                return "V"
            return "U"
        
        # W - Index, middle, ring up
        if fingers == [False, True, True, True, False]:
            return "W"
        
        # X - Index finger bent (hook shape)
        if fingers == [False, True, False, False, False]:
            return "X"
        
        # Y - Thumb and pinky up
        if fingers == [True, False, False, False, True]:
            return "Y"
        
        # Z - Index finger drawing Z (we'll just detect index up)
        if fingers == [False, True, False, False, False]:
            return "Z"
        
        # Numbers
        if finger_count == 1:
            return "1"
        elif finger_count == 2:
            return "2"
        elif finger_count == 3:
            return "3"
        elif finger_count == 4:
            return "4"
        elif finger_count == 5:
            return "5"
        
        return f"Unknown ({finger_count} fingers)"
    
    def run(self):
        """Main loop for MediaPipe ASL interpretation"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("MediaPipe ASL Interpreter Started!")
        print("Instructions:")
        print("- Show your hand clearly to the camera")
        print("- Try different ASL letters")
        print("- Press 'q' to quit")
        print("- Press 'r' to reset gesture buffer")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = self.hands.process(rgb_frame)
            
            detected_letter = "No hand detected"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    # Recognize the letter
                    letter = self.recognize_asl_letter(hand_landmarks.landmark)
                    
                    # Add to buffer for stability
                    self.gesture_buffer.append(letter)
                    
                    # Get most common letter from buffer
                    if len(self.gesture_buffer) >= 3:
                        letter_counts = {}
                        for l in self.gesture_buffer:
                            letter_counts[l] = letter_counts.get(l, 0) + 1
                        detected_letter = max(letter_counts, key=letter_counts.get)
                    else:
                        detected_letter = letter
            
            # Display the detected letter
            height, width = frame.shape[:2]
            
            # Main letter display
            text_size = cv2.getTextSize(detected_letter, cv2.FONT_HERSHEY_SIMPLEX, 3, 4)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 80
            
            # Background rectangle
            cv2.rectangle(frame, (text_x - 20, text_y - text_size[1] - 20), 
                         (text_x + text_size[0] + 20, text_y + 20), (0, 0, 0), -1)
            
            # Letter text
            cv2.putText(frame, detected_letter, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
            
            # Instructions
            cv2.putText(frame, "MediaPipe ASL Interpreter - 'q':quit 'r':reset", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show confidence indicator
            confidence = len(self.gesture_buffer) / 8.0
            cv2.rectangle(frame, (10, height - 30), (int(10 + confidence * 200), height - 10), 
                         (0, 255, 0), -1)
            cv2.putText(frame, "Confidence", (10, height - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('MediaPipe ASL Interpreter', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.gesture_buffer.clear()
                print("Gesture buffer reset")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    interpreter = MediaPipeASLInterpreter()
    interpreter.run()