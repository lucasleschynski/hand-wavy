import cv2
import mediapipe

class HandExtractor():
    def __init__(self, capture):
        self.drawing_module = mediapipe.solutions.drawing_utils
        self.hands_module = mediapipe.solutions.hands
        self.capture = capture
        self.gesture_coords = None
        self.landmark_frame = None
        pass

    def process_frame(self):
        with self.hands_module.Hands(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1,
        ) as hands:
            ret, frame = self.capture.read()
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks != None:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.drawing_module.draw_landmarks(
                        frame, hand_landmarks, self.hands_module.HAND_CONNECTIONS
                    )
                    # print("index tip")
                    # print(hand_landmarks.landmark[self.hands_module.HandLandmark.INDEX_FINGER_TIP])
                    index_tip = hand_landmarks.landmark[self.hands_module.HandLandmark.INDEX_FINGER_TIP]
                    # print("thumb tip")
                    # print(hand_landmarks.landmark[self.hands_module.HandLandmark.THUMB_TIP])
                    thumb_tip = hand_landmarks.landmark[self.hands_module.HandLandmark.THUMB_TIP]
                    self.gesture_coords = index_tip, thumb_tip
            self.landmark_frame = frame

    def get_landmarked_frame(self):
        return self.landmark_frame

    def get_gesture_coords(self):
        return self.gesture_coords
    
    def is_pinched():
        pass

