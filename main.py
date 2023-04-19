import cv2
import mediapipe
from mediapipe.framework.formats import landmark_pb2

drawing_module = mediapipe.solutions.drawing_utils
hands_module = mediapipe.solutions.hands

capture = cv2.VideoCapture(0)

with hands_module.Hands(
    static_image_mode=False,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2,
    max_num_hands=2,
) as hands:
    while True:
        ret, frame = capture.read()
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks != None:
            for hand_landmarks in results.multi_hand_landmarks:
                drawing_module.draw_landmarks(
                    frame, hand_landmarks, hands_module.HAND_CONNECTIONS
                )
                print("index tip")
                print(hand_landmarks.landmark[hands_module.HandLandmark.INDEX_FINGER_TIP])
                print("thumb tip")
                print(hand_landmarks.landmark[hands_module.HandLandmark.THUMB_TIP])

        cv2.imshow("Test hand", frame)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
capture.release()
