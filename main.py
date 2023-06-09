import cv2
import mediapipe
import processing_py as pr
from hand_extractor import HandExtractor


def main():
    capture = cv2.VideoCapture(0)
    extractor = HandExtractor(capture)

    # Main loop (video feed -> gesture detection -> drawing)
    while True:
        # updates gesture coordinates and landmarked frame (optional)
        extractor.process_frame()
        landmarked_frame = extractor.get_landmarked_frame()
        cv2.imshow("Image Feed", landmarked_frame)

        # get index/thumb tip coords
        pinch_coord = extractor.get_gesture_coords()

        if pinch_coord:
            # Do calcuations
            continue

        if cv2.waitKey(1) == 'q':
            cv2.destroyAllWindows()
            capture.release()
            break

if __name__=="__main__":
    main()
