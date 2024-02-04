import cv2
import mediapipe as mp
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    if detection_result.multi_hand_landmarks is None:
        return rgb_image

    hand_landmarks_list = detection_result.multi_hand_landmarks
    handedness_list = detection_result.multi_handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        if (handedness.classification[0].index == 0):
            # Draw left hand connections
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
        else:
            # Draw right hand connections
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
        y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN
        
        cv2.rectangle(annotated_image,
                      (int(min(x_coordinates) * width), int(min(y_coordinates) * height)), 
                      (int(max(x_coordinates) * width), int(max(y_coordinates) * height)),
                      color=(0, 255, 0),
                      thickness=1)

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness.classification[0].label}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

if __name__ == '__main__':
    mp_hands = mp.solutions.hands # Hands model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()
            
            # Flip
            frame = cv2.flip(frame, 1)

            # Make detections
            image, results = mediapipe_detection(frame, hands)
            
            # Draw landmarks
            drawed_image = draw_landmarks_on_image(image, results)

            # Show to screen
            cv2.imshow('OpenCV Feed', drawed_image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
