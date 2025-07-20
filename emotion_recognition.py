from facial_emotion_recognition import EmotionRecognition
import cv2

# Initialize the emotion recognition model (using CPU)
er = EmotionRecognition(device='cpu')
cam = cv2.VideoCapture(0)

while True:
    success, frame = cam.read()  # Capture frame-by-frame
    if not success:
        break

    # Perform emotion recognition
    frame = er.recognise_emotion(frame, return_type='BGR')

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' key to exit
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
