import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model from pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Mediapipe hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the hands module with a detection confidence threshold
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define label dictionary for predictions
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8:'I', 9:'HELLO'}

while True:
    # Initialize lists for storing x, y coordinates and data for model input
    data_aux = []
    x_ = []
    y_ = []

    # Capture a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Get the height and width of the frame
    H, W, _ = frame.shape

    # Convert the frame to RGB for Mediapipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(frame_rgb)

    # If hand landmarks are detected, proceed
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            # Reset x_ and y_ for each hand in the frame
            x_ = []
            y_ = []
            data_aux = []

            # Extract the x and y coordinates of each landmark
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize the coordinates (subtract min value)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalize x
                data_aux.append(y - min(y_))  # Normalize y

            # Ensure that we have exactly 42 features (21 landmarks with x and y for each)
            if len(data_aux) == 42:
                # Make prediction using the model
                prediction = model.predict([np.asarray(data_aux)])

                # Map the predicted label to a character
                predicted_character = labels_dict[int(prediction[0])]

                # Draw bounding box around the hand and display the predicted character
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

    # Show the frame with hand landmarks and prediction
    cv2.imshow('frame', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
