#########
##actual code
# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle

# def main():
#     st.set_page_config(page_title="Sign Language Recognition", page_icon=":diamond:")
#     st.title("Sign Language Recognition")

#     # Load the Random Forest Classifier model
#     model_dict = pickle.load(open('./model.p', 'rb'))
#     model = model_dict['model']
    
#     mp_hands = mp.solutions.hands
#     mp_drawing = mp.solutions.drawing_utils
#     mp_drawing_styles = mp.solutions.drawing_styles

#     hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

#     # labels_dict = {0:'0',1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z', 36:'hello', 37:'good morning', 38:'good afternoon', 39:'good evening', 40:'good night', 41:'how are you?', 42:'good day', 43:'happy birthday', 44:'happy anniversary', 45:'red', 46:'blue', 47:'green', 48:'yellow', 49:'white', 50:'purple',51:'black', 52:'pink', 53:'brown', 54:'gold', 55:'orange', 56:'please', 57:'thank you', 58:'you are welcome', 59:'excuse me'}
#     labels_dict ={0:'1',1:'2',2:'3',3:'4',4:'Good Morning',5:'Good Evening',6:'Good Day'}
#     cap = cv2.VideoCapture(0)

#     stframe = st.empty()

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             st.error("Error: Failed to capture frame from the camera.")
#             break

#         H, W, _ = frame.shape

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         results = hands.process(frame_rgb)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     frame,  # image to draw
#                     hand_landmarks,  # model output
#                     mp_hands.HAND_CONNECTIONS,  # hand connections
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style())

#             for hand_landmarks in results.multi_hand_landmarks:
#                 data_aux = []
#                 x_ = []
#                 y_ = []
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y

#                     x_.append(x)
#                     y_.append(y)

#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x - min(x_))
#                     data_aux.append(y - min(y_))

#                 # Pad the feature vector to have 100 features
#                 while len(data_aux) < 100:
#                     data_aux.append(0.0)

#                 x1 = int(min(x_) * W) - 10
#                 y1 = int(min(y_) * H) - 10

#                 x2 = int(max(x_) * W) - 10
#                 y2 = int(max(y_) * H) - 10

#                 prediction = model.predict([np.asarray(data_aux)])

#                 predicted_character = labels_dict[int(prediction[0])]

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#                 cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#                             cv2.LINE_AA)

#         stframe.image(frame, channels="BGR")

#     cap.release()


# if __name__ == "__main__":
#     main()


import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle

def main():
    st.set_page_config(page_title="Sign Language Recognition", page_icon=":diamond:")
    st.title("Sign Language Recognition")

    # Load the Random Forest Classifier model
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    labels_dict = {0: 'வணக்கம்', 1: 'வாருங்கள்', 2: 'ஒன்று', 3: 'நன்றி'}

    cap = cv2.VideoCapture(0)

    stframe = st.empty()

    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Failed to capture frame from the camera.")
            break

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        if len(results.multi_hand_landmarks) == 2:
            data_aux = []
            x_ = []
            y_ = []
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Pad the feature vector to have 100 features
            while len(data_aux) < 200:
                data_aux.append(0.0)

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            st.write(f"Predicted Sign: {predicted_character}")

        stframe.image(frame, channels="BGR")

    cap.release()

if __name__ == "__main__":
    main()
















# import os
# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# from PIL import ImageFont, ImageDraw, Image


# # Load the Random Forest Classifier model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# cap = cv2.VideoCapture(0)
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # Tamil labels
# # labels_dict = {0:'அ',1:'ஆ',2:'வணக்கம்',3:'மீண்டும் சந்திப்போம்',4:'படிப்பு',5:'நான்',6:'நீ'}
# labels_dict ={0:'1',1:'2',2:'3',3:'4',4:'Good Morning',5:'Good Evening',6:'Good Day'}

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture frame from the camera.")
#         break

#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())

#         for hand_landmarks in results.multi_hand_landmarks:
#             data_aux = []
#             x_ = []
#             y_ = []
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 x_.append(x)
#                 y_.append(y)
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))

#             # Pad the feature vector to have 100 features
#             while len(data_aux) < 100:
#                 data_aux.append(0.0)

#             x1 = int(min(x_) * W) - 10
#             y1 = int(min(y_) * H) - 10
#             x2 = int(max(x_) * W) - 10
#             y2 = int(max(y_) * H) - 10

#             prediction = model.predict([np.asarray(data_aux)])
#             predicted_character = labels_dict[int(prediction[0])]

#             # Display Tamil label with the specified font
#             font_path = 'D:\\sign-language-detector-python-master\\sign-language-detector-python-master\\catamaran\\Catamaran-Bold.ttf'  # Replace with the actual path
#             font_size = 30
#             font = ImageFont.truetype(font_path, font_size)
#             img_pil = Image.fromarray(frame)
#             draw = ImageDraw.Draw(img_pil)
#             draw.text((x1, y1 - 10), predicted_character, font=font, fill=(0, 0, 0))
#             frame = np.array(img_pil)

#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()