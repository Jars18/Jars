import cv2
import mediapipe as mp
import numpy as np
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


mp_hands = mp.solutions.hands

color_mouse_pointer = (255, 0, 255)
#cap = cv2.VideoCapture("video_0001.mp4")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

def detect_word_si(hand_landmarks):
     word_si = False
     color_base = (255, 0, 112)
     color_h8 = (255, 198, 82)
     #comando para la palabra "si"
     x_h0 = int(hand_landmarks.landmark[0].x * width) #posicion en x de hand 0(muñeca)
     y_h0 = int(hand_landmarks.landmark[0].y * height) #posicion en y de hand 0(muñeca)
     x_h9 = int(hand_landmarks.landmark[9].x * width) #posicion en x de hand 9(base del dedo del medio)
     y_h9 = int(hand_landmarks.landmark[9].y * height) #posicion en y de hand 9(base del dedo del medio)
     x_h8 = int(hand_landmarks.landmark[8].x * width) #posicion en x de hand 8(punta del indice) 
     y_h8 = int(hand_landmarks.landmark[8].y * height) #posicion en y de hand 8(punta del indice)
     x_h12 = int(hand_landmarks.landmark[12].x * width) #posicion en x de hand 12(punta del dedo del medio)
     y_h12 = int(hand_landmarks.landmark[12].y * height) #posicion en y de hand 12(punta del dedo del medio)
     x_h16 = int(hand_landmarks.landmark[16].x * width) #posicion en x de hand 16(punta del anular)
     y_h16 = int(hand_landmarks.landmark[16].y * height) #posicion en y de hand 16(punta del anular)
     x_h20 = int(hand_landmarks.landmark[20].x * width) #posicion en x de hand 20(punta del meñique)
     y_h20 = int(hand_landmarks.landmark[20].y * height) #posicion en y de hand 20(punta del meñique)
     d_base = calculate_distance(x_h0, y_h0, x_h9, y_h9) #distancia de 0 a 9 (mano)
     d_base_h8 = calculate_distance(x_h0, y_h0, x_h8, y_h8) #distancia de 0 a 8 (mano)
     d_base_h12 = calculate_distance(x_h0, y_h0, x_h12, y_h12) #distancia de 0 a 12 (mano)
     d_base_h16 = calculate_distance(x_h0, y_h0, x_h16, y_h16) #distancia de 0 a 16 (mano)
     d_base_h20 = calculate_distance(x_h0, y_h0, x_h20, y_h20) #distancia de 0 a 20 (mano)
     if d_base_h8 < d_base and d_base_h12 < d_base and d_base_h16 < d_base and d_base_h20 < d_base:
        word_si = True
        color_base = (255, 0, 255)
        color_h8 = (255, 0, 255)
     cv2.circle(output, (x_h0, y_h0), 5, color_base, 2)
     cv2.circle(output, (x_h8, y_h8), 5, color_h8, 2)
     cv2.line(output, (x_h0, y_h0), (x_h9, y_h9), color_base, 3)
     cv2.line(output, (x_h0, y_h0), (x_h8, y_h8), color_h8, 3)
     return word_si

with mp_holistic.Holistic(
     static_image_mode=False,
     model_complexity=1) as holistic:
     while True:
          ret, frame = cap.read()
          if ret == False:
               break
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          results = holistic.process(frame_rgb)
           # rostro
          #mp_drawing.draw_landmarks(
           #    frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            #   mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
             #  mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2))
          
          # Mano izquieda (azul)
          mp_drawing.draw_landmarks(
               frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
               mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=1),
               mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))
          
          # Mano derecha (verde)
          mp_drawing.draw_landmarks(
               frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
               mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
               mp_drawing.DrawingSpec(color=(57, 143, 0), thickness=2))
          
          # Postura
          mp_drawing.draw_landmarks(
               frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
               mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2, circle_radius=1),
               mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
          frame = cv2.flip(frame, 1)

          cv2.imshow("Frame", frame)
          if cv2.waitKey(1) & 0xFF == 27:
               break

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
color_mouse_pointer = (255, 0, 255)
# Puntos de la pantalla-juego
SCREEN_GAME_X_INI = 150
SCREEN_GAME_Y_INI = 160
SCREEN_GAME_X_FIN = 150 + 780
SCREEN_GAME_Y_FIN = 160 + 450
aspect_ratio_screen = (SCREEN_GAME_X_FIN - SCREEN_GAME_X_INI) / (SCREEN_GAME_Y_FIN - SCREEN_GAME_Y_INI)
print("aspect_ratio_screen:", aspect_ratio_screen)
X_Y_INI = 100
def calculate_distance(x1, y1, x2, y2):
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    return np.linalg.norm(p1 - p2)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        # Dibujando un área proporcional a la del juego
        area_width = width - X_Y_INI * 2
        area_height = int(area_width / aspect_ratio_screen)
        aux_image = np.zeros(frame.shape, np.uint8)
        aux_image = cv2.rectangle(aux_image, (X_Y_INI, X_Y_INI), (X_Y_INI + area_width, X_Y_INI +area_height), (255, 0, 0), -1)
        output = cv2.addWeighted(frame, 1, aux_image, 0.7, 0)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[9].x * width)
                y = int(hand_landmarks.landmark[9].y * height)
                xm = np.interp(x, (X_Y_INI, X_Y_INI + area_width), (SCREEN_GAME_X_INI, SCREEN_GAME_X_FIN))
                ym = np.interp(y, (X_Y_INI, X_Y_INI + area_height), (SCREEN_GAME_Y_INI, SCREEN_GAME_Y_FIN))
                pyautogui.moveTo(int(xm), int(ym))
                if detect_word_si(hand_landmarks):
                    pyautogui.click()
                cv2.circle(output, (x, y), 10, color_mouse_pointer, 3)
                cv2.circle(output, (x, y), 5, color_mouse_pointer, -1)
        #cv2.imshow('Frame', frame)
        cv2.imshow('output', output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()