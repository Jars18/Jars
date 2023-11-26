import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


mp_hands = mp.solutions.hands

color_mouse_pointer = (255, 0, 255)
#cap = cv2.VideoCapture("video_0001.mp4")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
#puntos en x,y de todos los puntos de la mano





#funciones para cada palabra
def detect_word_si(hand_landmarks):
     word_si = False
     color_base = (255, 0, 112)
     color_h8 = (255, 198, 82)
     #comando para la palabra "si"
     x_h0 = int(hand_landmarks.landmark[0].x * width) #posicion en x de hand 0(muñeca)
     y_h0 = int(hand_landmarks.landmark[0].y * height) #posicion en y de hand 0(muñeca)
     x_h1 = int(hand_landmarks.landmark[1].x * width) #posicion en x de hand 1(muñeca)
     y_h1 = int(hand_landmarks.landmark[1].y * height) #posicion en y de hand 1(muñeca)
     x_h2 = int(hand_landmarks.landmark[2].x * width) #posicion en x de hand 2(muñeca)
     y_h2 = int(hand_landmarks.landmark[2].y * height) #posicion en y de hand 2(muñeca)
     x_h3 = int(hand_landmarks.landmark[3].x * width) #posicion en x de hand 3(muñeca)
     y_h3 = int(hand_landmarks.landmark[3].y * height) #posicion en y de hand 3(muñeca)
     x_h4 = int(hand_landmarks.landmark[4].x * width) #posicion en x de hand 4
     y_h4 = int(hand_landmarks.landmark[4].y * height) #posicion en y de hand 4
     x_h5 = int(hand_landmarks.landmark[5].x * width) #posicion en x de hand 5
     y_h5 = int(hand_landmarks.landmark[5].y * height) #posicion en y de hand 5
     x_h6 = int(hand_landmarks.landmark[6].x * width) #posicion en x de hand 6
     y_h6 = int(hand_landmarks.landmark[6].y * height) #posicion en y de hand 6
     x_h7 = int(hand_landmarks.landmark[7].x * width) #posicion en x de hand 7
     y_h7 = int(hand_landmarks.landmark[7].y * height) #posicion en y de hand 7
     x_h8 = int(hand_landmarks.landmark[8].x * width) #posicion en x de hand 8(punta del indice) 
     y_h8 = int(hand_landmarks.landmark[8].y * height) #posicion en y de hand 8(punta del indice)
     x_h9 = int(hand_landmarks.landmark[9].x * width) #posicion en x de hand 9(punta del indice) 
     y_h9 = int(hand_landmarks.landmark[9].y * height) #posicion en y de hand 9(punta del indice)
     x_h10 = int(hand_landmarks.landmark[10].x * width) #posicion en x de hand 10(punta del indice) 
     y_h10 = int(hand_landmarks.landmark[10].y * height) #posicion en y de hand 10(punta del indice)
     x_h11 = int(hand_landmarks.landmark[11].x * width) #posicion en x de hand 11(punta del indice) 
     y_h11 = int(hand_landmarks.landmark[11].y * height) #posicion en y de hand 11(punta del indice)
     x_h12 = int(hand_landmarks.landmark[12].x * width) #posicion en x de hand 12(punta del dedo del medio)
     y_h12 = int(hand_landmarks.landmark[12].y * height) #posicion en y de hand 12(punta del dedo del medio)
     x_h13 = int(hand_landmarks.landmark[13].x * width) #posicion en x de hand 13(punta del dedo del medio)
     y_h13 = int(hand_landmarks.landmark[13].y * height) #posicion en y de hand 13(punta del dedo del medio)
     x_h14 = int(hand_landmarks.landmark[14].x * width) #posicion en x de hand 14(punta del dedo del medio)
     y_h14 = int(hand_landmarks.landmark[14].y * height) #posicion en y de hand 14(punta del dedo del medio)
     x_h15 = int(hand_landmarks.landmark[15].x * width) #posicion en x de hand 15(punta del dedo del medio)
     y_h15 = int(hand_landmarks.landmark[15].y * height) #posicion en y de hand 15(punta del dedo del medio)
     x_h16 = int(hand_landmarks.landmark[16].x * width) #posicion en x de hand 16(punta del dedo del medio)
     y_h16 = int(hand_landmarks.landmark[16].y * height) #posicion en y de hand 16(punta del dedo del medio)
     x_h17 = int(hand_landmarks.landmark[17].x * width) #posicion en x de hand 17(punta del dedo del medio)
     y_h17 = int(hand_landmarks.landmark[17].y * height) #posicion en y de hand 17(punta del dedo del medio)
     x_h18 = int(hand_landmarks.landmark[18].x * width) #posicion en x de hand 18(punta del dedo del medio)
     y_h18 = int(hand_landmarks.landmark[18].y * height) #posicion en y de hand 18(punta del dedo del medio)
     x_h19 = int(hand_landmarks.landmark[19].x * width) #posicion en x de hand 19(punta del dedo del medio)
     y_h19 = int(hand_landmarks.landmark[19].y * height) #posicion en y de hand 19(punta del dedo del medio)
     x_h20 = int(hand_landmarks.landmark[20].x * width) #posicion en x de hand 20(punta del dedo del medio)
     y_h20 = int(hand_landmarks.landmark[20].y * height) #posicion en y de hand 20(punta del dedo del medio)
     #distancias de la muñeca a puntos de la mano
     d_base_h4 = calculate_distance(x_h0, y_h0, x_h4, y_h4) #distancia de 0 a 4 (mano)
     d_base_h5 = calculate_distance(x_h0, y_h0, x_h5, y_h5) #distancia de 0 a 5 (mano)
     d_base_h6 = calculate_distance(x_h0, y_h0, x_h6, y_h6) #distancia de 0 a 6 (mano)
     d_base_h8 = calculate_distance(x_h0, y_h0, x_h8, y_h8) #distancia de 0 a 8 (mano)
     d_base_h9 = calculate_distance(x_h0, y_h0, x_h9, y_h9) #distancia de 0 a 9 (mano)
     d_base_h10 = calculate_distance(x_h0, y_h0, x_h10, y_h10) #distancia de 0 a 9 (mano)
     d_base_h12 = calculate_distance(x_h0, y_h0, x_h12, y_h12) #distancia de 0 a 12 (mano)
     d_base_h16 = calculate_distance(x_h0, y_h0, x_h16, y_h16) #distancia de 0 a 16 (mano)
     d_base_h20 = calculate_distance(x_h0, y_h0, x_h20, y_h20) #distancia de 0 a 20 (mano)
     if d_base_h8 < d_base_h9 and d_base_h12 < d_base_h9 and d_base_h16 < d_base_h9 and d_base_h20 < d_base_h9:
        word_si = True
        color_base = (255, 0, 255)
        color_h8 = (255, 0, 255)
     cv2.circle(output, (x_h0, y_h0), 5, color_base, 2)
     cv2.circle(output, (x_h8, y_h8), 5, color_h8, 2)
     cv2.line(output, (x_h0, y_h0), (x_h9, y_h9), color_base, 3)
     cv2.line(output, (x_h0, y_h0), (x_h8, y_h8), color_h8, 3)
     return word_si

def detect_word_no(hand_landmarks):
     word_no = False
     color_base = (255, 0, 112)
     color_h8 = (255, 198, 82)
     #comando para la palabra "no"
     x_h0 = int(hand_landmarks.landmark[0].x * width) #posicion en x de hand 0(muñeca)
     y_h0 = int(hand_landmarks.landmark[0].y * height) #posicion en y de hand 0(muñeca)
     x_h1 = int(hand_landmarks.landmark[1].x * width) #posicion en x de hand 1(muñeca)
     y_h1 = int(hand_landmarks.landmark[1].y * height) #posicion en y de hand 1(muñeca)
     x_h2 = int(hand_landmarks.landmark[2].x * width) #posicion en x de hand 2(muñeca)
     y_h2 = int(hand_landmarks.landmark[2].y * height) #posicion en y de hand 2(muñeca)
     x_h3 = int(hand_landmarks.landmark[3].x * width) #posicion en x de hand 3(muñeca)
     y_h3 = int(hand_landmarks.landmark[3].y * height) #posicion en y de hand 3(muñeca)
     x_h4 = int(hand_landmarks.landmark[4].x * width) #posicion en x de hand 4
     y_h4 = int(hand_landmarks.landmark[4].y * height) #posicion en y de hand 4
     x_h5 = int(hand_landmarks.landmark[5].x * width) #posicion en x de hand 5
     y_h5 = int(hand_landmarks.landmark[5].y * height) #posicion en y de hand 5
     x_h6 = int(hand_landmarks.landmark[6].x * width) #posicion en x de hand 6
     y_h6 = int(hand_landmarks.landmark[6].y * height) #posicion en y de hand 6
     x_h7 = int(hand_landmarks.landmark[7].x * width) #posicion en x de hand 7
     y_h7 = int(hand_landmarks.landmark[7].y * height) #posicion en y de hand 7
     x_h8 = int(hand_landmarks.landmark[8].x * width) #posicion en x de hand 8(punta del indice) 
     y_h8 = int(hand_landmarks.landmark[8].y * height) #posicion en y de hand 8(punta del indice)
     x_h9 = int(hand_landmarks.landmark[9].x * width) #posicion en x de hand 9(punta del indice) 
     y_h9 = int(hand_landmarks.landmark[9].y * height) #posicion en y de hand 9(punta del indice)
     x_h10 = int(hand_landmarks.landmark[10].x * width) #posicion en x de hand 10(punta del indice) 
     y_h10 = int(hand_landmarks.landmark[10].y * height) #posicion en y de hand 10(punta del indice)
     x_h11 = int(hand_landmarks.landmark[11].x * width) #posicion en x de hand 11(punta del indice) 
     y_h11 = int(hand_landmarks.landmark[11].y * height) #posicion en y de hand 11(punta del indice)
     x_h12 = int(hand_landmarks.landmark[12].x * width) #posicion en x de hand 12(punta del dedo del medio)
     y_h12 = int(hand_landmarks.landmark[12].y * height) #posicion en y de hand 12(punta del dedo del medio)
     x_h13 = int(hand_landmarks.landmark[13].x * width) #posicion en x de hand 13(punta del dedo del medio)
     y_h13 = int(hand_landmarks.landmark[13].y * height) #posicion en y de hand 13(punta del dedo del medio)
     x_h14 = int(hand_landmarks.landmark[14].x * width) #posicion en x de hand 14(punta del dedo del medio)
     y_h14 = int(hand_landmarks.landmark[14].y * height) #posicion en y de hand 14(punta del dedo del medio)
     x_h15 = int(hand_landmarks.landmark[15].x * width) #posicion en x de hand 15(punta del dedo del medio)
     y_h15 = int(hand_landmarks.landmark[15].y * height) #posicion en y de hand 15(punta del dedo del medio)
     x_h16 = int(hand_landmarks.landmark[16].x * width) #posicion en x de hand 16(punta del dedo del medio)
     y_h16 = int(hand_landmarks.landmark[16].y * height) #posicion en y de hand 16(punta del dedo del medio)
     x_h17 = int(hand_landmarks.landmark[17].x * width) #posicion en x de hand 17(punta del dedo del medio)
     y_h17 = int(hand_landmarks.landmark[17].y * height) #posicion en y de hand 17(punta del dedo del medio)
     x_h18 = int(hand_landmarks.landmark[18].x * width) #posicion en x de hand 18(punta del dedo del medio)
     y_h18 = int(hand_landmarks.landmark[18].y * height) #posicion en y de hand 18(punta del dedo del medio)
     x_h19 = int(hand_landmarks.landmark[19].x * width) #posicion en x de hand 19(punta del dedo del medio)
     y_h19 = int(hand_landmarks.landmark[19].y * height) #posicion en y de hand 19(punta del dedo del medio)
     x_h20 = int(hand_landmarks.landmark[20].x * width) #posicion en x de hand 20(punta del dedo del medio)
     y_h20 = int(hand_landmarks.landmark[20].y * height) #posicion en y de hand 20(punta del dedo del medio)
     #distancias de la muñeca a puntos de la mano
     d_base_h4 = calculate_distance(x_h0, y_h0, x_h4, y_h4) #distancia de 0 a 4 (mano)
     d_base_h5 = calculate_distance(x_h0, y_h0, x_h5, y_h5) #distancia de 0 a 5 (mano)
     d_base_h6 = calculate_distance(x_h0, y_h0, x_h6, y_h6) #distancia de 0 a 6 (mano)
     d_base_h8 = calculate_distance(x_h0, y_h0, x_h8, y_h8) #distancia de 0 a 8 (mano)
     d_base_h9 = calculate_distance(x_h0, y_h0, x_h9, y_h9) #distancia de 0 a 9 (mano)
     d_base_h10 = calculate_distance(x_h0, y_h0, x_h10, y_h10) #distancia de 0 a 9 (mano)
     d_base_h12 = calculate_distance(x_h0, y_h0, x_h12, y_h12) #distancia de 0 a 12 (mano)
     d_base_h16 = calculate_distance(x_h0, y_h0, x_h16, y_h16) #distancia de 0 a 16 (mano)
     d_base_h20 = calculate_distance(x_h0, y_h0, x_h20, y_h20) #distancia de 0 a 20 (mano)
     if d_base_h5 < d_base_h4 and d_base_h8 < d_base_h6 and d_base_h12 > d_base_h9:
        word_no = True
        color_base = (255, 0, 255)
        color_h8 = (255, 0, 255)
     cv2.circle(output, (x_h0, y_h0), 5, color_base, 2)
     cv2.circle(output, (x_h8, y_h8), 5, color_h8, 2)
     cv2.circle(output, (x_h4, y_h4), 5, color_h8, 2)
     cv2.circle(output, (x_h8, y_h6), 5, color_h8, 2)
     cv2.circle(output, (x_h8, y_h5), 5, color_h8, 2)
     cv2.line(output, (x_h0, y_h0), (x_h5, y_h5), color_base, 3)
     cv2.line(output, (x_h0, y_h0), (x_h8, y_h8), color_h8, 3)
     cv2.line(output, (x_h0, y_h0), (x_h5, y_h5), color_base, 3)
     cv2.line(output, (x_h0, y_h0), (x_h8, y_h8), color_h8, 3)
     return word_no

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
                    time.sleep(0.05)
                    pyautogui.click()
                    cv2.circle(output, (x, y), 10, color_mouse_pointer, 3)
                    cv2.circle(output, (x, y), 5, color_mouse_pointer, -1)
                
                elif detect_word_no(hand_landmarks):
                    time.sleep(0.05)
                    pyautogui.click()
                    cv2.circle(output, (x, y), 10, color_mouse_pointer, 3)
                    cv2.circle(output, (x, y), 5, color_mouse_pointer, -1)
        #cv2.imshow('Frame', frame)
        cv2.imshow('output', output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()