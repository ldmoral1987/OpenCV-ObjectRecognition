import cv2

# Este programa detecta peatones en un vídeo

# Vídeo de origen
IP_file = 'pedestrians.avi'

# Se carga el vídeo de origen
vid_file = cv2.VideoCapture(IP_file)

# Se cargan los clasificadores HAAR entrenados
pedestrian_classifier = 'pedestrian.xml'

# Se crean los tracker de los clasificadores
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier)

while True:
    # Empieza a leer el vídeo frame a frame
    (read_successful, frame) = vid_file.read()

    if read_successful:
        # Se convierte la imagen a escala de grises
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Se detectan los peatones
    pedestrians = pedestrian_tracker.detectMultiScale(gray_frame,1.1,9)

    # Se dibuja un cuadrado alrededor de los peatones
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Human', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Se muestra la imagen con las marcas
    cv2.imshow('Peatones detectados',frame)

    # Se captura una pulsación de tecla
    key = cv2.waitKey(1)

    # El programa finaliza si se pulsa ESC
    if key == 27:
        break

# Se libera el objeto capturador de vídeo
vid_file.release()
