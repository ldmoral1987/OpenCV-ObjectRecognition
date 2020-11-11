import cv2

# Este programa detecta peatones y coches en una imagen

# Imagen de origen
img_file = "image.png"

# Se carga la imagen
img = cv2.imread(img_file)

# Se cargan los clasificadores HAAR entrenados
car_classifier = 'cars.xml'
pedestrian_classifier = 'pedestrian.xml'

# Se convierte la imagen a escala de grises
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Se crean los tracker de los clasificadores
car_tracker = cv2.CascadeClassifier(car_classifier)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier)

# Se detectan los coches y los peatones
cars = car_tracker.detectMultiScale(gray_img)
pedestrian = pedestrian_tracker.detectMultiScale(gray_img)

# Se dibuja un rectángulo alrededor de los coches
for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(img, 'Coche', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Se dibuja un rectángulo alrededor de los peatones
for (x,y,w,h) in pedestrian:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
    cv2.putText(img, 'Peaton', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Se muestra la imagen con las marcas
cv2.imshow('Deteccion de peatones y coches',img)

# Se espera una pulsación de teclado para salir
cv2.waitKey()


print("I'm done")