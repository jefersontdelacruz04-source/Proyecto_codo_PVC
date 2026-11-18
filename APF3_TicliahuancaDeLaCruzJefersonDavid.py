import cv2

# Conectar directamente a la app de DroidCam usando su IP y puerto
cap = cv2.VideoCapture("http://192.168.101.12:4747/video")

# Cargar el clasificador entrenado
codo_cascade = cv2.CascadeClassifier('classifier/cascade.xml')

while True:
    # Descartar cuadros acumulados para evitar congelamiento
    cap.grab()
    ret, frame = cap.retrieve()
    
    if not ret:
        print("Error al conectar con la cámara")
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    codos = codo_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.1, 
    minNeighbors=21,  # Súbelo para eliminar los cuadros falsos y unificar las detecciones
    minSize=(100, 100)
)
    for (x, y, w, h) in codos:
        # Añadimos un margen para abarcar todo el codo
        margin = 15  
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = x + w + margin
        y2 = y + h + margin
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, 'Codo PVC', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Detector de Codo PVC', frame)

    # Presiona la tecla ESC (27) para salir
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()