import cv2
import numpy as np
import mss
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple, List, Optional
import os
import re


class CartaDetector:
    def __init__(self):
        # Rutas de datos
        self.BASE_PATH = r'Your path to folder where images are'
        self.TRAIN_PATH = os.path.join(self.BASE_PATH, 'train')
        self.TEST_PATH = os.path.join(self.BASE_PATH, 'test')
        self.VALID_PATH = os.path.join(self.BASE_PATH, 'valid')

        # Mapeo de nombres de carpetas a valores de cartas
        self.nombre_a_valor = {
            'ace': 'As',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'ten': '10',
            'jack': 'J',
            'queen': 'Q',
            'king': 'K'
        }

        # Crear el mapeo inverso para las predicciones
        self.clases = self.obtener_clases()
        self.valores_cartas = [self.extraer_valor(clase) for clase in self.clases]

        # Configuración de parámetros de detección
        self.MIN_AREA = 6000
        self.MIN_DIMENSION = 30
        self.MAX_RATIO = 2.0

        # Crear y entrenar el modelo si no existe, o cargar el existente
        self.modelo = self.cargar_o_crear_modelo()

    def extraer_valor(self, nombre_carpeta: str) -> str:
        """Extrae el valor de la carta del nombre de la carpeta."""
        nombre_lower = nombre_carpeta.lower()
        for key in self.nombre_a_valor:
            if key in nombre_lower:
                return self.nombre_a_valor[key]
        return nombre_carpeta  # Si no encuentra coincidencia, devuelve el nombre original

    def obtener_clases(self) -> List[str]:
        """Obtiene la lista de clases desde el directorio de entrenamiento."""
        return sorted(os.listdir(self.TRAIN_PATH))

    def cargar_o_crear_modelo(self):
        """Carga el modelo existente o crea uno nuevo si no existe."""
        modelo_path = os.path.join(self.BASE_PATH, 'modelo_cartas.h5')

        if os.path.exists(modelo_path):
            print("Cargando modelo existente...")
            return tf.keras.models.load_model(modelo_path)
        else:
            print("Creando y entrenando nuevo modelo...")
            return self.crear_y_entrenar_modelo(modelo_path)

    def crear_y_entrenar_modelo(self, modelo_path):
        """Crea y entrena un nuevo modelo."""
        # Configurar el generador de datos con aumentación
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest',
            validation_split=0.2
        )

        # Crear generadores de datos
        train_generator = datagen.flow_from_directory(
            self.TRAIN_PATH,
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical',
            color_mode='rgb',
            shuffle=True
        )

        valid_generator = datagen.flow_from_directory(
            self.VALID_PATH,
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical',
            color_mode='rgb',
            shuffle=True
        )

        print(f"Clases encontradas: {self.clases}")
        print(f"Valores mapeados: {self.valores_cartas}")

        # Crear modelo
        modelo = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.clases), activation='softmax')
        ])

        # Compilar modelo
        modelo.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Entrenar modelo
        modelo.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=15,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )

        # Guardar modelo
        modelo.save(modelo_path)
        return modelo

    def preprocesar_imagen(self, frame: np.ndarray) -> np.ndarray:
        """Preprocesa la imagen para mejorar la detección."""
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, (5, 5), 0)
        umbral = cv2.adaptiveThreshold(
            gris,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            4
        )
        return umbral

    def detectar_contornos(self, umbral: np.ndarray) -> List:
        """Encuentra los contornos en la imagen umbralizada."""
        contornos, _ = cv2.findContours(
            umbral,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contornos

    def filtrar_contorno(self, contorno: np.ndarray) -> Tuple[bool, Optional[Tuple]]:
        """Filtra los contornos según criterios específicos."""
        area = cv2.contourArea(contorno)
        if area < self.MIN_AREA:
            return False, None

        x, y, w, h = cv2.boundingRect(contorno)
        if w < self.MIN_DIMENSION or h < self.MIN_DIMENSION:
            return False, None

        ratio = max(w / h, h / w)
        if ratio > self.MAX_RATIO:
            return False, None

        return True, (x, y, w, h)

    def analizar_region_carta(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> str:
        """Analiza la región de la carta para identificar su valor."""
        # Extraer y preprocesar la región de la carta
        carta = frame[y:y + h, x:x + w]
        carta = cv2.resize(carta, (64, 64))
        carta = carta.astype("float32") / 255.0
        carta = np.expand_dims(carta, axis=0)

        # Realizar predicción
        prediccion = self.modelo.predict(carta, verbose=0)
        clase_predicha = np.argmax(prediccion[0])
        confianza = prediccion[0][clase_predicha]

        # Retornar valor solo si la confianza es suficiente
        if confianza > 0.6:  # Bajamos un poco el umbral ya que tenemos más clases
            return self.valores_cartas[clase_predicha]
        return "?"

    def procesar_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Procesa un frame completo y retorna el frame marcado y las cartas detectadas."""
        umbral = self.preprocesar_imagen(frame)
        contornos = self.detectar_contornos(umbral)

        cartas_detectadas = []
        frame_debug = frame.copy()

        for contorno in contornos:
            es_valido, dims = self.filtrar_contorno(contorno)
            if not es_valido:
                continue

            x, y, w, h = dims
            cv2.rectangle(frame_debug, (x, y), (x + w, y + h), (0, 255, 0), 3)

            valor_carta = self.analizar_region_carta(frame, x, y, w, h)
            if valor_carta != "?":
                cartas_detectadas.append(valor_carta)
                cv2.putText(frame_debug, valor_carta, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame_debug, cartas_detectadas


def main():
    detector = CartaDetector()

    with mss.mss() as sct:
        monitor = sct.monitors[1]

        try:
            while True:
                frame = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                frame_procesado, cartas = detector.procesar_frame(frame)

                if cartas:
                    print("Cartas detectadas:", cartas)

                cv2.imshow("Detección de Cartas", frame_procesado)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Programa terminado por el usuario")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()