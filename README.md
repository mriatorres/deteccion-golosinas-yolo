# 🍫 Detección de Golosinas y Estimación Calórica usando YOLO y Transfer Learning

Este proyecto implementa un modelo de **detección de objetos** basado en **YOLOv8** para identificar diferentes tipos de golosinas y estimar sus **calorías aproximadas** a partir de imágenes o video en tiempo real.

---

## 🎯 Objetivo del proyecto

El objetivo principal es **entrenar y evaluar un modelo YOLO personalizado** capaz de:
- Detectar golosinas como **Capri, Ducales, Nucita y WaferJet**.
- Realizar **estimaciones calóricas automáticas** por clase detectada.
- Probar el modelo con imágenes, videos o webcam.

---

## 🧠 Modelo y Arquitectura

Se utilizó la arquitectura **YOLOv8-small (`yolo11s.pt`)**, con un enfoque de **Transfer Learning**, entrenando sobre un conjunto de datos personalizado.

- **Modelo base:** `yolo11s.pt`
- **Framework:** [Ultralytics YOLOv8](https://docs.ultralytics.com)
- **Entrenamiento:** 60 épocas (`epochs=60`)
- **Tamaño de imagen:** 640x640 píxeles (`imgsz=640`)

---

## 🗂️ Estructura del conjunto de datos

El dataset fue organizado siguiendo el formato estándar YOLO:

/content/data
│
├── train/
│ ├── images/
│ └── labels/
│
├── validation/
│ ├── images/
│ └── labels/
│
└── data.yaml



### Contenido de `data.yaml`

```yaml
path: /content/data
train: train/images
val: validation/images
nc: 4
names:
  - Capri
  - Ducales
  - Nucita
  - WaferJet

```

### ⚙️ Configuración del entorno

En Google Colab o tu entorno local, asegúrate de activar la GPU:

Colab:

Ve a Entorno de ejecución → Cambiar tipo de entorno de ejecución → Acelerador de hardware: GPU

Verifica que la GPU esté disponible:

!nvidia-smi


Instala YOLO (Ultralytics):

!pip install ultralytics

```yaml
!pip install ultralytics
```

Entrenamiento del modelo

Ejecuta el siguiente comando para entrenar:

!yolo detect train data=/content/data.yaml model=yolo11s.pt epochs=60 imgsz=640


Durante el entrenamiento se generan métricas de desempeño como:

Loss

Precision

Recall

mAP (mean Average Precision)

📊 Evaluación del modelo

Una vez entrenado, YOLO crea una carpeta runs/detect/train/ con resultados:

results.png: curva de entrenamiento

confusion_matrix.png: matriz de confusión

PR_curve.png: curva precisión-recall

weights/best.pt: pesos finales del modelo

Estas gráficas muestran el comportamiento del modelo en términos de precisión, recall, pérdida y mAP, que sirven como indicadores del rendimiento alcanzado.

🔍 Prueba y predicción

Para evaluar el modelo sobre imágenes de validación:

!yolo detect predict model=runs/detect/train/weights/best.pt source=data/validation/images save=True


Visualiza los resultados:

import glob
from IPython.display import Image, display

for image_path in glob.glob(f'/content/runs/detect/predict/*.jpg')[:10]:
  display(Image(filename=image_path, height=400))
  print('\n')

💾 Guardar y descargar el modelo entrenado
!mkdir /content/my_model
!cp /content/runs/detect/train/weights/best.pt /content/my_model/my_model.pt
!cp -r /content/runs/detect/train /content/my_model

%cd my_model
!zip /content/my_model.zip my_model.pt
!zip -r /content/my_model.zip train
%cd /content

from google.colab import files
files.download('/content/my_model.zip')

🎥 Inferencia con cámara web (local)

En tu entorno local (Windows):

Activa tu entorno:

conda activate yolo-env1


Descarga el script de detección:

curl -o yolo_detect.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/yolo_detect.py


Ejecuta el modelo con tu cámara:

python yolo_detect.py --model my_model.pt --source 0 --resolution 1280x720


Para detener la ejecución, presiona Ctrl + C en la terminal.

📈 Tabla de resultados (Métricas del modelo)
Métrica	Descripción	Valor aproximado
Loss (Train/Val)	Error durante entrenamiento	↓ Estable y decreciente
Precision	Exactitud de detección	↑ Alta (ej. 0.90)
Recall	Capacidad de encontrar todos los objetos	↑ Alta (ej. 0.88)
mAP@0.5	Precisión promedio general	↑ Muy buena (ej. 0.91)

Incluye en tu presentación los gráficos generados automáticamente:

results.png (curvas de loss, precision, recall, mAP)

PR_curve.png (curva precisión-recall)

confusion_matrix.png (errores de clasificación)

🧩 Conclusiones

El modelo YOLO logró una alta precisión en la detección de golosinas, diferenciando correctamente las cuatro clases.

El transfer learning permitió obtener buenos resultados con pocas épocas.

Las métricas y curvas de entrenamiento muestran un buen equilibrio entre precisión y recall.

Este modelo puede ser extendido para sistemas de control calórico automático o aplicaciones móviles de reconocimiento alimentario.

👩‍💻 Autora del proyecto

Nombre: Marleny
Proyecto: Detección de Golosinas y Estimación Calórica usando YOLO y Transfer Learning
Fecha: Noviembre 2025
Entorno: Google Colab + Python (Ultralytics YOLO)


---

¿Quieres que te genere una **versión con enlaces e imágenes (por ejemplo, los gráficos y capturas de resultado
