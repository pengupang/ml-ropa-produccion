# Clasificador de ropa: del modelo a producción

Proyecto de la **Evidencia N°4** de Machine Learning (TIEL26, Unidad 4) — INACAP Osorno.

Tomamos la red convolucional (CNN) que entrenamos en la EVA3 para clasificar fotos de ropa en 6 clases y la llevamos a producción: la medimos, la optimizamos, la exportamos a ONNX, le pusimos una puerta de calidad automática y la desplegamos como una API.

## Estructura del repositorio

```
ml-ropa-produccion/
├── .github/workflows/calidad.yml   # GitHub Actions: corre la puerta de calidad
├── modelos/
│   ├── modelo_p65_int8.onnx        # modelo final (poda 65 % + int8)
│   └── modelo_sin_entrenar.onnx    # solo para probar que la puerta rechaza modelos malos
├── datos/test_data.npz             # 198 imágenes de prueba (uint8) y sus etiquetas
├── tests/test_puerta_calidad.py    # pruebas con pytest
├── app.py                          # API con FastAPI
└── requirements.txt
```

## El modelo

| | |
|---|---|
| Tarea | Clasificación de imágenes de ropa, 6 clases |
| Entrada | Imagen RGB de 128 × 128, valores entre 0 y 1 |
| Arquitectura | 3 bloques Conv2D + MaxPooling, promedio global, Dense(64), Dense(6) con softmax |
| Optimización | Poda por magnitud del 65 % + cuantificación dinámica a 8 bits |
| Formato | ONNX (opset 17), ejecutado con ONNX Runtime |

Las capas de aumento de datos (flip, rotación, zoom, brillo) se quitaron del modelo de inferencia, ya que solo se usan durante el entrenamiento.

## Presupuesto de mejora

Calculado a partir de la línea base medida en la clase 1 (exactitud 65,2 %, pesos de 0,41 MB):

| Medida | Regla | Umbral |
|---|---|---|
| Exactitud en test | Perder como máximo 1 punto | ≥ 64,2 % |
| Tamaño | Pesar la mitad o menos | ≤ 0,20 MB |
| Latencia (lote de 1) | No ser más lento que la línea base | ≤ 12,96 ms |

## Resultados

| Variante | Exactitud | Tamaño gzip | ¿Cumple? |
|---|---|---|---|
| Línea base (Keras) | 65,2 % | 0,381 MB | — |
| Poda 50 % | 70,7 % | 0,227 MB | ❌ tamaño |
| Poda 65 % | 70,7 % | 0,172 MB | ✅ |
| Poda 80 % | 53,5 % | 0,114 MB | ❌ exactitud |
| Base int8 (TFLite) | 66,2 % | 0,097 MB | ✅ |
| **Poda 65 % + int8 (TFLite)** | **70,7 %** | **0,057 MB** | ✅ |
| **Poda 65 % + int8 (ONNX, final)** | **71,2 %** | **0,046 MB** (archivo: 0,120 MB) | ✅ |

El aumento de exactitud tras la poda se debe a las 3 épocas extra de reentrenamiento, no a la poda en sí. La diferencia de 0,5 puntos entre la versión TFLite y la ONNX corresponde a una sola imagen de 198.

**Paridad ONNX:** la versión fp32 es idéntica al modelo de Keras (diferencia máxima < 0,000001, 100 % de clases iguales). La versión int8 coincide con Keras en el 99,5 % de las predicciones (1 imagen distinta de 198).

## Puerta de calidad (integración continua)

`tests/test_puerta_calidad.py` carga el modelo y comprueba las tres reglas del presupuesto: exactitud, tamaño del archivo y latencia (30 corridas de calentamiento y promedio de 300). Si alguna no se cumple, la prueba falla.

GitHub Actions ejecuta las pruebas:
- **en cada `push`**, revisando el modelo final, y
- **a mano**, desde *Actions → Run workflow*, indicando la ruta de otro modelo.

Con `modelos/modelo_sin_entrenar.onnx` la puerta bloquea el modelo, porque no cumple la exactitud ni el tamaño.

Para correr las pruebas localmente:

```bash
pip install -r requirements.txt
pytest tests/ -v

# revisar otro modelo
MODELO=modelos/modelo_sin_entrenar.onnx pytest tests/ -v
```

## API

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Mac/Linux: source venv/bin/activate)
pip install onnxruntime numpy fastapi uvicorn python-multipart pillow
uvicorn app:app --reload
```

Luego abrir **http://127.0.0.1:8000/docs** para probarla desde el navegador.

| Endpoint | Método | Descripción |
|---|---|---|
| `/salud` | GET | Chequeo de salud: confirma que el servicio está activo y el modelo cargado |
| `/predecir` | POST | Recibe una imagen (`imagen`) y devuelve la clase y la confianza |

Ejemplo de respuesta:

```json
{ "clase": "Short", "confianza": 0.462 }
```

## Limitaciones

- El modelo acierta cerca del 70 % de las veces en el conjunto de prueba, por lo que no debería usarse para tomar decisiones sin revisión humana.
- La latencia depende del hardware donde se ejecuta: los umbrales se midieron en Google Colab, y las pruebas de CI corren en los servidores de GitHub.
- Las imágenes muy distintas a las del entrenamiento (otra iluminación, fondos o tipos de prenda) pueden producir predicciones con baja confianza.

## Autores

Dupla: Franco Henríquez - Maciel Martínez — Ingeniería en Informática, INACAP Osorno.
Docente: Javier Elgueta.