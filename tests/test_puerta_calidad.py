import os, time
import numpy as np
import onnxruntime as ort
import pytest

# El modelo a revisar se puede cambiar con una variable de entorno (útil para probar el rechazo)
MODELO = os.environ.get("MODELO", "modelos/modelo_p65_int8.onnx")

# Presupuesto de la clase 1
EXACTITUD_MIN = 0.642
TAMANO_MAX_MB = 0.20
LATENCIA_MAX_MS = 12.96

@pytest.fixture(scope="module")
def sesion():
    return ort.InferenceSession(MODELO)

@pytest.fixture(scope="module")
def datos():
    d = np.load("datos/test_data.npz")
    x = d["x"].astype(np.float32) / 255.0   # volver a la escala 0–1
    return x, d["y"]

def predecir(sesion, x):
    nombre = sesion.get_inputs()[0].name
    return sesion.run(None, {nombre: x})[0]

def test_exactitud(sesion, datos):
    x, y = datos
    acc = float((predecir(sesion, x).argmax(axis=1) == y).mean())  
    assert acc >= EXACTITUD_MIN, f"Exactitud {acc:.3f} bajo el mínimo {EXACTITUD_MIN}"

def test_tamano():
    mb = os.path.getsize(MODELO) / 1e6   
    assert mb <= TAMANO_MAX_MB, f"Pesa {mb:.3f} MB, máximo {TAMANO_MAX_MB}"

def test_latencia(sesion, datos):
    x, _ = datos
    uno = x[:1]
    for _ in range(30):
        predecir(sesion, uno)
    t = []
    for _ in range(300):
        t0 = time.perf_counter()
        predecir(sesion, uno)
        t.append((time.perf_counter() - t0) * 1000)
    ms = float(np.mean(t))
    assert ms <= LATENCIA_MAX_MS, f"Latencia {ms:.2f} ms, máximo {LATENCIA_MAX_MS}"