import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

MODELO = "modelos/modelo_p65_int8.onnx"
CLASES =  ['Camisas', 'Faldas', 'Pantalones', 'Poleras', 'Polerones', 'Short']   

app = FastAPI(title="Clasificador de ropa")
sesion = ort.InferenceSession(MODELO)
entrada = sesion.get_inputs()[0].name

def preprocesar(datos: bytes) -> np.ndarray:
    # Mismo preprocesamiento del entrenamiento: RGB, 128x128, escala 0–1
    img = Image.open(io.BytesIO(datos)).convert("RGB").resize((128, 128), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    return x[np.newaxis, ...]          # forma (1, 128, 128, 3)

@app.get("/salud")
def salud():
    # Chequeo de salud: confirma que el servicio está vivo y el modelo cargado
    return {"estado": "ok", "modelo": MODELO}

@app.post("/predecir")
async def predecir(imagen: UploadFile = File(...)):
    try:
        x = preprocesar(await imagen.read())
    except Exception:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida")
    prob = sesion.run(None, {entrada: x})[0][0]
    i = int(prob.argmax())
    return {"clase": CLASES[i], "confianza": round(float(prob[i]), 3)}