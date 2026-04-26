import os
import sys
import uuid
import torch
import numpy as np
from PIL import Image

# --- CONFIGURACIÓN DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
trellis_path = os.path.join(current_dir, "TRELLIS")
if trellis_path not in sys.path:
    sys.path.append(trellis_path)

# Intentar importar TRELLIS
try:
    from trellis.pipelines import TrellisImageTo3DPipeline
except ImportError:
    print("Error: No se pudo importar TRELLIS. Revisa la carpeta TRELLIS.")

class TrellisModlyGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        # Carpeta para guardar los resultados dentro de la extensión
        self.output_dir = os.path.join(current_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self):
        if self.pipeline is None:
            print("Cargando modelo TRELLIS en la GPU...")
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained("Microsoft/TRELLIS-image-to-3d")
            self.pipeline.to(self.device)

    def generate(self, image, seed=0):
        """
        Este método es el que llama Modly. 
        'image' y 'seed' deben coincidir con los IDs del manifest.json
        """
        try:
            self.load_model()

            # 1. Manejar la entrada de imagen
            # Si Modly manda una ruta de archivo, la abrimos
            if isinstance(image, str) and os.path.exists(image):
                input_img = Image.open(image).convert("RGB")
            else:
                input_img = image.convert("RGB")

            # 2. Ejecutar la IA
            print(f"Procesando imagen con semilla {seed}...")
            torch.manual_seed(seed)
            
            outputs = self.pipeline(
                input_img,
                num_samples=1,
                return_flags=["mesh"],
                preprocess_image=True
            )

            # 3. Guardar el archivo .obj en una ruta real
            mesh = outputs['mesh'][0]
            filename = f"result_{uuid.uuid4().hex[:8]}.obj"
            output_path = os.path.join(self.output_dir, filename)
            
            mesh.export(output_path)
            print(f"Archivo guardado en: {output_path}")

            # 4. DEVOLVER DICCIONARIO (Vital para evitar el Error 400)
            # El ID 'mesh' debe coincidir con el ID del output en el manifest.json
            return {"mesh": output_path}

        except Exception as e:
            print(f"ERROR DURANTE LA GENERACIÓN: {str(e)}")
            return {"error": str(e)}
