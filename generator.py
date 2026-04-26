import os
import sys
import argparse

# --- CONFIGURACIÓN DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
trellis_path = os.path.join(current_dir, "TRELLIS")

if os.path.exists(trellis_path):
    if trellis_path not in sys.path:
        sys.path.append(trellis_path)
else:
    # En lugar de sys.exit, lanzamos un error que Modly pueda capturar
    raise RuntimeError(f"No se encontró la carpeta TRELLIS en {trellis_path}")

# --- IMPORTS DE LIBRERÍAS ---
try:
    import torch
    import numpy as np
    from PIL import Image
    # Intentamos importar TRELLIS solo si el path ya está listo
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils
except Exception as e:
    print(f"Error cargando dependencias: {e}")
    # No cerramos el proceso, dejamos que Modly maneje el error

class TrellisModlyGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None

    def load_model(self):
        """Carga el modelo solo cuando es necesario para ahorrar VRAM al inicio"""
        if self.pipeline is None:
            print("Cargando pipeline de Microsoft TRELLIS...")
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained("Microsoft/TRELLIS-image-to-3d")
            self.pipeline.to(self.device)

    def generate(self, image, seed=0):
        # Modly ahora pasa los objetos directamente o rutas
        self.load_model()
        
        # Si Modly pasa una ruta (string), la abrimos. Si es una imagen PIL, la usamos.
        if isinstance(image, str):
            input_img = Image.open(image).convert("RGB")
        else:
            input_img = image.convert("RGB")

        print(f"Generando 3D con semilla: {seed}")
        
        outputs = self.pipeline(
            input_img,
            num_samples=1,
            return_flags=["mesh"],
            preprocess_image=True
        )

        # Devolvemos la malla para que Modly la guarde donde necesite
        return outputs['mesh'][0]

# El bloque main solo se usa para tus pruebas manuales en la terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gen = TrellisModlyGenerator()
    mesh = gen.generate(args.input)
    mesh.export(args.output)
