import os
import sys
import argparse

# --- CONFIGURACIÓN DE RUTAS ---
# Añadimos la subcarpeta 'TRELLIS' al PATH de Python para que pueda importar sus módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
trellis_path = os.path.join(current_dir, "TRELLIS")
if os.path.exists(trellis_path):
    sys.path.append(trellis_path)
else:
    print(f"ERROR: No se encontró la carpeta {trellis_path}. Asegúrate de haber clonado el repo de TRELLIS ahí.")
    sys.exit(1)

# --- IMPORTS DE LIBRERÍAS ---
try:
    import torch
    import numpy as np
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils
except ImportError as e:
    print(f"ERROR DE IMPORTACIÓN: {e}")
    print("Asegúrate de tener activado el entorno 'trellis-env' y haber instalado las dependencias.")
    sys.exit(1)

class TrellisModlyGenerator:
    def __init__(self):
        # Seleccionamos la GPU (obligatoria para TRELLIS en tiempos razonables)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cpu":
            print("ADVERTENCIA: No se detectó GPU. TRELLIS será extremadamente lento o fallará.")

        print("Cargando pipeline de Microsoft TRELLIS...")
        try:
            # Cargamos el modelo desde Hugging Face
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained("Microsoft/TRELLIS-image-to-3d")
            self.pipeline.to(self.device)
            print("Modelo cargado correctamente.")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            sys.exit(1)

    def generate(self, input_path, output_path):
        print(f"Iniciando proceso para: {input_path}")
        
        # 1. Abrir y preparar la imagen
        image = Image.open(input_path).convert("RGB")

        # 2. Inferencia (Generación del modelo latente)
        # Nota: TRELLIS genera una estructura SLAT (Sparse Latent)
        print("Generando estructura 3D (esto consume mucha VRAM)...")
        outputs = self.pipeline(
            image,
            num_samples=1,
            return_flags=["mesh"],
            preprocess_image=True  # Recomendado para fotos con fondo
        )

        # 3. Procesamiento y Extracción de la malla
        # TRELLIS devuelve un diccionario; extraemos la malla de la lista
        mesh = outputs['mesh'][0]

        # 4. Exportar a archivo compatible con Modly (.obj o .glb)
        print(f"Exportando resultado a: {output_path}")
        mesh.export(output_path)
        print("¡Proceso finalizado!")

def main():
    parser = argparse.ArgumentParser(description="Generador TRELLIS para Modly")
    parser.add_argument("--input", required=True, help="Ruta de la imagen de entrada")
    parser.add_argument("--output", required=True, help="Ruta del archivo .obj de salida")
    args = parser.parse_args()

    # Ejecutar la generación
    generator = TrellisModlyGenerator()
    generator.generate(args.input, args.output)

if __name__ == "__main__":
    main()
