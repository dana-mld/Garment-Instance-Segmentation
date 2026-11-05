import sys
import os
import json
from pathlib import Path
from segmentation import GarmentSegmenter
ROOT = Path(__file__).resolve().parent
MODEL_PATH = str(ROOT / 'models/yolov8n_seg_best.pt')
EXTENSII_VALIDE = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def gaseste_poze(folder: str) -> list:
    poze = []
    for nume in sorted(os.listdir(folder)):
        if nume.lower().endswith(EXTENSII_VALIDE):
            poze.append(os.path.join(folder, nume))
    return poze

def proceseaza_folder(input_dir: str, output_dir: str):
    if not os.path.isdir(input_dir):
        print(f'Input directory does not exist: {input_dir}')
        sys.exit(1)
    poze = gaseste_poze(input_dir)
    if not poze:
        print(f'No images found in {input_dir} (supported extensions: {EXTENSII_VALIDE})')
        sys.exit(1)
    print(f'Loading model from {MODEL_PATH} ...')
    segmenter = GarmentSegmenter(MODEL_PATH)
    os.makedirs(output_dir, exist_ok=True)
    for cale_poza in poze:
        nume_fisier = os.path.splitext(os.path.basename(cale_poza))[0]
        print(f'\nProcessing: {cale_poza}')
        with open(cale_poza, 'rb') as f:
            image_bytes = f.read()
        try:
            garments = segmenter.segment(image_bytes)
        except ValueError as e:
            print(f'  Error: {e}')
            continue
        if not garments:
            print('  No garments detected.')
            continue
        folder_poza = os.path.join(output_dir, nume_fisier)
        os.makedirs(folder_poza, exist_ok=True)
        rezultate_json = []
        for i, g in enumerate(garments):
            crop_path = os.path.join(folder_poza, f'{g.category}_{i}_crop.png')
            mask_path = os.path.join(folder_poza, f'{g.category}_{i}_mask.png')
            with open(crop_path, 'wb') as f:
                f.write(g.image_bytes)
            with open(mask_path, 'wb') as f:
                f.write(g.mask_bytes)
            rezultate_json.append({'category': g.category, 'confidence': g.confidence, 'bbox': g.bbox, 'crop_bbox': g.crop_bbox, 'mask_polygon': g.mask_polygon, 'crop_file': os.path.basename(crop_path), 'mask_file': os.path.basename(mask_path)})
            print(f'  {g.category} (conf: {g.confidence:.2f}) -> {crop_path}')
        with open(os.path.join(folder_poza, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(rezultate_json, f, ensure_ascii=False, indent=2)
    print(f'\nDone. Results saved to: {output_dir}')
if __name__ == '__main__':
    input_dir = sys.argv[1] if len(sys.argv) >= 2 else str(ROOT / 'input')
    output_dir = sys.argv[2] if len(sys.argv) >= 3 else str(ROOT / 'output')
    proceseaza_folder(input_dir, output_dir)
