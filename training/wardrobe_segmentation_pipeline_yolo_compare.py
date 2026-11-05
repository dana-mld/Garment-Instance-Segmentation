from pathlib import Path
from pipeline_utils import (runtime_base, validate_splits, validate_training_data, validate_annotations, reset_labels, ensure_annotations, polygon_to_yolo, coco_item, clip_box, save_image, export_batch, image_files, benchmark, yolo_predict, maskrcnn_predict, evaluate_coco)


def main():
    import PIL
    print('Installed Pillow version:', PIL.__version__)
    import PIL
    print('Pillow version:', PIL.__version__)
    print('Pillow version displayed.')
    import os
    BASE = runtime_base()
    validate_splits(BASE)
    os.makedirs(f'{BASE}/images/train', exist_ok=True)
    os.makedirs(f'{BASE}/images/val', exist_ok=True)
    reset_labels(f'{BASE}/labels/train')
    reset_labels(f'{BASE}/labels/val')
    print('Directory structure created at:', BASE)
    import torch
    torch.manual_seed(42)
    print('GPU available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU is active. In Colab, select Runtime -> Change runtime type -> GPU.')
    ensure_annotations('instances_attributes_train2020.json')
    n_train = len(os.listdir(f'{BASE}/images/train'))
    n_val = len(os.listdir(f'{BASE}/images/val'))
    print(f'Training images: {n_train}')
    print(f'Validation images: {n_val}')
    if n_train == 0:
        print('Upload the images to Drive before continuing.')
    import json
    from collections import defaultdict
    KEEP_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25]
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(KEEP_IDS)}
    CLASS_NAMES = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf']
    with open('instances_attributes_train2020.json') as f:
        data = validate_annotations(json.load(f))
    images_by_id = {img['id']: img for img in data['images']}
    existing_images = set((os.path.splitext(f)[0] for f in os.listdir(f'{BASE}/images/train'))) | set((os.path.splitext(f)[0] for f in os.listdir(f'{BASE}/images/val')))
    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        if ann['category_id'] not in old_to_new:
            continue
        if 'segmentation' not in ann or len(ann['segmentation']) == 0:
            continue
        img_info = images_by_id[ann['image_id']]
        base_name = os.path.splitext(img_info['file_name'])[0]
        if base_name not in existing_images:
            continue
        annotations_by_image[ann['image_id']].append(ann)
    print(f'Matching images found: {len(annotations_by_image)}')

    def convert_polygon(segmentation, img_w, img_h):
        return polygon_to_yolo(segmentation, img_w, img_h)
    import shutil
    if os.path.exists('labels_tmp'):
        shutil.rmtree('labels_tmp')
    reset_labels('labels_tmp')
    count_written = 0
    count_skipped = 0
    for image_id, anns in annotations_by_image.items():
        img_info = images_by_id[image_id]
        img_w, img_h = (img_info['width'], img_info['height'])
        base_name = os.path.splitext(img_info['file_name'])[0]
        lines = []
        for ann in anns:
            try:
                poly_str = convert_polygon(ann['segmentation'], img_w, img_h)
                if len(poly_str.split()) < 6:
                    count_skipped += 1
                    continue
                new_class_id = old_to_new[ann['category_id']]
                lines.append(f'{new_class_id} {poly_str}')
            except Exception:
                count_skipped += 1
                continue
        if lines:
            with open(f'labels_tmp/{base_name}.txt', 'w') as f:
                f.write('\n'.join(lines))
            count_written += 1
    print(f'Label files written locally: {count_written}')
    print(f'Invalid polygons skipped: {count_skipped}')
    train_image_names = set((os.path.splitext(f)[0] for f in os.listdir(f'{BASE}/images/train')))
    val_image_names = set((os.path.splitext(f)[0] for f in os.listdir(f'{BASE}/images/val')))
    if os.path.exists('labels_final'):
        shutil.rmtree('labels_final')
    os.makedirs('labels_final/train', exist_ok=True)
    os.makedirs('labels_final/val', exist_ok=True)
    moved_train = 0
    moved_val = 0
    orphan = 0
    for fname in os.listdir('labels_tmp'):
        name = os.path.splitext(fname)[0]
        src = f'labels_tmp/{fname}'
        if name in train_image_names:
            shutil.copy(src, f'labels_final/train/{fname}')
            moved_train += 1
        elif name in val_image_names:
            shutil.copy(src, f'labels_final/val/{fname}')
            moved_val += 1
        else:
            orphan += 1
    print(f'Local -> train: {moved_train}, val: {moved_val}, orphans: {orphan}')
    reset_labels(f'{BASE}/labels/train')
    reset_labels(f'{BASE}/labels/val')
    shutil.copytree('labels_final/train', f'{BASE}/labels/train', dirs_exist_ok=True)
    shutil.copytree('labels_final/val', f'{BASE}/labels/val', dirs_exist_ok=True)
    print('Copied to Drive successfully.')
    import random
    val_current = len(os.listdir(f'{BASE}/images/val'))
    if val_current == 0:
        image_names = set((os.path.splitext(f)[0] for f in os.listdir(f'{BASE}/images/train')))
        label_names = set((os.path.splitext(f)[0] for f in os.listdir(f'{BASE}/labels/train')))
        valid_names = sorted(image_names & label_names)
        random.seed(42)
        random.shuffle(valid_names)
        val_count = max(1, int(len(valid_names) * 0.1))
        val_names = valid_names[:val_count]
        for name in val_names:
            for ext in ['.jpg', '.jpeg', '.png']:
                src = f'{BASE}/images/train/{name}{ext}'
                if os.path.exists(src):
                    shutil.move(src, f'{BASE}/images/val/{name}{ext}')
                    break
            label_src = f'{BASE}/labels/train/{name}.txt'
            if os.path.exists(label_src):
                shutil.move(label_src, f'{BASE}/labels/val/{name}.txt')
        print(f'Images moved to validation: {len(val_names)}')
    else:
        print(f'Validation already contains {val_current} images; splitting skipped.')
    for split in ['train', 'val']:
        n_img = len(os.listdir(f'{BASE}/images/{split}'))
        n_lbl = len(os.listdir(f'{BASE}/labels/{split}'))
        print(f'{split}: {n_img} images, {n_lbl} labels')
    yaml_content = f'path: {BASE}\ntrain: images/train\nval: images/val\n\nnames:\n'
    for idx, name in enumerate(CLASS_NAMES):
        yaml_content += f'  {idx}: {name}\n'
    with open(f'{BASE}/data.yaml', 'w') as f:
        f.write(yaml_content)
    print('data.yaml written to:', f'{BASE}/data.yaml')
    print(yaml_content)
    validate_training_data(BASE)
    from ultralytics import YOLO
    import time
    MODELS_TO_COMPARE = {'yolov8n-seg': 'yolov8n-seg.pt', 'yolo11n-seg': 'yolo11n-seg.pt'}
    for model_name, checkpoint in MODELS_TO_COMPARE.items():
        print(f'\n=== Smoke test: {model_name} ===')
        model = YOLO(checkpoint)
        model.train(seed=42, data=f'{BASE}/data.yaml', epochs=5, imgsz=640, batch=8, workers=0, project=f'{BASE}/runs', name=f'test_run_{model_name}')
    training_times = {}
    trained_paths = {}
    for model_name, checkpoint in MODELS_TO_COMPARE.items():
        print(f'\n=== Full training: {model_name} ===')
        model = YOLO(checkpoint)
        start = time.perf_counter()
        model.train(seed=42, data=f'{BASE}/data.yaml', epochs=60, imgsz=640, batch=8, workers=0, project=f'{BASE}/runs', name=f'full_run_{model_name}', patience=15)
        trained_paths[model_name] = str(model.trainer.best)
        elapsed = time.perf_counter() - start
        training_times[model_name] = elapsed
        print(f'{model_name} — training completed in {elapsed / 60:.1f} minutes')
    import glob
    comparison = {}
    val_images_sample = image_files(f'{BASE}/images/val')[:30]
    for model_name in MODELS_TO_COMPARE:
        weights_path = trained_paths[model_name]
        trained = YOLO(weights_path)
        metrics = trained.val(data=f'{BASE}/data.yaml')
        avg_inference_ms = benchmark(lambda p: yolo_predict(trained, p), val_images_sample)
        comparison[model_name] = {'box_mAP50-95': float(metrics.box.map), 'box_mAP50': float(metrics.box.map50), 'mask_mAP50-95': float(metrics.seg.map), 'mask_mAP50': float(metrics.seg.map50), 'training_time_min': training_times[model_name] / 60, 'avg_inference_ms': avg_inference_ms}
    print(f"\n{'Model':<15} {'mask mAP50-95':<15} {'mask mAP50':<12} {'box mAP50-95':<15} {'training (min)':<16} {'processing (ms)'}")
    print('-' * 90)
    for model_name, m in comparison.items():
        print(f"{model_name:<15} {m['mask_mAP50-95']:<15.3f} {m['mask_mAP50']:<12.3f} {m['box_mAP50-95']:<15.3f} {m['training_time_min']:<16.1f} {m['avg_inference_ms']:.1f}")
    with open(f'{BASE}/yolo_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    best_model_name = max(comparison, key=lambda k: comparison[k]['mask_mAP50-95'])
    print(f'\nBest segmentation model (mask mAP50-95): {best_model_name}')
    trained_model_path = trained_paths[best_model_name]
    trained_model = YOLO(trained_model_path)
    print(f'Using: {best_model_name}')
    test_image = 'pants2.jpg'
    if os.path.isfile(test_image):
        results = trained_model(test_image)
        for r in results:
            r.show()
            if r.boxes is not None:
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    print(f'{r.names[int(cls)]} - confidence: {float(conf):.2f}')
    import cv2
    import numpy as np
    import os, json, shutil, zipfile
    from datetime import datetime
    validate_training_data(BASE)
    from ultralytics import YOLO

    def extract_garments_from_image(image_path, model, images_out_dir, conf_threshold=0.4):
        results = model(image_path, conf=conf_threshold, verbose=False, retina_masks=True)[0]
        img = cv2.imread(image_path)
        if img is None:
            print(f'  [SKIP] Could not read image: {image_path}')
            return []
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        h, w = img.shape[:2]
        source_name = os.path.splitext(os.path.basename(image_path))[0]
        garments = []
        if results.masks is None:
            return garments
        for i, (mask, box, cls, conf) in enumerate(zip(results.masks.data, results.boxes.xyxy, results.boxes.cls, results.boxes.conf)):
            mask_np = mask.cpu().numpy()
            mask_resized = mask_np if mask_np.shape == (h, w) else cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            out = img_rgba.copy()
            out[:, :, 3] = (mask_resized * 255).astype(np.uint8)
            x1, y1, x2, y2 = clip_box(box.detach().cpu().tolist(), w, h)
            cropped = out[y1:y2, x1:x2]
            if cropped.size == 0:
                continue
            if cropped.size == 0:
                continue
            class_name = results.names[int(cls)].replace(', ', '_').replace(' ', '_')
            img_filename = f'{source_name}__{class_name}_{i}.png'
            img_out_path = os.path.join(images_out_dir, img_filename)
            save_image(img_out_path, cropped)
            garments.append({'image_path': img_filename, 'source_image': os.path.basename(image_path), 'category': results.names[int(cls)], 'category_clean': class_name, 'confidence': float(conf), 'bbox': [int(x1), int(y1), int(x2), int(y2)], 'bbox_pixels': {'x': int(x1), 'y': int(y1), 'width': int(x2 - x1), 'height': int(y2 - y1)}, 'aspect_ratio': round((y2 - y1) / (x2 - x1), 3) if x2 - x1 > 0 else 0, 'area_ratio': round((x2 - x1) * (y2 - y1) / (w * h), 4)})
        return garments

    def generate_batch_output(image_paths, model, local_output_dir='batch_output', drive_export_dir=f'{BASE}/exports', conf_threshold=0.4, max_images=None, resume=True):
        return export_batch(image_paths, model, extract_garments_from_image, local_output_dir, drive_export_dir, conf_threshold, max_images, resume, str(getattr(model, 'ckpt_path', 'yolo')))
    image_paths_A = [os.path.join(f'{BASE}/images/train', f) for f in os.listdir(f'{BASE}/images/train')] + [os.path.join(f'{BASE}/images/val', f) for f in os.listdir(f'{BASE}/images/val')]
    print(f'Option A - available images: {len(image_paths_A)}')
    model = trained_model
    image_paths = image_paths_A
    metadata_path = generate_batch_output(image_paths, model, max_images=10)
    metadata_path = generate_batch_output(image_paths, model, max_images=None, resume=True)
    with open(metadata_path) as f:
        final_metadata = validate_annotations(json.load(f))
    print(f"\nTotal images processed: {final_metadata['total_source_images']}")
    print(f"Total garments extracted:  {final_metadata['total_garments']}")
    print(f"Categories found: {final_metadata['categories_available']}")


if __name__ == "__main__":
    main()
