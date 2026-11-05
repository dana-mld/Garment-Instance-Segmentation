from pathlib import Path
from pipeline_utils import (runtime_base, validate_splits, validate_training_data, validate_annotations, reset_labels, ensure_annotations, polygon_to_yolo, coco_item, clip_box, save_image, export_batch, image_files, benchmark, yolo_predict, maskrcnn_predict, evaluate_coco)


def main():
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
    ensure_annotations('instances_attributes_train2020.json')
    n_train = len(os.listdir(f'{BASE}/images/train'))
    n_val = len(os.listdir(f'{BASE}/images/val'))
    print(f'Training images: {n_train}')
    print(f'Validation images: {n_val}')
    print('If the training set is empty, upload the images to Drive before continuing.')
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
    import shutil
    train_image_names = set((os.path.splitext(f)[0] for f in os.listdir(f'{BASE}/images/train')))
    val_image_names = set((os.path.splitext(f)[0] for f in os.listdir(f'{BASE}/images/val')))
    moved_train = 0
    moved_val = 0
    orphan = 0
    for fname in os.listdir('labels_tmp'):
        name = os.path.splitext(fname)[0]
        src = f'labels_tmp/{fname}'
        if name in train_image_names:
            shutil.copy(src, f'{BASE}/labels/train/{fname}')
            moved_train += 1
        elif name in val_image_names:
            shutil.copy(src, f'{BASE}/labels/val/{fname}')
            moved_val += 1
        else:
            orphan += 1
    print(f'Labels copied to training: {moved_train}')
    print(f'Labels copied to validation: {moved_val}')
    print(f'Orphan labels (no matching image): {orphan}')
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
        print(f'Validation already contains {val_current} images; automatic splitting skipped.')
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
    model = YOLO('yolov8n-seg.pt')
    results = model.train(seed=42, data=f'{BASE}/data.yaml', epochs=5, imgsz=640, batch=8, workers=0, project=f'{BASE}/runs', name='test_run')
    model_full = YOLO('yolov8n-seg.pt')
    results_full = model_full.train(seed=42, data=f'{BASE}/data.yaml', epochs=60, imgsz=640, batch=8, workers=0, project=f'{BASE}/runs', name='full_run', patience=15)
    trained_model_path = str(model_full.trainer.best)
    trained_model = YOLO(trained_model_path)
    test_image = 'test.jpg'
    if os.path.isfile(test_image):
        results = trained_model(test_image)
        for r in results:
            r.show()
            if r.boxes is not None:
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    print(f'{r.names[int(cls)]} - confidence: {float(conf):.2f}')
    import cv2
    import numpy as np

    def extract_garments(image_path, model, output_dir='output', conf_threshold=0.4):
        os.makedirs(output_dir, exist_ok=True)
        results = model(image_path, conf=conf_threshold, retina_masks=True)[0]
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f'Cannot read image: {image_path}')
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        h, w = img.shape[:2]
        garments = []
        if results.masks is None:
            print('No garments detected.')
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
            class_name = results.names[int(cls)].replace(', ', '_').replace(' ', '_')
            out_path = f'{output_dir}/{class_name}_{i}.png'
            save_image(out_path, cropped)
            garments.append({'class': results.names[int(cls)], 'confidence': float(conf), 'image_path': out_path, 'bbox': [x1, y1, x2, y2]})
            print(f'Saved: {out_path} ({results.names[int(cls)]}, conf {float(conf):.2f})')
        return garments
    if os.path.isfile('test.jpg'):
        garments = extract_garments('test.jpg', trained_model)
        print(garments)


if __name__ == "__main__":
    main()
