import json
import os
import time
import zipfile
from pathlib import Path


def runtime_base():
    base = os.environ.get('WARDROBE_BASE', '/content/drive/MyDrive/wardrobe_dataset' if Path('/content').is_dir() else 'wardrobe_dataset')
    if base.startswith('/content/drive/'):
        from google.colab import drive
        drive.mount('/content/drive')
    return str(Path(base).resolve())


def reset_labels(folder):
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    for label in path.glob('*.txt'):
        label.unlink()
    for cache in (path.with_suffix('.cache'), path / 'labels.cache'):
        if cache.is_file():
            cache.unlink()


def validate_annotations(data):
    import math
    clean = []
    for ann in data['annotations']:
        bbox = ann.get('bbox', [])
        if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox) or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        seg = ann.get('segmentation')
        if isinstance(seg, list):
            seg = [p for p in seg if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0 and all(math.isfinite(float(v)) for v in p)]
            if not seg:
                continue
            ann = dict(ann, segmentation=seg)
        elif not isinstance(seg, dict):
            continue
        clean.append(ann)
    print(f"Valid annotations: {len(clean)}/{len(data['annotations'])}")
    return dict(data, annotations=clean)


def image_files(folder):
    return sorted(str(p) for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})


def validate_splits(base):
    groups = []
    for split in ('train', 'val'):
        folder = Path(base) / 'images' / split
        folder.mkdir(parents=True, exist_ok=True)
        paths = image_files(folder)
        stems = [Path(p).stem for p in paths]
        if len(stems) != len(set(stems)):
            raise ValueError(f'Duplicate image stems in {folder}')
        groups.append(set(stems))
    if groups[0] & groups[1]:
        raise ValueError('Train and validation contain overlapping image names')
    if not groups[0]:
        raise ValueError(f'No training images in {base}/images/train')


def ensure_annotations(filename):
    if not Path(filename).is_file():
        raise FileNotFoundError(f'Place Fashionpedia instances_attributes_train2020.json in the working directory: {filename}')


def validate_training_data(base):
    validate_splits(base)
    if not image_files(Path(base) / 'images' / 'val'):
        raise ValueError('Validation images are required before training')


def polygon_to_yolo(segmentation, width, height):
    import math
    if width <= 0 or height <= 0 or not isinstance(segmentation, list):
        raise ValueError('Expected polygon segmentation and positive image dimensions')
    polygons = [p for p in segmentation if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0 and all(math.isfinite(float(x)) for x in p)]
    if not polygons:
        raise ValueError('No valid polygon')
    def area(p):
        return abs(sum(p[i] * p[(i + 3) % len(p)] - p[(i + 2) % len(p)] * p[i + 1] for i in range(0, len(p), 2))) / 2
    polygon = max(polygons, key=area)
    if area(polygon) <= 0:
        raise ValueError('Zero-area polygon')
    return ' '.join(f'{min(max(float(x) / (width if i % 2 == 0 else height), 0), 1):.6f}' for i, x in enumerate(polygon))


def clip_box(box, width, height):
    import math
    x1, y1, x2, y2 = map(float, box)
    return (max(0, min(width, math.floor(x1))), max(0, min(height, math.floor(y1))),
            max(0, min(width, math.ceil(x2))), max(0, min(height, math.ceil(y2))))


def coco_item(coco, img_id, images_dir):
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    info = coco.imgs[img_id]
    with Image.open(Path(images_dir) / info['file_name']) as image:
        image = image.convert('RGB')
    width, height = image.size
    boxes, labels, masks, areas, crowds = [], [], [], [], []
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
        x, y, w, h = ann['bbox']
        x1, y1 = max(0., x), max(0., y)
        x2, y2 = min(float(width), x + w), min(float(height), y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        mask = coco.annToMask(ann)
        if mask.shape != (height, width) or not mask.any():
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(ann['category_id'] + 1)
        masks.append(mask)
        areas.append(float(mask.sum()))
        crowds.append(int(ann.get('iscrowd', 0)))
    target = {
        'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        'labels': torch.tensor(labels, dtype=torch.int64),
        'masks': torch.as_tensor(np.stack(masks) if masks else np.zeros((0, height, width), dtype=np.uint8), dtype=torch.uint8),
        'image_id': torch.tensor([img_id]),
        'area': torch.tensor(areas, dtype=torch.float32),
        'iscrowd': torch.tensor(crowds, dtype=torch.int64),
    }
    return to_tensor(image), target


def save_image(path, image):
    import cv2
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Cannot write image: {path}')


def export_batch(image_paths, model, extract, output_dir, export_dir, threshold, max_images, resume, model_name):
    import shutil
    import hashlib
    output = Path(output_dir)
    images_out = output / 'images'
    images_out.mkdir(parents=True, exist_ok=True)
    metadata_path = output / 'metadata.json'
    paths = sorted({str(Path(p).resolve()) for p in image_paths if Path(p).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}})
    stems = [Path(p).stem for p in paths]
    if len(stems) != len(set(stems)):
        raise ValueError('Input names collide; use unique image stems across folders')
    signature = hashlib.sha256()
    checkpoint = Path(str(getattr(model, 'ckpt_path', '')))
    if checkpoint.is_file():
        with checkpoint.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                signature.update(chunk)
    elif hasattr(model, 'state_dict'):
        for key, tensor in model.state_dict().items():
            signature.update(key.encode())
            signature.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    else:
        signature.update(str(model_name).encode())
    model_signature = signature.hexdigest()
    metadata = {'model_used': model_name, 'model_signature': model_signature, 'confidence_threshold': threshold, 'processed_images': [], 'garments': []}
    if resume and metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding='utf-8'))
        if previous.get('model_signature') != model_signature or previous.get('model_used') != model_name or previous.get('confidence_threshold') != threshold:
            raise ValueError('Resume settings differ; use a new output directory')
        metadata = previous
    elif any(images_out.iterdir()) or metadata_path.exists():
        raise ValueError('Output directory already contains an export; use a new directory or resume=True')
    completed = set(metadata['processed_images'])
    todo = [p for p in paths if Path(p).name not in completed]
    if max_images is not None:
        if max_images < 0:
            raise ValueError('max_images must be non-negative')
        todo = todo[:max_images]
    for path in todo:
        garments = extract(path, model, str(images_out), threshold)
        metadata['garments'].extend(garments)
        completed.add(Path(path).name)
        metadata['processed_images'] = sorted(completed)
        metadata['total_source_images'] = len(completed)
        metadata['total_garments'] = len(metadata['garments'])
        metadata['categories_available'] = sorted({g['category'] for g in metadata['garments']})
        temporary = metadata_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        temporary.replace(metadata_path)
    if not metadata_path.exists():
        metadata.update(total_source_images=len(completed), total_garments=len(metadata['garments']), categories_available=sorted({g['category'] for g in metadata['garments']}))
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(output))
    export = Path(export_dir)
    export.mkdir(parents=True, exist_ok=True)
    destination = export / archive.name
    if archive.resolve() != destination.resolve():
        shutil.copy2(archive, destination)
    return str(metadata_path)


def yolo_predict(model, path):
    import cv2
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Cannot read {path}')
    result = model(image, conf=0.001, max_det=100, retina_masks=True, verbose=False)[0]
    if result.masks is None:
        return []
    return [(mask.cpu().numpy() > 0.5, int(label), float(score))
            for mask, label, score in zip(result.masks.data, result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())]


def maskrcnn_predict(model, path, device):
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    model.eval()
    model.roi_heads.score_thresh = 0.001
    model.roi_heads.detections_per_img = 100
    with Image.open(path) as image:
        tensor = to_tensor(image.convert('RGB')).to(device)
    with torch.inference_mode():
        result = model([tensor])[0]
    return [(mask[0].cpu().numpy() > 0.5, int(label) - 1, float(score))
            for mask, label, score in zip(result['masks'], result['labels'].cpu().tolist(), result['scores'].cpu().tolist()) if score >= 0.001]


def benchmark(predict, paths, warmup=3, repeats=3):
    import torch
    if not paths:
        raise ValueError('No validation images for benchmarking')
    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    for i in range(warmup):
        predict(paths[i % len(paths)])
    sync()
    start = time.perf_counter()
    for _ in range(repeats):
        for path in paths:
            predict(path)
    sync()
    return (time.perf_counter() - start) * 1000 / (len(paths) * repeats)


def evaluate_coco(model, ann_file, images_dir, kind, device):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_utils
    gt = COCO(ann_file)
    gt.dataset.setdefault('info', {})
    ids = gt.getImgIds()
    if not ids:
        raise ValueError('No validation annotations')
    if kind == 'maskrcnn':
        model.roi_heads.score_thresh = 0.001
        model.roi_heads.detections_per_img = 100
    predictions = []
    for img_id in ids:
        info = gt.imgs[img_id]
        path = str(Path(images_dir) / info['file_name'])
        output = yolo_predict(model, path) if kind == 'yolo' else maskrcnn_predict(model, path, device)
        for mask, label, score in output:
            if mask.shape != (info['height'], info['width']):
                raise ValueError('Prediction mask does not match original image dimensions')
            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('utf-8')
            predictions.append({'image_id': img_id, 'category_id': label, 'segmentation': rle, 'score': score})
    if not predictions:
        return {'mask_mAP50-95': 0.0, 'mask_mAP50': 0.0}
    dt = gt.loadRes(predictions)
    evaluator = COCOeval(gt, dt, 'segm')
    evaluator.params.imgIds = ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {'mask_mAP50-95': float(evaluator.stats[0]), 'mask_mAP50': float(evaluator.stats[1])}
