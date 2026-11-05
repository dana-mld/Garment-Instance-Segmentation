from pathlib import Path
from pipeline_utils import (runtime_base, validate_splits, validate_training_data, validate_annotations, reset_labels, ensure_annotations, polygon_to_yolo, coco_item, clip_box, save_image, export_batch, image_files, benchmark, yolo_predict, maskrcnn_predict, evaluate_coco)


def main():
    import os
    BASE = runtime_base()
    validate_splits(BASE)
    assert os.path.isdir(f'{BASE}/images/train'), 'images/train not found. Upload the training images before continuing.'
    n_train = len(os.listdir(f'{BASE}/images/train'))
    n_val = len(os.listdir(f'{BASE}/images/val'))
    print(f'BASE: {BASE}')
    print(f'Training images: {n_train}, validation: {n_val}')
    import torch
    torch.manual_seed(42)
    print('GPU available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU is active. In Colab, select Runtime -> Change runtime type -> GPU.')
    ensure_annotations('instances_attributes_train2020.json')
    import json, os
    from collections import defaultdict
    KEEP_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25]
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(KEEP_IDS)}
    CLASS_NAMES = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf']
    with open('instances_attributes_train2020.json') as f:
        data = validate_annotations(json.load(f))
    images_by_id = {img['id']: img for img in data['images']}

    def index_by_basename(folder):
        idx = {}
        for path in image_files(folder):
            f = os.path.basename(path)
            idx[os.path.splitext(f)[0]] = f
        return idx
    train_files = index_by_basename(f'{BASE}/images/train')
    val_files = index_by_basename(f'{BASE}/images/val')
    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        if ann['category_id'] not in old_to_new:
            continue
        if 'segmentation' not in ann or len(ann['segmentation']) == 0:
            continue
        annotations_by_image[ann['image_id']].append(ann)

    def build_coco_subset(files_idx):
        images_out = []
        annotations_out = []
        ann_id = 1
        for img_id, img_info in images_by_id.items():
            base_name = os.path.splitext(img_info['file_name'])[0]
            if base_name not in files_idx:
                continue
            anns = annotations_by_image.get(img_id, [])
            images_out.append({'id': img_id, 'file_name': files_idx[base_name], 'width': img_info['width'], 'height': img_info['height']})
            for ann in anns:
                annotations_out.append({'id': ann_id, 'image_id': img_id, 'category_id': old_to_new[ann['category_id']], 'segmentation': ann['segmentation'], 'bbox': ann['bbox'], 'area': ann['area'], 'iscrowd': 0})
                ann_id += 1
        categories_out = [{'id': i, 'name': name} for i, name in enumerate(CLASS_NAMES)]
        return {'images': images_out, 'annotations': annotations_out, 'categories': categories_out}
    train_coco = build_coco_subset(train_files)
    val_coco = build_coco_subset(val_files)
    print(f"Train: {len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations")
    print(f"Val:   {len(val_coco['images'])} images, {len(val_coco['annotations'])} annotations")
    os.makedirs('coco_tmp', exist_ok=True)
    with open('coco_tmp/train.json', 'w') as f:
        json.dump(train_coco, f)
    with open('coco_tmp/val.json', 'w') as f:
        json.dump(val_coco, f)
    os.makedirs(f'{BASE}/annotations_coco', exist_ok=True)
    import shutil
    shutil.copy('coco_tmp/train.json', f'{BASE}/annotations_coco/train.json')
    shutil.copy('coco_tmp/val.json', f'{BASE}/annotations_coco/val.json')
    print('Copied to Drive at:', f'{BASE}/annotations_coco')
    import json
    with open(f'{BASE}/annotations_coco/train.json') as f:
        t = json.load(f)
    with open(f'{BASE}/annotations_coco/val.json') as f:
        v = json.load(f)
    print(f"train.json: {len(t['images'])} images, {len(t['annotations'])} annotations")
    print(f"val.json:   {len(v['images'])} images, {len(v['annotations'])} annotations")
    import torch
    torch.manual_seed(42)
    import numpy as np
    from PIL import Image
    from pycocotools.coco import COCO
    from torch.utils.data import Dataset
    import torchvision.transforms.functional as TF

    class FashionpediaCocoDataset(Dataset):

        def __init__(self, images_dir, ann_file):
            self.images_dir = images_dir
            self.coco = COCO(ann_file)
            self.ids = sorted(self.coco.imgs.keys())

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            return coco_item(self.coco, self.ids[idx], self.images_dir)
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    NUM_CLASSES = len(CLASS_NAMES) + 1

    def build_model(num_classes=NUM_CLASSES):
        model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        return model
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)
    print('Model ready, device:', device)
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        return tuple(zip(*batch))
    validate_training_data(BASE)
    train_dataset = FashionpediaCocoDataset(f'{BASE}/images/train', f'{BASE}/annotations_coco/train.json')
    val_dataset = FashionpediaCocoDataset(f'{BASE}/images/val', f'{BASE}/annotations_coco/val.json')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f'Train: {len(train_dataset)} images, Val: {len(val_dataset)} images')

    def train_one_epoch(model, optimizer, data_loader, device, epoch, print_every=20):
        model.train()
        total_loss = 0.0
        for i, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum((loss for loss in loss_dict.values()))
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
            if (i + 1) % print_every == 0:
                print(f'  epoch {epoch} [{i + 1}/{len(data_loader)}] loss: {losses.item():.4f}')
        avg_loss = total_loss / max(1, len(data_loader))
        print(f'Epoch {epoch} — mean loss: {avg_loss:.4f}')
        return avg_loss
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    for epoch in range(2):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
    os.makedirs(f'{BASE}/runs_maskrcnn', exist_ok=True)
    torch.save(model.state_dict(), f'{BASE}/runs_maskrcnn/test_run.pt')
    print('Smoke-test checkpoint saved.')
    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    NUM_EPOCHS = 25
    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        torch.save(model.state_dict(), f'{BASE}/runs_maskrcnn/last.pt')
    print('Training complete. Model saved to:', f'{BASE}/runs_maskrcnn/last.pt')

    def load_trained_model(weights_path, num_classes=NUM_CLASSES):
        m = build_model(num_classes)
        m.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        m.to(device)
        m.eval()
        return m
    trained_model = load_trained_model(f'{BASE}/runs_maskrcnn/last.pt')
    test_image_path = 'pants2.jpg'
    if os.path.isfile(test_image_path):
        img = Image.open(test_image_path).convert('RGB')
        img_tensor = TF.to_tensor(img).to(device)
        with torch.no_grad():
            prediction = trained_model([img_tensor])[0]
        for label, score in zip(prediction['labels'], prediction['scores']):
            if score > 0.5:
                print(f'{CLASS_NAMES[label.item() - 1]} - confidence: {score.item():.2f}')
    import cv2
    import numpy as np

    def extract_garments_from_image(image_path, model, images_out_dir, conf_threshold=0.5):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f'  [SKIP] Could not read image: {image_path}')
            return []
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]
        img_tensor = TF.to_tensor(img_rgb).to(device)
        with torch.no_grad():
            pred = model([img_tensor])[0]
        img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        source_name = os.path.splitext(os.path.basename(image_path))[0]
        garments = []
        for i in range(len(pred['scores'])):
            score = pred['scores'][i].item()
            if score < conf_threshold:
                continue
            mask = pred['masks'][i, 0].cpu().numpy()
            mask_bin = (mask > 0.5).astype(np.uint8)
            out = img_rgba.copy()
            out[:, :, 3] = (mask_bin * 255).astype(np.uint8)
            x1, y1, x2, y2 = clip_box(pred['boxes'][i].detach().cpu().numpy(), w, h)
            cropped = out[y1:y2, x1:x2]
            if cropped.size == 0:
                continue
            label_id = pred['labels'][i].item() - 1
            class_name_raw = CLASS_NAMES[label_id]
            class_name = class_name_raw.replace(', ', '_').replace(' ', '_')
            img_filename = f'{source_name}__{class_name}_{i}.png'
            img_out_path = os.path.join(images_out_dir, img_filename)
            save_image(img_out_path, cropped)
            garments.append({'image_path': img_filename, 'source_image': os.path.basename(image_path), 'category': class_name_raw, 'category_clean': class_name, 'confidence': float(score), 'bbox': [int(x1), int(y1), int(x2), int(y2)], 'bbox_pixels': {'x': int(x1), 'y': int(y1), 'width': int(x2 - x1), 'height': int(y2 - y1)}, 'aspect_ratio': round((y2 - y1) / (x2 - x1), 3) if x2 - x1 > 0 else 0, 'area_ratio': round((x2 - x1) * (y2 - y1) / (w * h), 4)})
        return garments
    import os, json, shutil, zipfile
    from datetime import datetime

    def generate_batch_output(image_paths, model, model_name, local_output_dir='batch_output', drive_export_dir=f'{BASE}/exports', conf_threshold=0.4, max_images=None, resume=True):
        return export_batch(image_paths, model, extract_garments_from_image, local_output_dir, drive_export_dir, conf_threshold, max_images, resume, model_name)
    image_paths_A = [os.path.join(f'{BASE}/images/train', f) for f in os.listdir(f'{BASE}/images/train')] + [os.path.join(f'{BASE}/images/val', f) for f in os.listdir(f'{BASE}/images/val')]
    print(f'Option A - available images: {len(image_paths_A)}')
    image_paths = image_paths_A
    metadata_path = generate_batch_output(image_paths, trained_model, model_name='maskrcnn_resnet50_fpn_v2', local_output_dir='batch_output_maskrcnn', max_images=10)
    metadata_path = generate_batch_output(image_paths, trained_model, model_name='maskrcnn_resnet50_fpn_v2', local_output_dir='batch_output_maskrcnn', max_images=None, resume=True)
    with open(metadata_path) as f:
        final_metadata = validate_annotations(json.load(f))
    print(f"\nTotal images processed: {final_metadata['total_source_images']}")
    print(f"Total garments extracted:  {final_metadata['total_garments']}")
    print(f"Categories found: {final_metadata['categories_available']}")


if __name__ == "__main__":
    main()
