from pathlib import Path
from pipeline_utils import (runtime_base, validate_splits, validate_training_data, validate_annotations, reset_labels, ensure_annotations, polygon_to_yolo, coco_item, clip_box, save_image, export_batch, image_files, benchmark, yolo_predict, maskrcnn_predict, evaluate_coco)


def main():
    import PIL
    print('Pillow:', PIL.__version__)
    import os
    BASE = runtime_base()
    validate_splits(BASE)
    assert os.path.isdir(f'{BASE}/images/train'), 'images/train not found. Upload the images first.'
    print(f"Training images: {len(os.listdir(f'{BASE}/images/train'))}")
    print(f"Validation images: {len(os.listdir(f'{BASE}/images/val'))}")
    import torch
    torch.manual_seed(42)
    print('GPU available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
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
        return {Path(f).stem: Path(f).name for f in image_files(folder)}
    train_files = index_by_basename(f'{BASE}/images/train')
    val_files = index_by_basename(f'{BASE}/images/val')
    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        if ann['category_id'] not in old_to_new:
            continue
        if 'segmentation' not in ann or len(ann['segmentation']) == 0:
            continue
        if isinstance(ann['segmentation'], dict):
            continue
        annotations_by_image[ann['image_id']].append(ann)

    def build_coco_subset(files_idx):
        images_out, annotations_out = ([], [])
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
    import shutil
    os.makedirs(f'{BASE}/annotations_coco', exist_ok=True)
    shutil.copy('coco_tmp/train.json', f'{BASE}/annotations_coco/train.json')
    shutil.copy('coco_tmp/val.json', f'{BASE}/annotations_coco/val.json')
    print('Copied to Drive.')

    def convert_polygon(segmentation, img_w, img_h):
        return polygon_to_yolo(segmentation, img_w, img_h)

    def write_yolo_labels(coco_data, images_dir, labels_dir):
        reset_labels(labels_dir)
        images_by_id_local = {img['id']: img for img in coco_data['images']}
        anns_by_image = defaultdict(list)
        for ann in coco_data['annotations']:
            anns_by_image[ann['image_id']].append(ann)
        for img_id, img_info in images_by_id_local.items():
            base_name = os.path.splitext(img_info['file_name'])[0]
            lines = []
            for ann in anns_by_image.get(img_id, []):
                poly_str = convert_polygon(ann['segmentation'], img_info['width'], img_info['height'])
                lines.append(f"{ann['category_id']} {poly_str}")
            with open(os.path.join(labels_dir, f'{base_name}.txt'), 'w') as f:
                f.write('\n'.join(lines))
    write_yolo_labels(train_coco, f'{BASE}/images/train', 'labels_tmp/train')
    write_yolo_labels(val_coco, f'{BASE}/images/val', 'labels_tmp/val')
    import shutil
    reset_labels(f'{BASE}/labels/train')
    reset_labels(f'{BASE}/labels/val')
    shutil.copytree('labels_tmp/train', f'{BASE}/labels/train', dirs_exist_ok=True)
    shutil.copytree('labels_tmp/val', f'{BASE}/labels/val', dirs_exist_ok=True)
    print(f"Labels YOLO: {len(os.listdir(f'{BASE}/labels/train'))} train, {len(os.listdir(f'{BASE}/labels/val'))} val")
    yaml_content = f'path: {BASE}\ntrain: images/train\nval: images/val\n\nnames:\n'
    for idx, name in enumerate(CLASS_NAMES):
        yaml_content += f'  {idx}: {name}\n'
    with open(f'{BASE}/data.yaml', 'w') as f:
        f.write(yaml_content)
    print('data.yaml written.')
    validate_training_data(BASE)
    from ultralytics import YOLO
    import time
    start = time.perf_counter()
    yolo_model = YOLO('yolov8n-seg.pt')
    yolo_model.train(seed=42, data=f'{BASE}/data.yaml', epochs=60, imgsz=640, batch=8, workers=0, project=f'{BASE}/runs', name='unified_yolo', patience=15)
    yolo_training_time = time.perf_counter() - start
    print(f'YOLO — training completed in {yolo_training_time / 60:.1f} minutes')
    import numpy as np
    from PIL import Image
    from pycocotools.coco import COCO
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms.functional as TF
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    import os
    BASE = runtime_base()
    validate_splits(BASE)

    class FashionpediaCocoDataset(Dataset):

        def __init__(self, images_dir, ann_file):
            self.images_dir = images_dir
            self.coco = COCO(ann_file)
            self.ids = sorted(self.coco.imgs.keys())

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            return coco_item(self.coco, self.ids[idx], self.images_dir)

    def collate_fn(batch):
        return tuple(zip(*batch))
    validate_training_data(BASE)
    train_dataset = FashionpediaCocoDataset(f'{BASE}/images/train', f'{BASE}/annotations_coco/train.json')
    val_dataset = FashionpediaCocoDataset(f'{BASE}/images/val', f'{BASE}/annotations_coco/val.json')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)
    NUM_CLASSES = len(CLASS_NAMES) + 1

    def build_maskrcnn(num_classes=NUM_CLASSES):
        model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
        return model
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    maskrcnn_model = build_maskrcnn().to(device)
    print(f'Train: {len(train_dataset)} images, Val: {len(val_dataset)} images. Device: {device}')

    def train_one_epoch_maskrcnn(model, optimizer, data_loader, device, epoch, print_every=20):
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
        print(f'Epoch {epoch} — mean loss: {total_loss / max(1, len(data_loader)):.4f}')
    params = [p for p in maskrcnn_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    NUM_EPOCHS_MASKRCNN = 10
    os.makedirs(f'{BASE}/runs_maskrcnn', exist_ok=True)
    start = time.perf_counter()
    for epoch in range(NUM_EPOCHS_MASKRCNN):
        train_one_epoch_maskrcnn(maskrcnn_model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        torch.save(maskrcnn_model.state_dict(), f'{BASE}/runs_maskrcnn/unified_last.pt')
    maskrcnn_training_time = time.perf_counter() - start
    print(f'Mask R-CNN — training completed in {maskrcnn_training_time / 60:.1f} minutes')
    import glob
    yolo_best = YOLO(str(yolo_model.trainer.best))
    yolo_metrics = yolo_best.val(data=f'{BASE}/data.yaml')
    val_images_sample = image_files(f'{BASE}/images/val')[:30]
    yolo_inference_ms = benchmark(lambda p: yolo_predict(yolo_best, p), val_images_sample)
    yolo_map = evaluate_coco(yolo_best, f'{BASE}/annotations_coco/val.json', f'{BASE}/images/val', 'yolo', device)
    yolo_results = {'mask_mAP50-95': yolo_map['mask_mAP50-95'], 'mask_mAP50': yolo_map['mask_mAP50'], 'box_mAP50-95': float(yolo_metrics.box.map), 'training_time_min': yolo_training_time / 60, 'avg_inference_ms': yolo_inference_ms}
    print(yolo_results)
    from pycocotools.coco import COCO as COCOgt
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as maskUtils
    maskrcnn_map = evaluate_coco(maskrcnn_model, f'{BASE}/annotations_coco/val.json', f'{BASE}/images/val', 'maskrcnn', device)
    maskrcnn_inference_ms = benchmark(lambda p: maskrcnn_predict(maskrcnn_model, p, device), val_images_sample)
    maskrcnn_results = {'mask_mAP50-95': maskrcnn_map['mask_mAP50-95'], 'mask_mAP50': maskrcnn_map['mask_mAP50'], 'training_time_min': maskrcnn_training_time / 60, 'avg_inference_ms': maskrcnn_inference_ms}
    print(maskrcnn_results)
    print(f"{'Model':<15} {'mask mAP50-95':<15} {'mask mAP50':<12} {'training (min)':<16} {'processing (ms)'}")
    print('-' * 75)
    print(f"{'YOLOv8n-seg':<15} {yolo_results['mask_mAP50-95']:<15.3f} {yolo_results['mask_mAP50']:<12.3f} {yolo_results['training_time_min']:<16.1f} {yolo_results['avg_inference_ms']:.1f}")
    print(f"{'Mask R-CNN':<15} {maskrcnn_results['mask_mAP50-95']:<15.3f} {maskrcnn_results['mask_mAP50']:<12.3f} {maskrcnn_results['training_time_min']:<16.1f} {maskrcnn_results['avg_inference_ms']:.1f}")
    import json as _json
    with open(f'{BASE}/comparison_results.json', 'w') as f:
        _json.dump({'yolov8n-seg': yolo_results, 'maskrcnn': maskrcnn_results}, f, indent=2)
    print(f'\nResults saved to: {BASE}/comparison_results.json')
    import matplotlib.pyplot as plt
    import numpy as np
    models = ['YOLOv8n-seg', 'Mask R-CNN']
    mask_map_50_95 = [yolo_results['mask_mAP50-95'], maskrcnn_results['mask_mAP50-95']]
    mask_map_50 = [yolo_results['mask_mAP50'], maskrcnn_results['mask_mAP50']]
    training_min = [yolo_results['training_time_min'], maskrcnn_results['training_time_min']]
    inference_ms = [yolo_results['avg_inference_ms'], maskrcnn_results['avg_inference_ms']]
    colors = ['#4C72B0', '#DD8452']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('YOLOv8n-seg vs. Mask R-CNN comparison', fontsize=14, fontweight='bold')
    data = [('mask mAP50-95', mask_map_50_95, axes[0, 0]), ('mask mAP50', mask_map_50, axes[0, 1]), ('Training time (min)', training_min, axes[1, 0]), ('Processing time (ms)', inference_ms, axes[1, 1])]
    for title, values, ax in data:
        bars = ax.bar(models, values, color=colors, width=0.5)
        ax.set_title(title, fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(values) * 1.25)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.03, f'{val:.3f}' if val < 1 else f'{val:.1f}', ha='center', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{BASE}/comparatie_modele.png', dpi=200, bbox_inches='tight')
    print('Saved.')


if __name__ == "__main__":
    main()
