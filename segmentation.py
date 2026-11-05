from dataclasses import dataclass
from typing import List, Optional
import cv2
import numpy as np
from ultralytics import YOLO

@dataclass
class GarmentResult:
    category: str
    confidence: float
    bbox: list
    crop_bbox: list
    image_bytes: bytes
    mask_bytes: bytes
    mask_polygon: Optional[list] = None

class GarmentSegmenter:

    def __init__(self, weights_path: str, conf_threshold: float=0.5):
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def _get_mask_polygon(self, mask: np.ndarray, epsilon: float=0.001) -> list:
        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        all_points = np.vstack([cnt for cnt in contours])
        hull = cv2.convexHull(all_points)
        epsilon_val = epsilon * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon_val, True)
        polygon = approx.reshape(-1, 2).tolist()
        return polygon

    def segment(self, image_bytes: bytes) -> List[GarmentResult]:
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError('Invalid or corrupted image: could not decode the input.')
        results = self.model(img_bgr, conf=self.conf_threshold, verbose=False, retina_masks=True)[0]
        if results.masks is None:
            return []
        img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        h, w = img_bgr.shape[:2]
        garments = []
        for mask, box, cls, conf in zip(results.masks.data, results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            mask_resized = cv2.resize(mask.cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
            mask_uint8 = (mask_resized * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            all_contour_points = np.vstack([cnt for cnt in contours])
            x, y, cw, ch = cv2.boundingRect(all_contour_points)
            padding = 5
            x2 = min(w, x + cw + padding)
            y2 = min(h, y + ch + padding)
            x = max(0, x - padding)
            y = max(0, y - padding)
            cw = x2 - x
            ch = y2 - y
            out = img_rgba.copy()
            out[:, :, 3] = mask_uint8
            cropped = out[y:y + ch, x:x + cw]
            if cropped.size == 0:
                continue
            success, png_bytes = cv2.imencode('.png', cropped)
            if not success:
                continue
            mask_cropped = mask_uint8[y:y + ch, x:x + cw]
            success_mask, mask_png = cv2.imencode('.png', mask_cropped)
            if not success_mask:
                continue
            polygon = self._get_mask_polygon(mask_resized)
            garments.append(GarmentResult(category=results.names[int(cls)], confidence=float(conf), bbox=[int(x1) for x1 in box.tolist()], crop_bbox=[x, y, cw, ch], image_bytes=png_bytes.tobytes(), mask_bytes=mask_png.tobytes(), mask_polygon=polygon))
        return garments
if __name__ == '__main__':
    import sys
    import os
    if len(sys.argv) != 2:
        sys.exit(1)
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f'File not found: {image_path}')
        sys.exit(1)
    from pathlib import Path
    segmenter = GarmentSegmenter(str(Path(__file__).resolve().parent / 'models/yolov8n_seg_best.pt'))
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    garments = segmenter.segment(image_bytes)
    if not garments:
        print('No garments detected.')
    else:
        os.makedirs('test_output', exist_ok=True)
        for i, g in enumerate(garments):
            crop_path = f'test_output/{g.category}_{i}_crop.png'
            with open(crop_path, 'wb') as f:
                f.write(g.image_bytes)
            mask_path = f'test_output/{g.category}_{i}_mask.png'
            with open(mask_path, 'wb') as f:
                f.write(g.mask_bytes)
            print(f'{g.category} (conf: {g.confidence:.2f})')
            print(f'  Crop bbox: {g.crop_bbox}')
            print(f'  Original bbox: {g.bbox}')
            print(f'  Polygon: {len(g.mask_polygon)} points')
            print(f'  -> {crop_path}, {mask_path}')
