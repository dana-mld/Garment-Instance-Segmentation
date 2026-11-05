import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import pipeline_utils as u


class PipelineTests(unittest.TestCase):
    def test_imports_do_not_train(self):
        for name in ('garment_segmentation_maskrcnn', 'yolo_wardrobe_segmentation_pipeline', 'yolo_vs_maskrcnn_unified', 'wardrobe_segmentation_pipeline_yolo_compare'):
            module = importlib.import_module(name)
            self.assertTrue(callable(module.main))

    def test_polygon_selection_and_validation(self):
        large = [0, 0, 10, 0, 10, 10, 0, 10]
        many_points_small = [0, 0, 1, 0, 2, 0, 2, 1, 2, 2, 1, 2, 0, 2, 0, 1]
        values = list(map(float, u.polygon_to_yolo([many_points_small, large], 10, 10).split()))
        self.assertEqual(values, [0, 0, 1, 0, 1, 1, 0, 1])
        for invalid in ({'counts': 'x'}, [], [[0, 1, 2]], [[0, 0, 0, 0, 0, 0]]):
            with self.assertRaises(ValueError):
                u.polygon_to_yolo(invalid, 10, 10)

    def test_boxes(self):
        self.assertEqual(u.clip_box([-4.2, -1, 12.1, 50], 10, 20), (0, 0, 10, 20))

    def test_empty_exports_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / 'input.jpg'
            image.write_bytes(b'input')
            calls = []
            def extract(path, model, output, threshold):
                calls.append(path)
                return []
            args = ([str(image)], object(), extract, str(root / 'out'), str(root / 'exports'), 0.4, None, True, 'test')
            metadata = Path(u.export_batch(*args))
            u.export_batch(*args)
            self.assertEqual(len(calls), 1)
            data = json.loads(metadata.read_text())
            self.assertEqual(data['total_source_images'], 1)
            self.assertEqual(data['total_garments'], 0)
            changed = list(args)
            changed[5] = 0.8
            with self.assertRaises(ValueError):
                u.export_batch(*changed)
            empty = list(args)
            empty[0] = []
            empty[3] = str(root / 'empty')
            self.assertTrue(Path(u.export_batch(*empty)).is_file())

    def test_split_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ('train', 'val'):
                p = root / 'images' / split
                p.mkdir(parents=True)
                (p / 'same.jpg').write_bytes(b'x')
            with self.assertRaises(ValueError):
                u.validate_splits(root)

    def test_coco_empty_and_filtered_targets(self):
        class FakeCoco:
            imgs = {1: {'file_name': 'image.png'}}
            anns = []
            def getAnnIds(self, imgIds):
                return list(range(len(self.anns)))
            def loadAnns(self, ids):
                return [self.anns[i] for i in ids]
            def annToMask(self, ann):
                return np.ones((4, 5), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            Image.new('RGB', (5, 4)).save(Path(folder) / 'image.png')
            coco = FakeCoco()
            _, target = u.coco_item(coco, 1, folder)
            self.assertEqual(tuple(target['boxes'].shape), (0, 4))
            self.assertEqual(tuple(target['masks'].shape), (0, 4, 5))
            coco.anns = [{'bbox': [0, 0, 0, 1], 'category_id': 0}, {'bbox': [-1, -1, 8, 8], 'category_id': 2}]
            _, target = u.coco_item(coco, 1, folder)
            self.assertEqual(target['boxes'].tolist(), [[0, 0, 5, 4]])
            self.assertEqual(target['labels'].tolist(), [3])
            for key in ('boxes', 'labels', 'masks', 'area', 'iscrowd'):
                self.assertEqual(len(target[key]), 1)


if __name__ == '__main__':
    unittest.main()
