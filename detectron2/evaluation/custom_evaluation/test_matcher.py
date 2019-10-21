# coding: utf-8
import unittest

import matcher
import torch

class TestCollator(unittest.TestCase):
    def test_single_object(self):
        instances = [
            {"bbox": [0, 0, 2, 2], "image_id": 42362},
        ]

        bbox, ids, image_ids = matcher.collate_bbox(instances)
        target_bbox = torch.tensor([[[0, 0, 2, 2]]], dtype=torch.float)
        target_ids = torch.tensor([[0]])
        target_image_ids = (42362,)


        self.assertEqual(bbox.tolist(), target_bbox.tolist())
        self.assertEqual(ids.tolist(), target_ids.tolist())
        self.assertEqual(image_ids, target_image_ids)


    def test_multi_object_single_image(self):
        instances = [
            {"bbox": [0, 0, 2, 2], "image_id": 1},
            {"bbox": [0, 0, 2, 2], "image_id": 1},
            {"bbox": [0, 0, 2, 2], "image_id": 1},
            {"bbox": [0, 0, 2, 2], "image_id": 1},
        ]

        bbox, ids, image_ids = matcher.collate_bbox(instances)
        target_bbox = torch.tensor([
            [[0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2]],
        ], dtype=torch.float)
        target_ids = torch.tensor([[0, 1, 2, 3]])
        target_image_ids = (1,)


        self.assertEqual(bbox.tolist(), target_bbox.tolist())
        self.assertEqual(ids.tolist(), target_ids.tolist())
        self.assertEqual(image_ids, target_image_ids)


    def test_multi_image_equal_num_instances(self):
        instances = [
            {"bbox": [0, 0, 2, 2], "image_id": 1},
            {"bbox": [4, 4, 7, 7], "image_id": 1},
            {"bbox": [0, 0, 2, 2], "image_id": 2},
            {"bbox": [0, 0, 5, 5], "image_id": 2},
        ]

        bbox, ids, image_ids = matcher.collate_bbox(instances)
        target_bbox = torch.tensor([
            [[0, 0, 2, 2], [4, 4, 11, 11]],
            [[0, 0, 2, 2], [0, 0, 5, 5]],
        ], dtype=torch.float)
        target_ids = torch.tensor([[0, 1], [2, 3]])
        target_image_ids = (1, 2)


        self.assertEqual(bbox.tolist(), target_bbox.tolist())
        self.assertEqual(ids.tolist(), target_ids.tolist())
        self.assertEqual(image_ids, target_image_ids)

    def test_multi_image_unequal_num_instances(self):
        instances = [
            {"bbox": [0, 0, 2, 2], "image_id": 1},
            {"bbox": [0, 0, 2, 2], "image_id": 2},
            {"bbox": [0, 0, 5, 5], "image_id": 69},
            {"bbox": [4, 4, 7, 7], "image_id": 69},
        ]

        bbox, ids, image_ids = matcher.collate_bbox(instances)
        target_bbox = torch.tensor([
            [[0, 0, 2, 2], [0, 0, 0, 0]],
            [[0, 0, 2, 2], [0, 0, 0, 0]],
            [[0, 0, 5, 5], [4, 4, 11, 11]],
        ], dtype=torch.float)
        target_ids = torch.tensor([[0, -1], [1, -1], [2, 3]])
        target_image_ids = (1, 2, 69)


        self.assertEqual(bbox.tolist(), target_bbox.tolist())
        self.assertEqual(ids.tolist(), target_ids.tolist())
        self.assertEqual(image_ids, target_image_ids)


    def test_image_id_ordering(self):
        instances = [
            {"bbox": [0, 0, 5, 5], "image_id": 3},
            {"bbox": [0, 0, 2, 2], "image_id": 1},
            {"bbox": [0, 0, 2, 2], "image_id": 2},
            {"bbox": [4, 4, 7, 7], "image_id": 3},
        ]

        bbox, ids, image_ids = matcher.collate_bbox(instances)
        target_bbox = torch.tensor([
            [[0, 0, 2, 2], [0, 0, 0, 0]],
            [[0, 0, 2, 2], [0, 0, 0, 0]],
            [[0, 0, 5, 5], [4, 4, 11, 11]],
        ], dtype=torch.float)
        target_ids = torch.tensor([[1, -1], [2, -1], [0, 3]])
        target_image_ids = (1, 2, 3)


        self.assertEqual(bbox.tolist(), target_bbox.tolist())
        self.assertEqual(ids.tolist(), target_ids.tolist())
        self.assertEqual(image_ids, target_image_ids)


class TestIOU(unittest.TestCase):
    def test_half_overlap(self):
        bbox = torch.tensor([[[0, 0, 2, 2], [1, 0, 3, 2]]])
        iou = matcher.compute_iou(bbox, bbox)

        # Remember union has width of 3 and indersection has width of 1
        target_iou = torch.tensor([[[1, 1/3], [1/3, 1]]])
        self.assertEqual(iou.tolist(), target_iou.tolist())


    def test_full_overlap(self):
        bbox = torch.tensor([[[0, 0, 2, 2], [0, 0, 2, 2]]])
        iou = matcher.compute_iou(bbox, bbox)

        target_iou = torch.tensor([[[1, 1], [1, 1]]], dtype=torch.float)
        self.assertEqual(iou.tolist(), target_iou.tolist())


    def test_no_overlap(self):
        bbox = torch.tensor([[[0, 0, 2, 2], [0, 2, 2, 4]]])
        iou = matcher.compute_iou(bbox, bbox)

        target_iou = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float)
        self.assertEqual(iou.tolist(), target_iou.tolist())


class TestMatcher(unittest.TestCase):
    def test_perfect_pred(self):
        gt_instances = [
            {"bbox": [0, 0, 2, 2], "image_id": 3, "id": 9346},
            {"bbox": [0, 0, 2, 2], "image_id": 1, "id": 777},
            {"bbox": [0, 0, 2, 2], "image_id": 2, "id": 22222},
            {"bbox": [0, 1, 2, 2], "image_id": 3, "id": 1},
        ]

        pred_instances = [
            {"bbox": [0, 0, 2, 2], "image_id": 3, "id": 432},
            {"bbox": [0, 0, 2, 2], "image_id": 1, "id": 1111},
            {"bbox": [0, 0, 2, 2], "image_id": 2, "id": 9191},
            {"bbox": [0, 1, 2, 2], "image_id": 3, "id": 1112},
        ]

        matcher.compute_matches_inplace(gt_instances, pred_instances)

        self.assertEqual(
            gt_instances[0]["matched_preds"][0],
            (432, 1.)
        )

        self.assertEqual(
            pred_instances[0]["matched_gts"][0],
            (9346, 1.)
        )

if __name__ == "__main__":
    unittest.main()

