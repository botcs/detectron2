"""
Matcher:

- for every instance in ground truth list all overlapping prediction
- for every instance in predictions list all overlapping ground truth

- for every instance in ground truth list all overlapping ground truth
- for every instance in predictions list all overlapping prediction

- this is universal: no shortcuts
"""

import json
import logging
import math
from collections import Counter, defaultdict
import torch
from tqdm import tqdm


###############
# MATCHER UTILS
###############
def already_matched(gt_instances, pred_instances):
    # Sanity check for cases when matching is invoked multiple times
    any_matched = any(
        "matched_preds" in gt_inst for gt_inst in gt_instances
    ) or any(
        "matched_gts" in pred_inst for pred_inst in pred_instances
    )

    all_matched = all(
        "matched_preds" in gt_inst for gt_inst in gt_instances
    ) and all(
        "matched_gts" in pred_inst for pred_inst in pred_instances
    )
    assert any_matched == all_matched
    return all_matched


def generate_id_map(instances):
    """
    Args:
        instances: list of instances
    Returns:
        id_map: dict, [image_id, inst_id] -> list idx
    """
    id_map = {(inst["image_id"], inst["id"]): inst for inst in instances}
    return id_map


def collate_bbox(instances):
    """
    Args:
        instances: list of instance dictionaries, that has "image_id" and
            "bbox" fields

    Returns:

        bbox_tensor: Torch tensor of shape [I, J, BBOX], where:
            I: number of unique "image_id" fields in instances
            J: max number of instances per image
            BBOX: 4 ~ xmin, ymin, xmax, ymax for bbox coordinates
            Images with less instances than dim2 are padded with 0

        image_ids: list of image_id values represinting each corresponding row
            in bbox_tensor
    """

    assert isinstance(instances, list)
    def xywh_to_xyxy(inst_bbox):
        xmin, ymin, width, height = inst_bbox
        xmax = xmin + width
        ymax = ymin + height
        return xmin, ymin, xmax, ymax


    inst_id_per_img = {}
    bbox_per_img = {}
    for inst_list_id, inst in enumerate(instances):
        image_id = inst["image_id"]


        inst_bbox = xywh_to_xyxy(inst["bbox"])
        if image_id not in bbox_per_img:
            bbox_per_img[image_id] = []
        bbox_per_img[image_id].append(inst_bbox)


        if image_id not in inst_id_per_img:
            inst_id_per_img[image_id] = []
        inst_id_per_img[image_id].append(inst_list_id)

    # Assumption that "instances" are NOT sorted!
    # Could be optimized by sorting instances by 1) "image_id" and 2) "id"
    image_ids, bbox_list = zip(*sorted(bbox_per_img.items(), key=lambda x: x[0]))
    image_ids, id_list = zip(*sorted(inst_id_per_img.items(), key=lambda x: x[0]))
    inst_per_img = list(len(x) for x in bbox_list)

    bbox_tensor = torch.zeros(len(image_ids), max(inst_per_img), 4)
    id_tensor = torch.ones(len(image_ids), max(inst_per_img), dtype=torch.long) * -1
    for i in range(len(image_ids)):
        for j in range(inst_per_img[i]):
            bbox_tensor[i][j] = torch.tensor(bbox_list[i][j])
            id_tensor[i][j] = id_list[i][j]

    return bbox_tensor, id_tensor, image_ids


def compute_iou(bbox1, bbox2):
    """
    Args:
        bbox1: Torch tensor of shape [I, J, BBOX], where:
            I: number of unique "image_id" fields in instances
            J: max number of instances per image
            BBOX: 4 ~ xmin, ymin, xmax, ymax for bbox coordinates
            Images with less instances than dim2 are padded with 0
        bbox1: Torch tensor of shape [I, K, BBOX], where:
            I: number of unique "image_id" fields in instances
            K: max number of instances per image
            BBOX: 4 ~ xmin, ymin, xmax, ymax for bbox coordinates
            Images with less instances than dim2 are padded with 0
    Returns:
        iou: Torch tensor of shape [I, J, K], where the (i, j, k) element
            represents the IoU score of the j_th instance from bbox1 with
            the k_th instance from bbox2 on the i_th image.

    Note:
    Each row represents a different image, and each column represents the
    instances on that image, the third column in bbox1 and bbox2 only
    concatenates the (xyxy) coords. Thanks to this IoU scores can be computed
    by linear algebraic operations, that allows parallelism.

    The resulting tensor has the scores of the pairwise comparison, therefore
    it has the same number of rows (#images), and J rows and K columns.
    """

    bbox1 = bbox1.to(torch.float)
    bbox2 = bbox2.to(torch.float)

    # reshape for parallel computation
    A = bbox1[:, :, None, :]
    B = bbox2[:, None, :, :]

    # get min coordinates
    pairwise_min = torch.min(A, B)
    xmax_intersection = pairwise_min[:, :, :, 2]
    ymax_intersection = pairwise_min[:, :, :, 3]

    # get max coordinates
    pairwise_max = torch.max(A, B)
    xmin_intersection = pairwise_max[:, :, :, 0]
    ymin_intersection = pairwise_max[:, :, :, 1]

    # compute intersection
    width_intersection = xmax_intersection - xmin_intersection
    height_intersection = ymax_intersection - ymin_intersection
    area_intersection = width_intersection * height_intersection

    # compute union
    xmin_A, ymin_A, xmax_A, ymax_A = bbox1.split(1, dim=-1)
    xmin_B, ymin_B, xmax_B, ymax_B = bbox2.split(1, dim=-1)
    area_A = (xmax_A - xmin_A) * (ymax_A - ymin_A)
    area_B = (xmax_B - xmin_B) * (ymax_B - ymin_B)

    # reshape for parallel computation
    area_A = area_A[:, :, None, 0]
    area_B = area_B[:, None, :, 0]

    area_union =  area_A + area_B - area_intersection

    iou = area_intersection / area_union
    return iou


def find_matching(iou, A_ids, B_ids):
    """
    Matches overlapping objects between A and B using the precomputed IOU

    Args:
        iou: tensor [dim1, dim2, dim3], where
            dim1: number of unique images
            dim2: max number of isntance per image in dataset A
            dim2: max number of isntance per image in dataset B

        A_ids: tensor [dim1, dim2] referencing the list index in A_instances
        B_ids: tensor [dim1, dim3] referencing the list index in B_instances

    Returns:
        matched_tensor_indices: dict where
            key: index of reference bbox in A
                i: image dim
                j: ref. instance per image dim
            value: list of indices of matched bbox in B
                i: image dim
                k: compared instance per image dim

    """

    num_images = iou.shape[0]
    max_inst_per_img_A = iou.shape[1]
    max_inst_per_img_B = iou.shape[2]

    iou_mask = iou > 0

    matched_tensor_indices = defaultdict(list)
    for i in tqdm(range(num_images)):
        for j in range(max_inst_per_img_A):
            # No more reference instances in this row
            if A_ids[i, j] == -1:
                break

            for k in range(max_inst_per_img_B):
                # No more instances to compare against in this column
                if B_ids[i, k] == -1:
                    break

                if iou_mask[i, j, k]:
                    matched_tensor_indices[i, j].append((i, k))
    return matched_tensor_indices


#############################
# MAIN MATCHER IMPLEMENTATION
#############################
def compute_matches_inplace(gt_instances, pred_instances):
    for gt_inst in gt_instances:
        gt_inst["matched_preds"] = []

    for pred_inst in pred_instances:
        pred_inst["matched_gts"] = []

    logging.info("Reshaping BBoxes into tensor")
    gt_bbox, gt_ids, gt_image_ids = collate_bbox(gt_instances)
    pred_bbox, pred_ids, pred_image_ids = collate_bbox(pred_instances)
    image_ids = gt_image_ids

    logging.info("Computing IOU between Preds and GTs")
    iou = compute_iou(gt_bbox, pred_bbox)

    logging.info("Matching instances")
    matched_indices = find_matching(iou, gt_ids, pred_ids)

    logging.info("Registering matched instance IDs")
    for (i, j), matches in tqdm(matched_indices.items()):
        reference_inst = gt_instances[gt_ids[i, j]]
        reference_inst_id = reference_inst["id"]
        for (i, k) in matches:
            matched_inst = pred_instances[pred_ids[i, k]]
            matched_inst_id = matched_inst["id"]

            reference_inst["matched_preds"].append(
                (matched_inst_id, iou[i, j, k].item())
            )
            matched_inst["matched_gts"].append(
                (reference_inst_id, iou[i, j, k].item())
            )




def batch_compute_matches_inplace(gt_instances, pred_instances, batch_size=10000):
    """
    To avoid OOM errors, this function computes the matches in smaller chunks

    Args:
        batch_size: an upper bound that determines the max number of instances
            matched at once
    """

    # Group by image_id
    gt_instances_per_img = defaultdict(list)
    for gt_inst in gt_instances:
        gt_instances_per_img[gt_inst["image_id"]].append(gt_inst)

    pred_instances_per_img = defaultdict(list)
    for pred_inst in pred_instances:
        pred_instances_per_img[pred_inst["image_id"]].append(pred_inst)


    # Feasibility check: more instance on a single image than the max capacity
    assert max(len(x) for x in gt_instances_per_img.values()) < batch_size
    assert max(len(x) for x in pred_instances_per_img.values()) < batch_size
    unique_image_ids = set().union(
        gt_instances_per_img.keys(),
        pred_instances_per_img.keys(),
    )

    num_instances = len(gt_instances) + len(pred_instances)
    num_batches = int(math.ceil(num_instances / batch_size))
    batches_done = 1

    gt_instances_batch = []
    pred_instances_batch = []
    for image_id in unique_image_ids:
        iter_size = 0
        if image_id in gt_instances_per_img:
            iter_size += len(gt_instances_per_img[image_id])
        if image_id in pred_instances_per_img:
            iter_size += len(pred_instances_per_img[image_id])

        # If adding instances from this image would spill over the batch size
        # then do the matching and empty the batch container
        #
        # otherwise just extend the two batch list
        curr_size = len(gt_instances_batch) + len(pred_instances_batch)
        if curr_size + iter_size > batch_size:
            logging.info(f"BATCH {batches_done} / {num_batches}")
            compute_matches_inplace(gt_instances_batch, pred_instances_batch)
            batches_done += 1
            gt_instances_batch = gt_instances_per_img[image_id]
            pred_instances_batch = pred_instances_per_img[image_id]
        else:
            gt_instances_batch.extend(gt_instances_per_img[image_id])
            pred_instances_batch.extend(pred_instances_per_img[image_id])
    if len(gt_instances_batch) > 0 or len(pred_instances_batch) > 0:
            logging.info(f"BATCH {batches_done} / {num_batches}")
            compute_matches_inplace(gt_instances_batch, pred_instances_batch)


def has_inconsistent_matches(gt_instances, pred_instances):
    """
    Returns False if ANY of the references are invalid
    """

    # checking ground truth
    pred_id_map = generate_id_map(pred_instances)
    for gt_inst in gt_instances:
        image_id = gt_inst["image_id"]
        matched_preds = gt_inst["matched_preds"]
        category_id = gt_inst["category_id"]

        for pred_match_id, _ in matched_preds:
            if (image_id, pred_match_id) not in pred_id_map:
                return True

    # checking predictions
    gt_id_map = generate_id_map(gt_instances)
    for pred_inst in pred_instances:
        image_id = pred_inst["image_id"]
        matched_gts = pred_inst["matched_gts"]
        category_id = pred_inst["category_id"]

        for gt_match_id, _ in matched_gts:
            if (image_id, gt_match_id) not in gt_id_map:
                return True

    return False


def remove_inconsistent_matches_inplace(gt_instances, pred_instances, mode="both"):
    """
    3 modes:
        gt: cleans only the ground truth instances of pred instances that are not
            present in the pred_instanes
        pred: cleans only the predicted instances of gt instances that are not
            present in the gt_instanes
        both: cleans both pred and gt

    """

    # cleaning ground truth
    if mode in ("gt", "both"):
        pred_id_map = generate_id_map(pred_instances)
        for gt_inst in gt_instances:
            image_id = gt_inst["image_id"]
            matched_preds = gt_inst["matched_preds"]
            category_id = gt_inst["category_id"]
            consistent_pred_matches = []

            for pred_match_id, pred_match_id in matched_preds:
                if (image_id, pred_match_id) in pred_id_map:
                    consistent_pred_matches.append(
                        (pred_match_id, pred_match_iou)
                    )

            gt_inst["matched_preds"] = consistent_pred_matches

    # cleaning predictions
    if mode in ("pred", "both"):
        gt_id_map = generate_id_map(gt_instances)
        for pred_inst in pred_instances:
            image_id = pred_inst["image_id"]
            matched_gts = pred_inst["matched_gts"]
            category_id = pred_inst["category_id"]
            consistent_gt_matches = []

            for gt_match_id, gt_match_iou in matched_gts:
                if (image_id, gt_match_id) in gt_id_map:
                    consistent_gt_matches.append(
                        (gt_match_id, gt_match_iou)
                    )

            pred_inst["matched_gts"] = consistent_gt_matches
