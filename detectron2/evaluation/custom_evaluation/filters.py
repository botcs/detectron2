"""
This module implements filters that reduces the matched / assigned instances
by some pre-defined rules, such as sizes, matching categories, etc.

Filters:
- takes matched/assigned instance lists as input
- outputs filtered instances in dict / json format
- reusable, can be pipelined, parallelized

"""
import matcher
import assignment
import logging

def filter_size_range_inplace(gt_instances, pred_instances, min_area, max_area):
    """
    1) remove instances outside the area range (inclusive)
    2) clean up references in matches
    """
    logging.debug(f"Filter instances by area in range [{min_area}:{max_area}]")
    def area(inst):
        xmin, ymin, xmax, ymax = inst["bbox"]
        w = xmax - xmin
        h = ymax - ymin
        # sanity check just for fun
        assert w > 0 and h > 0
        return w * h

    gt_instances[:] = [
        gt_inst
        for gt_inst in gt_instances
        if min_area <= area(gt_inst) < max_area
    ]
    pred_instances[:] = [
        pred_inst
        for pred_inst in pred_instances
        if min_area <= area(pred_inst) < max_area
    ]

    matcher.remove_inconsistent_matches_inplace(gt_instances, pred_instances)


def filter_size_range_by_gt_reference_inplace(
    gt_instances, pred_instances, min_area, max_area
):
    """
    1) remove pred whose assigned gt's (if any) size is out of the area range
    2) remove gt and unassigned pred outside the area range
    3) clean up references in matches
    """
    assert assignment.already_assigned(gt_instances, pred_instances)
    def area(inst):
        xmin, ymin, xmax, ymax = inst["bbox"]
        w = xmax - xmin
        h = ymax - ymin
        # sanity check just for fun
        assert w > 0 and h > 0
        return w * h

    logging.debug(
        "Filter predictions by assigned GT's area "
        f"in range [{min_area}:{max_area}]"
    )
    gt_id_map = matcher.generate_id_map(gt_instances)
    filtered_pred_instances = []
    for pred_inst in pred_instances:
        image_id = pred_inst["image_id"]
        assigned_gt = pred_inst["assigned_gt"]
        if assigned_gt is not None:
            assigned_gt_id, assigned_gt_iou = assigned_gt
            assigned_gt_inst = gt_id_map[image_id, assigned_gt_id]
            if min_area <= area(gt_inst) <= max_area:
                filtered_pred_instances.append(pred_inst)
        else:
            if min_area <= area(pred_inst) <= max_area:
                filtered_pred_instances.append(pred_inst)
    pred_instances[:] = filtered_pred_instances


    logging.debug(
        f"Filter GT instances by area in range [{min_area}:{max_area}] "
    )
    gt_instances[:] = [
        gt_inst
        for gt_inst in gt_instances
        if min_area <= area(gt_inst) < max_area
    ]

    matcher.remove_inconsistent_matches_inplace(gt_instances, pred_instances)


def filter_conf_range_inplace(gt_instances, pred_instances, min_conf, max_conf):
    """
    1) remove predictions out of given confidence range (inclusive)
    2) clean up references in gt's matches
    """
    logging.debug(
        f"Filter predictions by conf in range [{min_conf}:{max_conf}] "
    )
    pred_instances[:] = [
        inst for inst in pred_instances if min_conf <= inst["score"] <= max_conf
    ]
    matcher.remove_inconsistent_matches_inplace(gt_instances, pred_instances, "gt")


def filter_iou_range_inplace(gt_instances, pred_instances, min_iou, max_iou):
    """
    remove match references out of given iou range (inclusive)
    """
    logging.debug(
        f"Filter match references by iou in range [{min_iou}:{max_iou}] "
    )
    for gt_inst in gt_instances:
        matched_preds = gt_inst["matched_preds"]
        gt_inst["matched_preds"] = [
            (pred_id, pred_iou)
            for pred_id, pred_iou in matched_preds
            if min_iou <= pred_iou <= max_iou
        ]

    for pred_inst in pred_instances:
        matched_preds = pred_inst["matched_gts"]
        pred_inst["matched_gts"] = [
            (gt_id, gt_iou)
            for gt_id, gt_iou in matched_preds
            if min_iou <= gt_iou <= max_iou
        ]


def filter_correct_class_inplace(gt_instances, pred_instances):
    """
    remove match references of incorrect classes
    """
    logging.debug("Remove match references of incorrect classes")

    # Could be done in one pass if linked lists were used,
    # but deleting from a list is O(k)
    # filtering gts
    pred_id_map = matcher.generate_id_map(pred_instances)
    for gt_inst in gt_instances:
        image_id = gt_inst["image_id"]
        matched_preds = gt_inst["matched_preds"]
        category_id = gt_inst["category_id"]
        correct_class_pred_matches = []

        for pred_match_id, pred_match_iou in matched_preds:
            pred_inst = pred_id_map[image_id, pred_match_id]
            if pred_inst["category_id"] == category_id:
                correct_class_pred_matches.append(
                    (pred_match_id, pred_match_iou)
                )

        gt_inst["matched_preds"] = correct_class_pred_matches

    # filtering preds
    gt_id_map = matcher.generate_id_map(gt_instances)
    for pred_inst in pred_instances:
        image_id = pred_inst["image_id"]
        matched_gts = pred_inst["matched_gts"]
        category_id = pred_inst["category_id"]
        correct_class_gt_matches = []

        for gt_match_id, gt_match_iou in matched_gts:
            gt_inst = gt_id_map[image_id, gt_match_id]
            if gt_inst["category_id"] == category_id:
                correct_class_gt_matches.append(
                    (gt_match_id, gt_match_iou)
                )

        pred_inst["matched_gts"] = correct_class_gt_matches
