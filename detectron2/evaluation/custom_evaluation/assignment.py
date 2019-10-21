"""
This module implements the assignment logic that results in a 1-to-1 mapping
between objects.

Having multiple candidate matches we would like to assign the "best" match.
Different definitions of "best" yield different evaluation metrics.
"""
import logging

import filters
import matcher


##################
# ASSIGNMENT UTILS
##################
def already_assigned(gt_instances, pred_instances):
    # Sanity check for cases when assignment is called multiple times
    any_assigned = any(
        "assigned_pred" in gt_inst for gt_inst in gt_instances
    ) or any(
        "assigned_gt" in pred_inst for pred_inst in pred_instances
    )

    all_assigned = all(
        "assigned_pred" in gt_inst for gt_inst in gt_instances
    ) and all(
        "assigned_gt" in pred_inst for pred_inst in pred_instances
    )

    assert any_assigned == all_assigned
    return all_assigned


def remove_assignments_inplace(gt_instances, pred_instances):
    for gt_inst in gt_instances:
        if "assigned_pred" in gt_inst:
            del gt_inst["assigned_pred"]
    for pred_inst in pred_instances:
        if "assigned_gt" in pred_inst:
            del pred_inst["assigned_gt"]


#################################
# MAIN ASSIGNMENT IMPLEMENTATIONS
#################################
def pred_first_conf_rank_assign_inplace(
    gt_instances, pred_instances
):
    """
    1) go through predictions first, sorted by confidence score.
    2) ground truth having the largest iou gets assigned to the instance

    remarks:
        if a prediction of higher confidence gets assigned to a GT
        that GT won't be listed for a better candidate (higher iou) with lower
        score

        doing a GT first assignment should be considered
    """
    assert not already_assigned(gt_instances, pred_instances)

    logging.debug("Sorting predictions by confidence")
    pred_instances.sort(key=lambda x: x["score"])

    logging.debug("Finding non-assigned GT with largest IoU")
    gt_id_map = matcher.generate_id_map(gt_instances)
    pred_id_map = matcher.generate_id_map(pred_instances)
    for gt_inst in gt_instances:
        gt_inst["assigned_pred"] = None
    for pred_inst in pred_instances:
        pred_inst["assigned_gt"] = None

    for pred_inst in pred_instances:
        image_id = pred_inst["image_id"]
        matched_gts = pred_inst["matched_gts"]
        matched_gts.sort(key=lambda x: x[1], reverse=True)

        for gt_match_id, _ in matched_gts:
            gt_inst = gt_id_map[image_id, gt_match_id]
            if gt_inst["assigned_pred"] is None:
                gt_inst["assigned_pred"] = pred_inst["id"]
                pred_inst["assigned_gt"] = gt_inst["id"]
                break
