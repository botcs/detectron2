"""
This module implements the evaluation heads
- takes matched (optionally filtered) instance lists as input
- outputs custom metrics evaluated in dict / json format
- modularity: MUST be unit tested

Code organization: if possible, orthogonal modules are implemented separately
    - eval: aims for end-to-end evaluation of a given metric
    - compute: implements a specific stage of evaluation

Usually we reduce many aligned/mis-aligned instance pairs into just a few
numbers by some logic like mAP, precision-recall-curves, class confusion matrix,
etc.
"""

import matcher
import assignment
import torch
import logging
import filters
from copy import deepcopy
from collections import Counter

from tqdm import tqdm


############
# EVAL UTILS
############
def safety_copy(gt_instances, pred_instances):
    logging.debug("Hard-copying instances before filtering")
    copied_gt_instances = deepcopy(gt_instances)
    copied_pred_instances = deepcopy(pred_instances)
    return copied_gt_instances, copied_pred_instances

def prepare_for_eval(gt_instances, pred_instances, copy=True):
    if matcher.already_matched(gt_instances, pred_instances):
        logging.info("Found cached match references, validating")
        assert not matcher.has_inconsistent_matches(gt_instances, pred_instances)
    else:
        logging.info("Matching Preds with GTs")
        matcher.compute_matches_inplace(gt_instances, pred_instances)

    if copy:
        gt_instances, pred_instances = safety_copy(gt_instances, pred_instances)

    return gt_instances, pred_instances




###########################
# MAIN EVAL IMPLEMENTATIONS
###########################
# CLASS AGNOSTIC SCORES
#######################
def compute_prec_rec(gt_instances, pred_instances):
    assignment.remove_assignments_inplace(gt_instances, pred_instances)
    assignment.pred_first_conf_rank_assign_inplace(gt_instances, pred_instances)

    tp = fp = fn = 0
    for gt_inst in gt_instances:
        if gt_inst["assigned_pred"] is not None:
            tp += 1
        else:
            fn += 1

    for pred_inst in pred_instances:
        if pred_inst["assigned_gt"] is None:
            fp += 1

    prec = tp / (tp + fp) if tp or fp else 0.
    rec = tp / (tp + fn) if tp or fn else 0.

    return prec, rec


def eval_prec_rec_at_conf_levels(
    gt_instances, pred_instances, conf_levels, iou_threshold
):
    gt_instances, pred_instances = prepare_for_eval(gt_instances, pred_instances)

    filters.filter_correct_class_inplace(gt_instances, pred_instances)
    filters.filter_iou_range_inplace(
        gt_instances, pred_instances, min_iou=iou_threshold, max_iou=1.
    )

    precs, recs = {}, {}
    for conf_threshold in sorted(conf_levels):
        logging.info(
            "Evaluating Precision / Recall "
            f"@ conf>={conf_threshold} "
            f"& IoU>={iou_threshold}"
        )
        filters.filter_conf_range_inplace(
            gt_instances, pred_instances, min_conf=conf_threshold, max_conf=1.
        )
        prec, rec = compute_prec_rec(gt_instances, pred_instances)
        precs[conf_threshold] = prec
        recs[conf_threshold] = rec

    return precs, recs


def eval_prec_rec_at_iou_levels(
    gt_instances, pred_instances, conf_threshold, iou_levels
):
    gt_instances, pred_instances = prepare_for_eval(gt_instances, pred_instances)

    filters.filter_correct_class_inplace(gt_instances, pred_instances)
    filters.filter_conf_range_inplace(
        gt_instances, pred_instances, min_conf=conf_threshold, max_conf=1.
    )

    precs, recs = {}, {}
    for iou_threshold in sorted(iou_levels):
        logging.info(
            "Evaluating Precision / Recall "
            f"@ conf>={conf_threshold} "
            f"& IoU>={iou_threshold}"
        )
        filters.filter_iou_range_inplace(
            gt_instances, pred_instances, min_iou=iou_threshold, max_iou=1.
        )
        prec, rec = compute_prec_rec(gt_instances, pred_instances)
        precs[iou_threshold] = prec
        recs[iou_threshold] = rec

    return precs, recs


def eval_prec_rec_at_iou_conf_grid(
    gt_instances, pred_instances, conf_levels, iou_levels
):
    gt_instances, pred_instances = prepare_for_eval(gt_instances, pred_instances)

    filters.filter_correct_class_inplace(gt_instances, pred_instances)
    conf_levels.sort()
    iou_levels.sort()
    precs, recs = {}, {}

    logging.info(
        f"Grid evaluation @ conf[{min(conf_levels)}:{max(conf_levels)}] "
        f"& IoU[{min(iou_levels)}:{max(iou_levels)}] "
    )
    pbar = tqdm(total=len(conf_levels)*len(iou_levels))
    for conf_threshold in conf_levels:
        curr_gt_instances, curr_pred_instances = safety_copy(
            gt_instances, pred_instances
        )
        filters.filter_conf_range_inplace(
            gt_instances, pred_instances, min_conf=conf_threshold, max_conf=1.
        )
        for iou_threshold in iou_levels:
            filters.filter_iou_range_inplace(
                curr_gt_instances,
                curr_pred_instances,
                min_iou=iou_threshold,
                max_iou=1.
            )
            prec, rec = compute_prec_rec(curr_gt_instances, curr_pred_instances)
            precs[conf_threshold, iou_threshold] = prec
            recs[conf_threshold, iou_threshold] = rec
            pbar.set_description(
                f"conf>={conf_threshold:.2f} & iou>={iou_threshold:.2f}"
            )
            pbar.update()
    pbar.close()
    return precs, recs


####################
# CLASS AWARE SCORES
####################
def compute_prec_rec_per_class(gt_instances, pred_instances):
    assignment.remove_assignments_inplace(gt_instances, pred_instances)
    assignment.pred_first_conf_rank_assign_inplace(
        gt_instances,
        pred_instances,
    )

    prec, rec = {}, {}
    TP, FP, FN = Counter(), Counter(), Counter()

    for gt_inst in gt_instances:
        category_id = gt_inst["category_id"]
        if gt_inst["assigned_pred"] is not None:
            TP[category_id] += 1
        else:
            FN[category_id] += 1

    for pred_inst in pred_instances:
        category_id = pred_inst["category_id"]
        if pred_inst["assigned_gt"] is None:
            FP[category_id] += 1

    for category_id in set().union(TP.keys(), FP.keys(), FN.keys()):
        tp = TP[category_id]
        fp = FP[category_id]
        fn = FN[category_id]
        prec[category_id] = tp / (tp + fp) if tp or fp else 0.
        rec[category_id] = tp / (tp + fn) if tp or fn else 0.

    return prec, rec


def eval_prec_rec_per_class(
    gt_instances, pred_instances, conf_threshold=.7, iou_threshold=.5
):
    gt_instances, pred_instances = prepare_for_eval(gt_instances, pred_instances)

    filters.filter_correct_class_inplace(gt_instances, pred_instances)
    filters.filter_conf_range_inplace(
        gt_instances, pred_instances, min_conf=conf_threshold, max_conf=1.
    )
    filters.filter_iou_range_inplace(
        gt_instances, pred_instances, min_iou=iou_threshold, max_iou=1.
    )

    prec, rec = compute_prec_rec_per_class(gt_instances, pred_instances)


def eval_prec_rec_per_class_at_conf_levels(
    gt_instances, pred_instances, conf_levels, iou_threshold=.5
):
    gt_instances, pred_instances = prepare_for_eval(gt_instances, pred_instances)

    filters.filter_correct_class_inplace(gt_instances, pred_instances)
    filters.filter_iou_range_inplace(
        gt_instances, pred_instances, min_iou=iou_threshold, max_iou=1.
    )

    precs, recs = {}, {}
    for conf_threshold in sorted(conf_levels):
        logging.info(
            "Evaluating Precision / Recall "
            f"@ conf>={conf_threshold} "
            f"& IoU>={iou_threshold}"
        )
        filters.filter_conf_range_inplace(
            gt_instances, pred_instances, min_conf=conf_threshold, max_conf=1.
        )
        prec, rec = compute_prec_rec_per_class(gt_instances, pred_instances)
        for category_id, prec_val in prec.items():
            precs[conf_threshold, category_id] = prec_val
        for category_id, rec_val in rec.items():
            recs[conf_threshold, category_id] = rec_val

    return precs, recs


def eval_prec_rec_per_class_at_iou_levels(
    gt_instances, pred_instances, conf_threshold, iou_levels
):
    gt_instances, pred_instances = prepare_for_eval(gt_instances, pred_instances)

    filters.filter_correct_class_inplace(gt_instances, pred_instances)
    filters.filter_conf_range_inplace(
        gt_instances, pred_instances, min_conf=conf_threshold, max_conf=1.
    )

    precs, recs = {}, {}
    for iou_threshold in sorted(iou_levels):
        logging.info(
            "Evaluating Precision / Recall "
            f"@ conf>={conf_threshold} "
            f"& IoU>={iou_threshold}"
        )
        filters.filter_iou_range_inplace(
            gt_instances, pred_instances, min_iou=iou_threshold, max_iou=1.
        )
        prec, rec = compute_prec_rec_per_class(gt_instances, pred_instances)
        for category_id, prec_val in prec.items():
            precs[conf_threshold, category_id] = prec_val
        for category_id, rec_val in rec.items():
            recs[conf_threshold, category_id] = rec_val

    return precs, recs


def eval_prec_rec_per_class_at_iou_conf_grid(
    gt_instances, pred_instances, conf_levels, iou_levels
):
    gt_instances, pred_instances = prepare_for_eval(gt_instances, pred_instances)

    filters.filter_correct_class_inplace(gt_instances, pred_instances)
    conf_levels.sort()
    iou_levels.sort()
    precs, recs = {}, {}

    logging.info(
        f"Grid evaluation @ conf[{min(conf_levels)}:{max(conf_levels)}] "
        f"& IoU[{min(iou_levels)}:{max(iou_levels)}] "
    )
    pbar = tqdm(total=len(conf_levels)*len(iou_levels))
    for conf_threshold in conf_levels:
        curr_gt_instances, curr_pred_instances = safety_copy(
            gt_instances, pred_instances
        )
        filters.filter_conf_range_inplace(
            gt_instances, pred_instances, min_conf=conf_threshold, max_conf=1.
        )
        for iou_threshold in iou_levels:
            filters.filter_iou_range_inplace(
                curr_gt_instances,
                curr_pred_instances,
                min_iou=iou_threshold,
                max_iou=1.
            )

            prec, rec = compute_prec_rec_per_class(
                curr_gt_instances, curr_pred_instances
            )
            for category_id, prec_val in prec.items():
                precs[conf_threshold, iou_threshold, category_id] = prec_val
            for category_id, rec_val in rec.items():
                recs[conf_threshold, iou_threshold, category_id] = rec_val
            pbar.set_description(
                f"conf>={conf_threshold:.2f} & iou>={iou_threshold:.2f}"
            )
            pbar.update()
    pbar.close()
    return precs, recs


def compute_F_score(prec, rec):
    assert prec.keys() == rec.keys()
    F_score = {}
    for measurement_point in prec.keys():
        p = prec[measurement_point]
        r = rec[measurement_point]
        F_score[measurement_point] = 2 * (p * r) / (p + r) if p or r else 0.

    return F_score


def eval_F_score(gt_instances, pred_instances, conf_levels):
    # Should be omitted, since no filters/modifiers are applied
    # gt_instances, pred_instances = prepare_for_eval(gt_instances, pred_instances)
    prec, rec = eval_prec_rec(gt_instances, pred_instances, conf_levels)
    F_score = compute_F_score(prec, rec)
    return F_score


def eval_F_score_per_class(gt_instances, pred_instances, conf_levels):
    # Should be omitted, since no filters/modifiers are applied
    # gt_instances, pred_instances = prepare_for_eval(gt_instances, pred_instances)
    prec, rec = eval_prec_rec_per_class(gt_instances, pred_instances, conf_levels)
    F_score = compute_F_score(prec, rec)
    return F_score


def compute_optimal_thresholds_per_class(F_score):
    conf_levels, category_ids = map(list, map(set, zip(*F_score.keys())))
    F_score_tensor = torch.zeros(len(conf_levels), len(category_ids))
    for i, conf in enumerate(conf_levels):
        for j, category_id in enumerate(category_ids):
            fs = F_score[conf, category_id]
            F_score_tensor[i, j] = fs

    best_fs, indices = F_score_tensor.max(dim=0)
    optimal_thresholds = {
        category_id: conf_levels[idx]
        for category_id, idx in zip(category_ids, indices)
    }
    return best_fs, optimal_thresholds


def eval_optimal_F_score_per_class(gt_instances, pred_instances, conf_levels):
    # Should be omitted, since no filters/modifiers are applied
    # gt_instances, pred_instances = prepare_for_eval(gt_instances, pred_instances)
    F_score = eval_F_score_per_class(gt_instances, pred_instances, conf_levels)
    best_fs, optimal_thresholds = compute_optimal_thresholds_per_class(F_score)
    return best_fs, optimal_thresholds
