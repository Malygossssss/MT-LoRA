#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import json
import logging
import os

import torch

from evaluation.edge_metrics import evaluate_edge_directory, write_pr_csv
from mtl_loss_schemes import BalancedCrossEntropyLoss

eval_logger = logging.getLogger('eval')


class EdgeMeter(object):
    def __init__(self, pos_weight):
        self.loss_function = BalancedCrossEntropyLoss(
            size_average=True, pos_weight=pos_weight)
        self.formal_results = None
        self.reset()

    @torch.no_grad()
    def update(self, pred, gt):
        gt = gt.squeeze()
        pred = pred.float().squeeze() / 255.0
        loss = self.loss_function(pred, gt).item()
        numel = gt.numel()
        self.n += numel
        self.loss += numel * loss

    def reset(self):
        self.loss = 0.0
        self.n = 0
        self.formal_results = None

    def set_formal_results(self, results):
        self.formal_results = dict(results) if results is not None else None

    def get_score(self, verbose=True):
        eval_dict = {"loss": self.loss / self.n if self.n > 0 else 0.0}
        if self.formal_results:
            for key in ("odsF", "oisF", "ap", "best_threshold", "num_images"):
                if key in self.formal_results:
                    eval_dict[key] = self.formal_results[key]

        if verbose:
            eval_logger.info("\nEdge Detection Evaluation")
            eval_logger.info("loss: %.6f" % eval_dict["loss"])
            if "odsF" in eval_dict:
                eval_logger.info("odsF: %.6f" % eval_dict["odsF"])
                eval_logger.info("oisF: %.6f" % eval_dict["oisF"])
                eval_logger.info("ap: %.6f" % eval_dict["ap"])

        return eval_dict


def eval_edge_predictions(database, save_dir, gt_root=None, overfit=False, write_outputs=True):
    del overfit
    if database != "NYUD":
        raise NotImplementedError("formal edge evaluation is only implemented for NYUD")
    if gt_root is None:
        raise ValueError("gt_root must be provided for NYUD edge evaluation")

    eval_logger.info("Evaluate the saved images (edge)")
    eval_results = evaluate_edge_directory(pred_root=save_dir, gt_root=gt_root)

    if write_outputs:
        base_name = database + "_" + "test" + "_edge"
        json_path = os.path.join(save_dir, base_name + ".json")
        csv_path = os.path.join(save_dir, base_name + "_pr.csv")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(eval_results, handle, indent=2)
        write_pr_csv(csv_path, eval_results)

    eval_logger.info("odsF: %.6f" % eval_results["odsF"])
    eval_logger.info("oisF: %.6f" % eval_results["oisF"])
    eval_logger.info("ap: %.6f" % eval_results["ap"])
    return eval_results
