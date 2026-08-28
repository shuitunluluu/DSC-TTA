import argparse
import logging
import os
import random
from datetime import datetime

import clip
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from reproject import (
    build_reprojection,
    build_reproj_text_cls,
    encode_image_preproj,
    get_visual_proj,
)
from utils import (
    build_test_data_loader,
    clip_classifier,
    cls_acc,
    get_prompt_bank,
    get_clip_logits_aug,
    get_config_file,
)


DEFAULT_CFG = {
    "text_alpha": 1e-4,
    "trust_kappa": 50.0,
    "filter_top": 50,
    "filter_bottom": 50,
    "fusion_rho": 0.02,
    "fusion_eta": 0.30,
}


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class DSC_TTA(nn.Module):
    """Dual-space collaborative test-time adaptation."""

    def __init__(self, cfg, num_classes, clip_model, classnames, template, init_centers, text_weights, prompt_bank=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_classes = num_classes
        self.cfg = self._build_cfg(cfg)

        self._init_params()
        self._init_cross_modal_branch(init_centers, text_weights)
        self._init_visual_induction_branch(clip_model, classnames, template, prompt_bank)

    @staticmethod
    def _build_cfg(cfg):
        merged = dict(cfg)
        for key, value in DEFAULT_CFG.items():
            merged.setdefault(key, value)
        return merged

    @staticmethod
    def _float(value, name):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value for {name}: {value!r}") from exc

    @staticmethod
    def _int(value, name):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid integer value for {name}: {value!r}") from exc

    def _init_params(self):
        self.text_alpha = self._float(self.cfg["text_alpha"], "text_alpha")
        self.trust_kappa = self._float(self.cfg["trust_kappa"], "trust_kappa")
        self.filter_top = self._int(self.cfg["filter_top"], "filter_top")
        self.filter_bottom = self._int(self.cfg["filter_bottom"], "filter_bottom")
        self.fusion_rho = self._float(self.cfg["fusion_rho"], "fusion_rho")
        self.fusion_eta = self._float(self.cfg["fusion_eta"], "fusion_eta")

    def _init_cross_modal_branch(self, init_centers, text_weights):
        self.cross_centers = init_centers.T.to(self.device).float()
        self.cross_counts = torch.ones(
            self.num_classes, dtype=torch.float32, device=self.device
        )
        self.base_text = F.normalize(text_weights.T.to(self.device).float(), dim=1)
        self.dynamic_text = self.base_text.clone()

    def _init_visual_induction_branch(self, clip_model, classnames, template, prompt_bank=None):
        image_proj = get_visual_proj(clip_model).to(self.device)
        text_proj = clip_model.text_projection.data.float().t().to(self.device)

        text_proj, image_proj = build_reprojection(
            text_proj=text_proj,
            image_proj=image_proj,
            drop_top=self.filter_top,
            drop_bottom=self.filter_bottom,
        )
        self.visual_proj = image_proj.to(self.device).float()
        self.visual_text = build_reproj_text_cls(
            classnames=classnames,
            template=template,
            clip_model=clip_model,
            text_proj=text_proj,
            prompt_bank=prompt_bank,
        ).to(self.device).float()

        feat_dim = self.visual_text.shape[0]
        self.visual_centers = torch.zeros(
            self.num_classes, feat_dim, dtype=torch.float32, device=self.device
        )
        self.visual_counts = torch.zeros(
            self.num_classes, dtype=torch.float32, device=self.device
        )

    def _current_cross_text(self):
        with torch.no_grad():
            trust = 1.0 - torch.exp(
                -(self.cross_counts.float().unsqueeze(1) - 1.0) / self.trust_kappa
            )
            text = (1.0 - trust) * self.base_text + trust * self.dynamic_text
            return F.normalize(text, dim=1).T.contiguous().half()

    def _cross_logits(self, images, clip_model):
        return get_clip_logits_aug(images, clip_model, self._current_cross_text())

    def _update_cross_centers(self, features, probs):
        features = features.to(self.device).float()
        probs = probs.to(self.device).float()
        with torch.no_grad():
            weights = probs.sum(dim=0)
            weighted_features = probs.T @ features
            denom = weights.unsqueeze(1) + self.cross_counts.unsqueeze(1)
            self.cross_centers = (
                weighted_features + self.cross_counts.unsqueeze(1) * self.cross_centers
            ) / denom
            self.cross_counts += weights

    def _update_dynamic_text(self):
        if self.text_alpha <= 0:
            return
        with torch.no_grad():
            visual_text = F.normalize(self.cross_centers.float(), dim=1)
            dynamic_text = F.normalize(self.dynamic_text.float(), dim=1)
            alignment = torch.clamp(
                torch.sum(dynamic_text * visual_text, dim=1, keepdim=True), min=0.0
            )
            scale = torch.sqrt(self.cross_counts.float()).clamp_min(1.0).unsqueeze(1)
            step = self.text_alpha * alignment / scale
            updated = dynamic_text + step * (visual_text - dynamic_text)
            self.dynamic_text = F.normalize(updated, dim=1)

    def _visual_feature(self, clip_model, images):
        # The visual induction branch uses only the original view.
        image = images[0] if isinstance(images, (list, tuple)) else images
        with torch.no_grad():
            pre_feature = encode_image_preproj(clip_model, image.to(self.device)).float()
            return F.normalize(pre_feature @ self.visual_proj, dim=-1)

    def _visual_assignments(self, feature):
        anchor_logits = 100.0 * (feature @ self.visual_text)
        return anchor_logits, anchor_logits.softmax(dim=-1)

    def _visual_logits(self, feature, anchor_logits):
        if torch.all(self.visual_counts == 0):
            return anchor_logits
        valid = self.visual_counts > 0
        centers = self.visual_centers.clone()
        centers[valid] = F.normalize(centers[valid], dim=-1)
        return 100.0 * (feature @ centers.t())

    def _visual_weight(self):
        return torch.clamp(
            self.fusion_rho * self.visual_counts.mean(), max=self.fusion_eta
        )

    def _update_visual_centers(self, feature, probs):
        with torch.no_grad():
            weights = probs.sum(dim=0)
            weighted_features = probs.T @ feature
            denom = self.visual_counts.unsqueeze(1) + weights.unsqueeze(1)
            self.visual_centers = (
                self.visual_counts.unsqueeze(1) * self.visual_centers + weighted_features
            ) / denom.clamp_min(1e-12)
            self.visual_counts += weights

    def forward_step(self, images, target, clip_model):
        target = target.to(self.device)

        cross_feat, cross_logits, _, cross_probs, _ = self._cross_logits(images, clip_model)

        visual_feat = self._visual_feature(clip_model, images)
        anchor_logits, visual_probs = self._visual_assignments(visual_feat)
        visual_logits = self._visual_logits(visual_feat, anchor_logits)

        logits = cross_logits.float() + self._visual_weight().float() * visual_logits.float()
        accuracy = cls_acc(logits, target)

        self._update_cross_centers(cross_feat, cross_probs)
        self._update_dynamic_text()
        self._update_visual_centers(visual_feat, visual_probs)

        return accuracy

    def run_test(self, loader, clip_model, logger, recent_samples=1000):
        accuracies = []

        with torch.no_grad():
            for index, (images, target) in enumerate(tqdm(loader, desc="Processed test images: ")):
                accuracies.append(self.forward_step(images, target, clip_model))

                if (index + 1) % recent_samples == 0:
                    logger.info(
                        "Last %d samples - Accuracy: %.2f%% | Overall accuracy: %.2f%%",
                        recent_samples,
                        sum(accuracies[-recent_samples:]) / recent_samples,
                        sum(accuracies) / len(accuracies),
                    )

        count = min(recent_samples, len(accuracies))
        return {
            "overall_accuracy": sum(accuracies) / len(accuracies),
            "recent_accuracy": sum(accuracies[-count:]) / count,
        }


def get_arguments():
    parser = argparse.ArgumentParser(description="Run DSC-TTA.")
    parser.add_argument(
        "--config",
        default="configs/vit",
        help="Directory containing per-dataset YAML configurations.",
    )
    parser.add_argument(
        "--datasets",
        default="I",
        type=str,
        help="Datasets separated by '/'. Example: I/A/R/S/V",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./dataset/",
        help="Path to the datasets directory.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="ViT-B/16",
        choices=["ViT-B/16", "RN50"],
        help="CLIP visual backbone.",
    )
    parser.add_argument(
        "--log-path", type=str, default="./log", help="Directory for log files."
    )
    return parser.parse_args()


def build_logger(log_path, backbone, datasets):
    os.makedirs(log_path, exist_ok=True)
    date = datetime.now().strftime("%b%d_%H-%M-%S")
    backbone_name = backbone.replace("/", "_")
    dataset_names = datasets.replace("/", "-")
    filename = f"{date}_{backbone_name}_{dataset_names}_dsc_tta.log"

    logger = logging.getLogger("DSC_TTA")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(message)s")
    file_handler = logging.FileHandler(os.path.join(log_path, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def main():
    args = get_arguments()
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    logger = build_logger(args.log_path, args.backbone, args.datasets)
    datasets = [name for name in args.datasets.split("/") if name]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dataset_name in datasets:
        setup_seeds(1)
        logger.info("Processing %s dataset.", dataset_name)

        cfg = DSC_TTA._build_cfg(get_config_file(args.config, dataset_name))
        logger.info("Configuration: %s", cfg)

        test_loader, classnames, template = build_test_data_loader(
            dataset_name, args.data_root, preprocess
        )
        prompt_bank = get_prompt_bank(dataset_name)
        text_weights = clip_classifier(
            classnames, template, clip_model, prompt_bank=prompt_bank
        )
        init_centers = torch.full(text_weights.shape, 0.001, device=device)

        method = DSC_TTA(
            cfg=cfg,
            num_classes=text_weights.shape[1],
            clip_model=clip_model,
            classnames=classnames,
            template=template,
            init_centers=init_centers,
            text_weights=text_weights,
            prompt_bank=prompt_bank,
        )
        method.eval()

        logger.info("text_alpha = %s", method.text_alpha)
        logger.info("trust_kappa = %s", method.trust_kappa)
        logger.info("filter_top = %s", method.filter_top)
        logger.info("filter_bottom = %s", method.filter_bottom)
        logger.info("fusion_rho = %s", method.fusion_rho)
        logger.info("fusion_eta = %s", method.fusion_eta)
        logger.info(method.run_test(test_loader, clip_model, logger))


if __name__ == "__main__":
    main()
