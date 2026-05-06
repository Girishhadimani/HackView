"""
Grad-CAM Heatmap Generator
===========================
Produces pixel-level explanation maps showing which image regions
triggered the deepfake decision.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Works with any CNN that exposes feature maps before global pooling.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients:   Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, 3, H, W) — single image
            target_class: None → use predicted class

        Returns:
            heatmap: (H, W) float32 in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        score  = output["logit"] if "logit" in output else output["score"]

        # Backward pass for target
        self.model.zero_grad()
        score.backward(torch.ones_like(score))

        # Grad-CAM formula: global avg pool of gradients × activations
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam     = F.relu(cam)

        # Resize to input resolution
        H, W = input_tensor.shape[-2:]
        cam   = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam   = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_heatmap(
    image_np: np.ndarray,
    heatmap:  np.ndarray,
    alpha:    float = 0.4,
    colormap: int   = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        image_np: (H, W, 3) uint8 RGB
        heatmap:  (H, W) float32 in [0, 1]
        alpha:    blend weight for heatmap
        colormap: OpenCV colormap

    Returns:
        blended: (H, W, 3) uint8 RGB
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)
    return blended


def generate_heatmap_image(
    model:        torch.nn.Module,
    target_layer: torch.nn.Module,
    image_tensor: torch.Tensor,
    image_np:     np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-level helper: generate and overlay Grad-CAM heatmap.

    Returns:
        heatmap_raw: (H, W) raw float heatmap
        overlay:     (H, W, 3) uint8 visualization
    """
    cam_gen  = GradCAM(model, target_layer)
    heatmap  = cam_gen.generate(image_tensor.unsqueeze(0))
    cam_gen.remove_hooks()

    overlay  = overlay_heatmap(image_np, heatmap)
    return heatmap, overlay
