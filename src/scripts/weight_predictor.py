import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.structures import PolygonMasks
from detectron2.utils.visualizer import GenericMask
from detectron2.engine import DefaultPredictor
from typing import  Dict


class Predictor(DefaultPredictor):
    """Custom Detectron2 predictor.

    In addition to default detection fields, this predictor also does:
    1. Mask area calculation for each instance under instance 'areas' field.
    2. Weight prediction under 'weight' field.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.beta = cfg.AREA.REFERENCE_MM / cfg.AREA.REFERENCE_PX

    def __call__(self, original_image: np.ndarray, age, percent) -> Dict:
        """
        Args:
            original_image: An image of shape (H, W, C) (in BGR order).
            date: Date string in YYYY-MM-DD. If not given then weight prediction is not done.

        Returns:
            predictions dict with instance 'areas' and 'weight' in addition to default fields.
        """
        predictions = super().__call__(original_image)

        instances = predictions['instances'].to('cpu')
        detections = instances[instances.scores > percent]
        predictions['instances'] = detections

        with torch.no_grad():
            # Get mask areas
            
            image_size = predictions["instances"].image_size
            masks = list(predictions["instances"].pred_masks.cpu().numpy())
            polygons = [GenericMask(mask, *image_size).polygons for mask in masks]
            areas = PolygonMasks(polygons).area()
            areas = torch.round(self.beta ** 2 * areas)
            predictions["instances"].set("areas", areas)

            x_in = np.array([1.0, np.log(age), np.log(areas.numpy().mean())])
            theta = np.array(self.cfg.WEIGHT.THETA, dtype=x_in.dtype)
            y = theta.T @ x_in
            predictions["weight"] = np.exp(y).item()

            return predictions
