from detectron2.utils.visualizer import Visualizer, VisImage
import numpy as np

class WeightVisualizer(Visualizer):
    
    def __init__(self, img_rgb: np.ndarray, age: str, *args, **kwargs):
        """
        Args:
            img_rgb: Numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            age: Age string.
            metadata (Metadata): dataset metadata (e.g. class names and colors).
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """
        super().__init__(img_rgb, *args, **kwargs)
        self.age = age
        
    def draw_weight_predictions(self, predictions: dict, alpha: float = 0.5) -> VisImage:
        """Draw area and weight prediction results on an image.

        Args:
            predictions: Predictions dict containing area and weight fields.
            alpha: Color transparency.

        Returns:
            VisImage image object with visualization.
        """
        instances = predictions["instances"].to(self.cpu_device)

        age_label = f"Umur: {self.age} hari"
        area_labels = [f"{area:,.0f}\nmm2".replace(",", ".") for area in instances.areas]
        weight_label = f"Berat: {predictions['weight']:.3f} kg".replace(".", ",")

        self.overlay_instances(masks=instances.pred_masks, labels=area_labels, alpha=alpha)
        self.draw_text(
            age_label,
            (10, 10),
            color="white",
            font_size=16,
            horizontal_alignment="left",
        )
        self.draw_text(weight_label, (200, 10), color="white", font_size=16, horizontal_alignment="left", )
        return self.output
