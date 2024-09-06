import utils
import torch.nn as nn
from flwr_monitoring.model import Net
import torch


class ModelLoader:
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes

    def load_model(self) -> nn.Module:
        """Load and return the specified model."""
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Net().to(DEVICE)
        return model
        # return utils.load_efficientnet(classes=self.num_classes)
