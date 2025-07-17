import torch
import torch.nn as nn
import torchvision.transforms as T

import open_clip
from PIL import Image


class BIOCLIP_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder, self.image_processor = (
            open_clip.create_model_from_pretrained("hf-hub:imageomics/bioclip-2")
        )

        # set the image to resize both height and width to 224
        self.image_processor.transforms[0] = T.Resize(
            (224, 224), interpolation=T.InterpolationMode.BICUBIC
        )

        # Remove the center crop transform
        del self.image_processor.transforms[1]

    def preprocess(self, image):
        model_input = self.image_processor(image)
        return model_input

    def forward(self, x):
        if isinstance(x, Image.Image):
            x = self.preprocess(x).unsqueeze(0)
            x = x.to(next(self.parameters()).device)
        elif isinstance(x, torch.Tensor):
            if x.dim() == 3:
                x = x.unsqueeze(0)
        elif isinstance(x, list):
            x = [self.preprocess(img).unsqueeze(0) for img in x]
            x = torch.cat(x, dim=0).to(next(self.parameters()).device)

        return self.image_encoder.encode_image(x)
