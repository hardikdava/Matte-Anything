import os
import time
import subprocess
from typing import List
from pathlib import Path
from uuid import uuid4

from cog import BasePredictor, Input, Path
import torch
from PIL import Image
import supervision as sv

from infer import MatteAnything


class CogPredictor(BasePredictor):

    def setup(self) -> None:
        self.predictor = MatteAnything()

    @torch.inference_mode()
    def predict(
            self,
            image: Path = Input(
                description="Input image",
                default=None
            ),
            prompt: str = Input(
                description="Object prompt",
                default="person"
            ),
            background_color: str = Input(
                description="Solid background color",
                default="(255, 255, 255)"
            ),
            background_image: Path = Input(
                description="Background image",
                default=None
            ),

    ) -> list[Path]:
        tmp_output_path = os.path.join("output", str(uuid4()))
        """Run a single prediction on the model"""
        _ = self.predictor.process_image(
            image_path=str(image),
            output_path=tmp_output_path,
            object_prompt=prompt,  # Detect a cat instead of a person
            erode_kernel_size=5,  # Sharper edges
            dilate_kernel_size=15,  # Softer blending
            background_paths=background_image,  # Custom background
            background_color=background_color
        )
        outputs = []
        image_files = sv.list_files_with_extensions(tmp_output_path)
        for image_file in image_files:
            outputs.append(Path(image_file))
        return outputs
