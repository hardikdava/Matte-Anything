import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.ops import box_convert
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from segment_anything import sam_model_registry, SamPredictor
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model as dino_load_model, predict as dino_predict
import os
from pathlib import Path


class ModelPaths:
    """Class to hold paths to pretrained models"""
    def __init__(self, models_dir="./pretrained"):
        self.models_dir = Path(models_dir)

        self.sam_models = {
            'vit_h': self.models_dir / 'sam_vit_h_4b8939.pth',
            'vit_b': self.models_dir / 'sam_vit_b_01ec64.pth'
        }

        self.vitmatte_models = {
            'vit_b': self.models_dir / 'ViTMatte_B_DIS.pth',
        }

        self.vitmatte_config = {
            'vit_b': './configs/matte_anything.py',
        }

        self.grounding_dino = {
            'config': './configs/swint_ogc.py',
            'weight': self.models_dir / 'groundingdino_swint_ogc.pth'
        }


class BackgroundGenerator:
    """Class for generating checkerboard background images"""
    @staticmethod
    def generate_checkerboard(height, width, num_squares):
        num_squares_h = num_squares
        square_size_h = height // num_squares_h
        square_size_w = square_size_h
        num_squares_w = width // square_size_w

        new_height = num_squares_h * square_size_h
        new_width = num_squares_w * square_size_w
        image = np.zeros((new_height, new_width), dtype=np.uint8)

        for i in range(num_squares_h):
            for j in range(num_squares_w):
                start_x = j * square_size_w
                start_y = i * square_size_h
                color = 255 if (i + j) % 2 == 0 else 200
                image[start_y:start_y + square_size_h, start_x:start_x + square_size_w] = color

        image = cv2.resize(image, (width, height))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image


class TrimapGenerator:
    """Class for generating trimaps for matting"""
    @staticmethod
    def generate_trimap(mask, erode_kernel_size=10, dilate_kernel_size=10):
        erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        eroded = cv2.erode(mask, erode_kernel, iterations=5)
        dilated = cv2.dilate(mask, dilate_kernel, iterations=5)
        trimap = np.zeros_like(mask)
        trimap[dilated==255] = 128
        trimap[eroded==255] = 255
        return trimap

    @staticmethod
    def convert_pixels(gray_image, boxes):
        converted_image = np.copy(gray_image)

        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            converted_image[y1:y2, x1:x2][converted_image[y1:y2, x1:x2] == 1] = 0.5

        return converted_image


class ModelHandler:
    """Class to handle model initialization and inference"""
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_paths = ModelPaths()
        self.sam_model = None
        self.vitmatte = None
        self.grounding_dino = None

    def init_segment_anything(self, model_type='vit_h'):
        """Initialize the segmenting anything model"""
        print(f"Loading SAM model ({model_type})...")
        sam = sam_model_registry[model_type](checkpoint=str(self.model_paths.sam_models[model_type])).to(self.device)
        self.sam_model = SamPredictor(sam)
        return self.sam_model

    def init_vitmatte(self, model_type='vit_b'):
        """Initialize the vitmatte model"""
        print(f"Loading ViTMatte model ({model_type})...")
        cfg = LazyConfig.load(self.model_paths.vitmatte_config[model_type])
        self.vitmatte = instantiate(cfg.model)
        self.vitmatte.to(self.device)
        self.vitmatte.eval()
        DetectionCheckpointer(self.vitmatte).load(str(self.model_paths.vitmatte_models[model_type]))
        return self.vitmatte

    def init_grounding_dino(self):
        """Initialize the grounding dino model"""
        print("Loading GroundingDINO model...")
        self.grounding_dino = dino_load_model(
            self.model_paths.grounding_dino['config'],
            str(self.model_paths.grounding_dino['weight'])
        )
        return self.grounding_dino

    def load_all_models(self, sam_model_type='vit_h', vitmatte_model_type='vit_b'):
        """Load all required models"""
        print('Initializing all models...')
        self.init_segment_anything(sam_model_type)
        self.init_vitmatte(vitmatte_model_type)
        self.init_grounding_dino()
        print('All models loaded successfully.')


class MatteAnything:
    """Main class for image matting without UI dependencies"""
    def __init__(self, model_handler=None):
        # Initialize model handler if not provided
        if model_handler is None:
            self.model_handler = ModelHandler()
            self.model_handler.load_all_models()
        else:
            self.model_handler = model_handler

        self.bg_generator = BackgroundGenerator()
        self.trimap_generator = TrimapGenerator()
        self.device = self.model_handler.device
        self.default_fg_caption = "person"  # Default to detecting person

    def process_image(self,
                      image_path,
                      output_path=None,
                      points=None,
                      object_prompt="person",
                      erode_kernel_size=10,
                      dilate_kernel_size=10,
                      fg_box_threshold=0.25,
                      fg_text_threshold=0.25,
                      tr_caption="glass,lens,crystal,diamond,bubble,bulb,web,grid",
                      tr_box_threshold=0.5,
                      tr_text_threshold=0.25,
                      background_paths=None):
        """
        Process an image to extract the subject with alpha matting

        Parameters:
        -----------
        image_path : str or numpy.ndarray
            Path to the input image or numpy array of image
        output_path : str, optional
            Directory to save outputs
        points : list of tuples, optional
            List of (x, y) point coordinates to use for segmentation guidance
        object_prompt : str, optional
            Text prompt for object detection (default: "person")
        erode_kernel_size : int, optional
            Size of erosion kernel for trimap generation
        dilate_kernel_size : int, optional
            Size of dilation kernel for trimap generation
        fg_box_threshold : float, optional
            Confidence threshold for foreground object detection
        fg_text_threshold : float, optional
            Text matching threshold for foreground object detection
        tr_caption : str, optional
            Text prompt for transparent regions
        tr_box_threshold : float, optional
            Confidence threshold for transparent region detection
        tr_text_threshold : float, optional
            Text matching threshold for transparent region detection
        background_paths : list of str, optional
            List of paths to background images

        Returns:
        --------
        dict
            Dictionary containing various results:
            - mask: Binary segmentation mask
            - alpha: Alpha matte
            - foreground_mask: Foreground with mask applied
            - foreground_alpha: Foreground with alpha matte applied
            - composites: List of background composites if backgrounds provided
        """
        # Load image
        if isinstance(image_path, str):
            input_x = cv2.imread(image_path)
            input_x = cv2.cvtColor(input_x, cv2.COLOR_BGR2RGB)
        else:
            input_x = image_path

        # Initialize points if not provided
        if points is None:
            points = []

        # Set the image in SAM model
        self.model_handler.sam_model.set_image(input_x)

        # Transform the image for DINO
        dino_transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = dino_transform(Image.fromarray(input_x), None)

        # Process points if any
        point_coords, point_labels = None, None
        if len(points) > 0:
            # Format points as (x, y, label) where label is 1 for foreground
            formatted_points = [(p, 1) for p in points]
            points_tensor = torch.Tensor([p for p, _ in formatted_points]).to(self.device).unsqueeze(1)
            labels_tensor = torch.Tensor([l for _, l in formatted_points]).to(self.device).unsqueeze(1)
            transformed_points = self.model_handler.sam_model.transform.apply_coords_torch(points_tensor, input_x.shape[:2])
            point_coords = transformed_points.permute(1, 0, 2)
            point_labels = labels_tensor.permute(1, 0)

        # Use provided object prompt or default
        fg_caption = object_prompt if object_prompt else self.default_fg_caption

        # Run Grounding DINO detection
        fg_boxes, logits, phrases = dino_predict(
            model=self.model_handler.grounding_dino,
            image=image_transformed,
            caption=fg_caption,
            box_threshold=fg_box_threshold,
            text_threshold=fg_text_threshold,
            device=self.device
        )

        # Process detected boxes
        transformed_boxes = None
        if fg_boxes.shape[0] > 0:
            h, w, _ = input_x.shape
            fg_boxes = torch.Tensor(fg_boxes).to(self.device)
            fg_boxes = fg_boxes * torch.Tensor([w, h, w, h]).to(self.device)
            fg_boxes = box_convert(boxes=fg_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            transformed_boxes = self.model_handler.sam_model.transform.apply_boxes_torch(fg_boxes, input_x.shape[:2])

        # Predict segmentation using SAM
        masks, scores, logits = self.model_handler.sam_model.predict_torch(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # Convert masks to numpy
        masks = masks.cpu().detach().numpy()

        # Visualize masks
        mask_colored = np.ones((input_x.shape[0], input_x.shape[1], 3))
        for ann in masks:
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                mask_colored[ann[0] == True, i] = color_mask[i]

        # Overlay mask on image for visualization
        vis_img = input_x / 255 * 0.3 + mask_colored * 0.7

        # Generate alpha matte
        torch.cuda.empty_cache()
        mask = masks[0][0].astype(np.uint8) * 255
        trimap = self.trimap_generator.generate_trimap(mask, erode_kernel_size, dilate_kernel_size).astype(np.float32)
        trimap[trimap == 128] = 0.5
        trimap[trimap == 255] = 1

        # Detect transparent regions
        boxes, logits, phrases = dino_predict(
            model=self.model_handler.grounding_dino,
            image=image_transformed,
            caption=tr_caption,
            box_threshold=tr_box_threshold,
            text_threshold=tr_text_threshold,
            device=self.device
        )

        # Process transparent regions
        if boxes.shape[0] > 0:
            h, w, _ = input_x.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            trimap = self.trimap_generator.convert_pixels(trimap, xyxy)

        # Process with ViTMatte
        input_data = {
            "image": torch.from_numpy(input_x).permute(2, 0, 1).unsqueeze(0) / 255,
            "trimap": torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0),
        }

        torch.cuda.empty_cache()
        alpha = self.model_handler.vitmatte(input_data)['phas'].flatten(0, 2)
        alpha = alpha.detach().cpu().numpy()

        # Create checkerboard background
        background = self.bg_generator.generate_checkerboard(input_x.shape[0], input_x.shape[1], 8)

        # Calculate foreground with alpha blending
        foreground_alpha = input_x * np.expand_dims(alpha, axis=2).repeat(3, 2) / 255 + \
                           background * (1 - np.expand_dims(alpha, axis=2).repeat(3, 2)) / 255

        # Calculate foreground with mask
        foreground_mask = input_x * np.expand_dims(mask / 255, axis=2).repeat(3, 2) / 255 + \
                          background * (1 - np.expand_dims(mask / 255, axis=2).repeat(3, 2)) / 255

        # Save outputs if path is provided
        if output_path:
            os.makedirs(output_path, exist_ok=True)

            # Save PNG with alpha channel
            cv2_alpha = (np.expand_dims(alpha, axis=2) * 255).astype(np.uint8)
            cv2_input_x = cv2.cvtColor(input_x, cv2.COLOR_RGB2BGR)
            rgba = np.concatenate((cv2_input_x, cv2_alpha), axis=2)
            cv2.imwrite(os.path.join(output_path, 'rgba_output.png'), rgba)

            # Save mask
            cv2.imwrite(os.path.join(output_path, 'mask.png'), mask)

            # Save visualization
            cv2.imwrite(os.path.join(output_path, 'visualization.jpg'),
                        cv2.cvtColor((vis_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            # Save foreground extractions
            cv2.imwrite(os.path.join(output_path, 'foreground_mask.png'),
                        cv2.cvtColor((foreground_mask * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_path, 'foreground_alpha.png'),
                        cv2.cvtColor((foreground_alpha * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        # Clip values for display
        foreground_alpha[foreground_alpha > 1] = 1
        foreground_mask[foreground_mask > 1] = 1

        # Process with new backgrounds if provided
        composites = []
        if background_paths:
            for i, bg_path in enumerate(background_paths):
                bg = cv2.imread(bg_path)
                bg = cv2.resize(bg, (input_x.shape[1], input_x.shape[0]))
                bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

                # Create composite
                composite = input_x * np.expand_dims(alpha, axis=2).repeat(3, 2) / 255 + \
                            bg * (1 - np.expand_dims(alpha, axis=2).repeat(3, 2)) / 255

                composites.append(composite)

                # Save composite if output path is provided
                if output_path:
                    cv2.imwrite(os.path.join(output_path, f'composite_{i}.jpg'),
                                cv2.cvtColor((composite * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        # Return results
        return {
            'mask': mask,
            'alpha': alpha,
            'foreground_mask': foreground_mask,
            'foreground_alpha': foreground_alpha,
            'composites': composites,
            'visualization': vis_img
        }


if __name__ == "__main__":
    # Initialize the matting engine
    matte = MatteAnything()

    # Advanced usage with custom settings
    results = matte.process_image(
        image_path='1.png',
        output_path='output_folder',
        object_prompt="person",  # Detect a cat instead of a person
        erode_kernel_size=5,  # Sharper edges
        dilate_kernel_size=15,  # Softer blending
        background_paths=None  # Custom background
    )

    # Access the results
    alpha_matte = results['alpha']
    foreground = results['foreground_alpha']
    composites = results['composites']