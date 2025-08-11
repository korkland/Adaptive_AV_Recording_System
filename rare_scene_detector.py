from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from typing import List, Dict, Optional, Union
import time
import logging
from dataclasses import dataclass
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionConfig:
    model_id: str = "google/paligemma-3b-mix-224"
    device: str = "auto" # auto, cuda, mps, cpu
    torch_dtype: str = "float16" # float16, float32
    max_new_tokens: int = 20
    confidence_threshold: float = 0.7
    batch_size: int = 8
    max_image_size: tuple = (224, 224)
    use_onnx: bool = False
    onnx_model_path: Optional[str] = None

@dataclass
class SceneDetection:
    scene_description: str
    question: str
    answer: str
    confidence: float
    is_detected: bool

@dataclass
class DetectionResults:
    detected_scenes: List[SceneDetection]
    processing_time_ms: float
    total_queries: int
    found_rare_scenes: bool
    image_path: str

    def get_detected_scene_names(self) -> List[str]:
        return [scene.scene_description for scene in self.detected_scenes if scene.is_detected]

    def get_confidence_scores(self) -> Dict[str, float]:
        return {scene.scene_description: scene.confidence for scene in self.detected_scenes}

class RareSceneDetector:
    """
    Production-ready rare scene detector for automotive environments

    Features:
    - Batch processing for efficiency
    - ONNX export capability
    - Configurable confidence thresholds
    - Automotive-specific scene queries
    - Memory-efficient processing
    """
    def __init__(self, config: DetectionConfig):
        self.config = config

        self.device = self._setup_device()

        if config.use_onnx and config.onnx_model_path:
            self._load_onnx_model()
        else:
            self._load_pytorch_model()

        print(f"✓ RareSceneDetector initialized on {self.device}")

    def _setup_device(self) -> torch.device:
        if self.config.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)

    def _load_pytorch_model(self):
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            use_fast=True
        )

        torch_dtype = getattr(torch, self.config.torch_dtype)

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device.type != "mps" else None,
            low_cpu_mem_usage=True
        )

        if self.device.type == "mps":
            self.model = self.model.to(self.device)

        self.model.eval()
        self.use_onnx = False

    def _load_onnx_model(self):
        try:
            import onnxruntime as ort

            providers = []
            if self.device.type == "cuda":
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')

            self.onnx_session = ort.InferenceSession(
                self.config.onnx_model_path,
                providers=providers
            )

            # Still need processor for tokenization
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_id,
                use_fast=True
            )

            self.use_onnx = True
            logger.info(f"✓ ONNX model loaded from {self.config.onnx_model_path}")

        except ImportError:
            logger.error("ONNX Runtime not installed. Install with: pip install onnxruntime")
            raise
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            logger.info("Falling back to PyTorch model")
            self._load_pytorch_model()

    def detect_rare_scenes(
        self,
        image_path: Union[str, Path],
        scene_descriptions: List[str]
    ) -> DetectionResults:
        """
        Detect rare scenes in an image

        Args:
            image_path: Path to image file
            scene_descriptions: List of scene descriptions to detect

        Returns:
            DetectionResults object with all detection information
        """
        if not scene_descriptions:
            return DetectionResults([], 0.0, 0, False, str(image_path))

        start_time = time.time()

        try:
            # Load and validate image
            image = self._load_and_validate_image(image_path)

            # Process in batches if needed
            all_detections = []

            for i in range(0, len(scene_descriptions), self.config.batch_size):
                batch_scenes = scene_descriptions[i:i + self.config.batch_size]
                batch_detections = self._process_batch(image, batch_scenes)
                all_detections.extend(batch_detections)

            processing_time = (time.time() - start_time) * 1000

            # Filter detected scenes
            detected_scenes = [d for d in all_detections if d.is_detected]

            return DetectionResults(
                detected_scenes=all_detections,
                processing_time_ms=processing_time,
                total_queries=len(scene_descriptions),
                found_rare_scenes=len(detected_scenes) > 0,
                image_path=str(image_path)
            )

        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            processing_time = (time.time() - start_time) * 1000

            # Return empty results on error
            empty_detections = [
                SceneDetection(scene, f"Do you see {scene.lower()}?", "error", 0.0, False)
                for scene in scene_descriptions
            ]

            return DetectionResults(
                detected_scenes=empty_detections,
                processing_time_ms=processing_time,
                total_queries=len(scene_descriptions),
                found_rare_scenes=False,
                image_path=str(image_path)
            )

    def _load_and_validate_image(self, image_path: Union[str, Path]) -> Image.Image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")

            # Resize if too large
            if image.size[0] > self.config.max_image_size[0] or image.size[1] > self.config.max_image_size[1]:
                image.thumbnail(self.config.max_image_size, Image.Resampling.LANCZOS)
                logger.debug(f"Resized image to {image.size}")

            return image

        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")

    def _process_batch(self, image: Image.Image, scene_descriptions: List[str]) -> List[SceneDetection]:
        # Convert to questions with image tokens
        prompts = []
        for scene in scene_descriptions:
            if scene.endswith('?'):
                question = f"<image>{scene}"
            else:
                question = f"<image>Do you see {scene.lower()}?"
            prompts.append(question)

        # Prepare batch
        images = [image] * len(prompts)

        if self.use_onnx:
            answers = self._run_onnx_inference(prompts, images)
        else:
            answers = self._run_pytorch_inference(prompts, images)

        # Process results
        detections = []
        for scene, question, answer in zip(scene_descriptions, prompts, answers):
            clean_question = question.replace("<image>", "").strip()
            is_positive = self._is_positive_answer(answer)
            confidence = self._calculate_confidence(answer)

            detections.append(SceneDetection(
                scene_description=scene,
                question=clean_question,
                answer=answer,
                confidence=confidence,
                is_detected=is_positive and confidence >= self.config.confidence_threshold
            ))

        return detections

    def _run_pytorch_inference(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        # Process batch
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        input_len = inputs["input_ids"].shape[-1]

        # Generate answers
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            generated_texts = outputs[:, input_len:]
            answers = [
                self.processor.decode(out, skip_special_tokens=True).strip()
                for out in generated_texts
            ]

        return answers

    def _run_onnx_inference(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        # TODO
        # This is a placeholder - actual ONNX implementation would need
        # custom preprocessing and postprocessing for PaliGemma
        raise NotImplementedError("ONNX inference not yet implemented for PaliGemma")

    def _is_positive_answer(self, answer: str) -> bool:
        """Check if answer indicates positive detection"""
        answer_lower = answer.lower().strip()
        positive_indicators = ['yes', 'there is', 'there are', 'i can see', 'i see', 'visible']
        negative_indicators = ['no', 'not', "don't", "can't", 'unable', 'cannot']

        has_positive = any(indicator in answer_lower for indicator in positive_indicators)
        has_negative = any(indicator in answer_lower for indicator in negative_indicators)

        return has_positive and not has_negative

    def _calculate_confidence(self, answer: str) -> float:
        """Calculate confidence score based on answer clarity"""
        answer_lower = answer.lower().strip()

        # High confidence for clear yes/no answers
        if answer_lower in ['yes', 'no']:
            return 0.95

        # Medium-high confidence for descriptive answers
        if any(phrase in answer_lower for phrase in ['there is', 'there are', 'i can see']):
            return 0.85

        # Medium confidence for qualified answers
        if any(phrase in answer_lower for phrase in ['yes,', 'visible', 'appears']):
            return 0.75

        # Lower confidence for uncertain answers
        if len(answer_lower) > 30:
            return 0.6

        return 0.7

    def export_to_onnx(self, output_path: str, example_image_path: str):
        """
        Export model to ONNX format for production deployment

        Args:
            output_path: Path to save ONNX model
            example_image_path: Path to example image for tracing
        """
        if self.use_onnx:
            logger.warning("Model is already using ONNX")
            return

        try:
            # Load example image
            example_image = Image.open(example_image_path).convert("RGB")
            example_prompt = "<image>Do you see a person?"

            # Prepare inputs
            inputs = self.processor(
                text=[example_prompt],
                images=[example_image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Export to ONNX
            torch.onnx.export(
                self.model,
                (inputs["input_ids"], inputs["pixel_values"]),
                output_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input_ids', 'pixel_values'],
                output_names=['output'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'pixel_values': {0: 'batch_size'},
                    'output': {0: 'batch_size', 1: 'sequence'}
                }
            )

            logger.info(f"✓ Model exported to ONNX: {output_path}")

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise

# Automotive-specific scene definitions
AUTOMOTIVE_RARE_SCENES = [
    "two people riding one bicycle together",
    "person selling food on sidewalk",
    "street performer playing music",
    "pedestrian crossing outside crosswalk",
    "car parked on sidewalk",
    "motorcycle lane splitting",
    "construction workers without safety gear",
    "emergency vehicle with lights flashing",
    # "person jaywalking across street",
    # "bicycle rider without helmet",
    # "food truck blocking traffic",
    # "person walking in bike lane",
    # "scooter on sidewalk",
    # "person loading delivery truck"
]

def main():
    """Example usage and testing"""
    # Configuration for production
    config = DetectionConfig(
        confidence_threshold=0.7,
        batch_size=4,
        torch_dtype="float16"
    )

    # Initialize detector
    detector = RareSceneDetector(config)

    # Test image
    image_path = "example_scene.png"

    if not os.path.exists(image_path):
        logger.error(f"Test image not found: {image_path}")
        return

    # Run detection
    results = detector.detect_rare_scenes(image_path, AUTOMOTIVE_RARE_SCENES)

    # Display results
    print("\n" + "="*60)
    print("RARE SCENE DETECTION RESULTS")
    print("="*60)
    print(f"Image: {results.image_path}")
    print(f"Processing time: {results.processing_time_ms:.1f}ms")
    print(f"Total queries: {results.total_queries}")
    print(f"Detected scenes: {len(results.get_detected_scene_names())}")
    print()

    if results.found_rare_scenes:
        print("🎯 DETECTED RARE SCENES:")
        print("-" * 40)
        for scene in results.detected_scenes:
            if scene.is_detected:
                print(f"✓ {scene.scene_description}")
                print(f"  Answer: {scene.answer}")
                print(f"  Confidence: {scene.confidence:.2f}")
                print()
    else:
        print("No rare scenes detected above threshold")

    # Show confidence scores for all queries
    print("\nALL SCENE CONFIDENCE SCORES:")
    print("-" * 40)
    for scene in results.detected_scenes:
        status = "✓" if scene.is_detected else "✗"
        print(f"{status} {scene.scene_description}: {scene.confidence:.2f}")

if __name__ == "__main__":
    main()