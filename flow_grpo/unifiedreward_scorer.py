from PIL import Image
import torch
import re
from typing import List, Union
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import warnings
warnings.filterwarnings("ignore")



class UnifiedRewardScorer(torch.nn.Module):
    def __init__(self, model_path="/mnt/workspace/tyz/A_MODELS/UnifiedReward-qwen-7b", device="cuda", dtype=torch.bfloat16):
        """
        UnifiedReward scorer for image generation assessment.

        Args:
            model_path: Path to the UnifiedReward model
            device: Device to run the model on
            dtype: Data type for the model (bfloat16 recommended for Qwen2 models)
        """
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, device_map={"": device}
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.model.eval()
        if dtype is not None:
            self.model = self.model.to(dtype=dtype)
        self.model.requires_grad_(False)

        print("UnifiedReward-Qwen model loaded successfully!")

    def _extract_score(self, text_output: str) -> float:
        """
        Extract numerical score from model output.
        Expected format: "Final Score: X" where X is between 1-5
        """
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        match = re.search(pattern, text_output)
        if match:
            try:
                score = float(match.group(1))
                return score
            except ValueError:
                return 0.0
        return 0.0

    @torch.no_grad()
    def __call__(self, prompts: List[str], images: List[Image.Image]) -> List[float]:
        """
        Evaluate images based on text prompts.

        Args:
            prompts: List of text prompts/captions
            images: List of PIL Images

        Returns:
            List of scores normalized to [0, 1]
        """
        scores = []

        for prompt, image in zip(prompts, images):

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": f'You are given a text caption and a generated image based on that caption. '
                                    f'Your task is to evaluate this image based on two key criteria:\n'
                                    f'1. Alignment with the Caption: Assess how well this image aligns with the provided caption. '
                                    f'Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n'
                                    f'2. Overall Image Quality: Examine the visual quality of this image, including clarity, '
                                    f'detail preservation, color accuracy, and overall aesthetic appeal.\n'
                                    f'Extract key elements from the provided text caption, evaluate their presence in the generated image using the format: '
                                    f'\'element (type): value\' (where value=0 means not generated, and value=1 means generated), '
                                    f'and assign a score from 1 to 5 after \'Final Score:\'.\n'
                                    f'Your task is provided as follows:\n'
                                    f'Text Caption: [{prompt}]'
                        },
                    ],
                }
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)


            try:
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()

                # Extract score (1-5 range)
                raw_score = self._extract_score(output_text)
                # Normalize to [0, 1]
                normalized_score = raw_score / 5.0
                scores.append(normalized_score)

            except Exception as e:
                print(f"Error during UnifiedReward inference: {e}")
                scores.append(0.0)

        return scores


# Usage example
def main():
    scorer = UnifiedRewardScorer(
        device="cuda",
        dtype=torch.bfloat16
    )

    # Test with an example image
    test_image = Image.new('RGB', (512, 512), color='red')
    prompts = ['A red colored square']

    scores = scorer(prompts, [test_image])
    print(f"Scores: {scores}")


if __name__ == "__main__":
    main()
