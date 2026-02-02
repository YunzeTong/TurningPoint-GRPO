from paddleocr import PaddleOCR
import torch
import numpy as np
from Levenshtein import distance
from typing import List, Union, Tuple, Optional
from PIL import Image
import os

class OcrScorer:
    def __init__(self,
                use_gpu: bool = False,
                det_model_dir: Optional[str] = None,
                rec_model_dir: Optional[str] = None
    ):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        :param det_model_dir: Path to detection model directory (optional, auto-download if None)
        :param rec_model_dir: Path to recognition model directory (optional, auto-download if None)
        """
        ocr_kwargs = {
            'use_textline_orientation': False,  # Updated from deprecated use_angle_cls
        }

        # Add custom model paths if provided (using new API parameter names)
        if det_model_dir is not None:
            ocr_kwargs['text_detection_model_dir'] = det_model_dir
        if rec_model_dir is not None:
            ocr_kwargs['text_recognition_model_dir'] = rec_model_dir

        # Only set lang when not using custom model paths
        # (lang is ignored when model directories are specified)
        if det_model_dir is None and rec_model_dir is None:
            ocr_kwargs['lang'] = "en"

        # Note: use_gpu parameter might be deprecated in newer versions
        # GPU is used automatically if available in some versions
        # We keep it for backward compatibility but it may be ignored
        if use_gpu:
            ocr_kwargs['use_gpu'] = use_gpu

        self.ocr = PaddleOCR(**ocr_kwargs)

    @torch.no_grad()
    def __call__(self,
                images: Union[List[Image.Image], List[np.ndarray]],
                prompts: List[str]) -> torch.Tensor:
        """
        Calculate OCR reward
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: Reward tensor (CPU)
        """
        # Extract target text from prompts (handle different formats)
        extracted_prompts = []
        for prompt in prompts:
            if '"' in prompt:
                # Extract text between first pair of quotes
                parts = prompt.split('"')
                if len(parts) >= 2:
                    extracted_prompts.append(parts[1])
                else:
                    print(f"Warning: Malformed prompt with quotes: {prompt}")
                    extracted_prompts.append(prompt)
            else:
                # No quotes, use the whole prompt
                extracted_prompts.append(prompt)

        rewards = []
        # Ensure input lengths are consistent
        assert len(images) == len(extracted_prompts), "Images and prompts must have the same length"

        for idx, (img, target_text) in enumerate(zip(images, extracted_prompts)):
            # Convert image format
            if isinstance(img, Image.Image):
                img = np.array(img)

            try:
                # OCR recognition (cls parameter removed in PaddleOCR 3.2.0)
                result = self.ocr.ocr(img)
                
                # DEBUG INFO, 暂时关闭因为太费时
                # # Debug: Print OCR result for first few samples
                # if idx < 2:  # Only print first 2 samples to avoid spam
                #     print(f"[OCR Debug {idx}] Target text: '{target_text}'")
                #     if result and len(result) > 0:
                #         print(f"[OCR Debug {idx}] Result structure: {type(result)}, len={len(result)}")
                #         print(f"[OCR Debug {idx}] First page type: {type(result[0])}")
                #         # Check if result[0] has __len__
                #         if hasattr(result[0], '__len__'):
                #             print(f"[OCR Debug {idx}] OCR found {len(result[0])} text regions")
                #     else:
                #         print(f"[OCR Debug {idx}] OCR found no text")

                # Extract recognized text (handle PaddleOCR 3.2.0 OCRResult object)
                recognized_text = ''
                if result and len(result) > 0:
                    try:
                        ocr_result = result[0]

                        # Check if it's an OCRResult dictionary (PaddleOCR 3.2.0)
                        if isinstance(ocr_result, dict) and 'rec_texts' in ocr_result:
                            # New PaddleX OCRResult format (dict with 'rec_texts' and 'rec_scores')
                            text_parts = []
                            rec_texts = ocr_result.get('rec_texts', [])
                            rec_scores = ocr_result.get('rec_scores', [])

                            for i, text in enumerate(rec_texts):
                                # Check confidence if available
                                if i < len(rec_scores) and rec_scores[i] > 0:
                                    text_parts.append(str(text))
                                elif i >= len(rec_scores):
                                    # No score available, include text
                                    text_parts.append(str(text))
                            recognized_text = ''.join(text_parts)

                            # if idx < 2:
                            #     print(f"[OCR Debug {idx}] Extracted from OCRResult dict: {text_parts[:5]}")
                            #     print(f"[OCR Debug {idx}] Scores: {rec_scores[:5]}")

                        elif isinstance(ocr_result, (list, tuple)):
                            # Old format: list of [[bbox], (text, conf)]
                            text_parts = []
                            for detection in ocr_result:
                                if isinstance(detection, (list, tuple)) and len(detection) >= 2:
                                    text_info = detection[1]
                                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                        text, confidence = text_info[0], text_info[1]
                                        if confidence > 0:
                                            text_parts.append(str(text))
                            recognized_text = ''.join(text_parts)

                        else:
                            # Debug: print available attributes if format is unknown
                            if idx < 2:
                                print(f"[OCR Debug {idx}] Unknown OCR result format")
                                print(f"[OCR Debug {idx}] Type: {type(ocr_result)}")
                                print(f"[OCR Debug {idx}] Attributes: {dir(ocr_result)}")
                                # Try common attribute names
                                for attr in ['text', 'texts', 'rec_text', 'rec_texts', 'result', 'data']:
                                    if hasattr(ocr_result, attr):
                                        attr_val = getattr(ocr_result, attr)
                                        print(f"[OCR Debug {idx}] Found attribute '{attr}': {type(attr_val)} = {str(attr_val)[:200]}")
                            recognized_text = ''

                    except Exception as parse_error:
                        print(f"[OCR Debug {idx}] Parse error: {parse_error}")
                        import traceback
                        traceback.print_exc()
                        recognized_text = ''

                # Normalize text for comparison
                recognized_text_norm = recognized_text.replace(' ', '').lower()
                target_text_norm = target_text.replace(' ', '').lower()

                # if idx < 2:
                #     print(f"[OCR Debug {idx}] Recognized: '{recognized_text}' -> '{recognized_text_norm}'")
                #     print(f"[OCR Debug {idx}] Target normalized: '{target_text_norm}'")

                # Calculate edit distance
                if not target_text_norm:
                    # Empty target text, perfect match
                    dist = 0
                elif target_text_norm in recognized_text_norm:
                    # Exact match
                    dist = 0
                else:
                    dist = distance(recognized_text_norm, target_text_norm)

                # Cap distance at target length
                if dist > len(target_text_norm):
                    dist = len(target_text_norm)

            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                print(f"[OCR Error {idx}] Processing failed: {str(e)}")
                import traceback
                traceback.print_exc()
                dist = len(target_text) if target_text else 1

            # Calculate reward
            reward = 1 - dist / max(len(target_text), 1)  # Avoid division by zero
            rewards.append(reward)

            # if idx < 2:
            #     print(f"[OCR Debug {idx}] Distance: {dist}, Reward: {reward:.4f}")

        return rewards

class OcrScorer_video_or_image:
    def __init__(self, use_gpu: bool = False, det_model_dir: Optional[str] = None, rec_model_dir: Optional[str] = None):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        :param det_model_dir: Path to detection model directory (optional, auto-download if None)
        :param rec_model_dir: Path to recognition model directory (optional, auto-download if None)
        """
        ocr_kwargs = {
            'use_textline_orientation': False,  # Updated from deprecated use_angle_cls
        }

        # Add custom model paths if provided (using new API parameter names)
        if det_model_dir is not None:
            ocr_kwargs['text_detection_model_dir'] = det_model_dir
        if rec_model_dir is not None:
            ocr_kwargs['text_recognition_model_dir'] = rec_model_dir

        # Only set lang when not using custom model paths
        # (lang is ignored when model directories are specified)
        if det_model_dir is None and rec_model_dir is None:
            ocr_kwargs['lang'] = "en"

        # Note: use_gpu parameter might be deprecated in newer versions
        # GPU is used automatically if available in some versions
        # We keep it for backward compatibility but it may be ignored
        if use_gpu:
            ocr_kwargs['use_gpu'] = use_gpu

        self.ocr = PaddleOCR(**ocr_kwargs)
        self.frame_interval = 4

    @torch.no_grad()
    def __call__(self, images: Union[List[Image.Image], List[np.ndarray]], prompts: List[str]) -> Tuple[List[float], torch.Tensor]:
        """
        :param images: List of images or videos (each video as np.ndarray of shape [F, H, W, C])
        :param prompts: List of prompts containing target text
        :return: (List of OCR rewards, Tensor of attention regions)
        """
        prompts = [prompt.split('"')[1] for prompt in prompts]
        assert len(images) == len(prompts), "Mismatch between images and prompts."

        rewards = []
        for img, prompt in zip(images, prompts):
            prompt = prompt.replace(' ', '').lower()
            frame_rewards = []

            # Handle video: shape (F, H, W, C)
            if isinstance(img, np.ndarray) and img.ndim == 4:
                sampled_frames = img[::self.frame_interval]
            else:
                sampled_frames = [img]

            for frame in sampled_frames:
                region = None
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)
                try:
                    result = self.ocr.ocr(frame)

                    # Handle PaddleOCR 3.2.0 OCRResult dictionary format
                    if result and len(result) > 0:
                        ocr_result = result[0]
                        if isinstance(ocr_result, dict) and 'rec_texts' in ocr_result:
                            # New format: dictionary with 'rec_texts' and 'rec_scores'
                            rec_texts = ocr_result.get('rec_texts', [])
                            rec_scores = ocr_result.get('rec_scores', [])
                            text_parts = [str(text) for i, text in enumerate(rec_texts)
                                         if i < len(rec_scores) and rec_scores[i] > 0]
                            text = ''.join(text_parts)
                        elif isinstance(ocr_result, (list, tuple)):
                            # Old format: list of [[bbox], (text, conf)]
                            text = ''.join([res[1][0] if res[1][1] > 0 else '' for res in ocr_result])
                        else:
                            text = ''
                    else:
                        text = ''

                    text = text.replace(' ', '').lower()
                    dist = distance(text, prompt)
                    dist = min(dist, len(prompt))

                except Exception as e:
                    print(f"OCR failed on frame: {e}")
                    dist = len(prompt)

                reward = 1 - dist / len(prompt)
                if reward > 0:
                    frame_rewards.append(reward)

            if frame_rewards:
                rewards.append(sum(frame_rewards) / len(frame_rewards))
            else:
                rewards.append(0.0)

        return rewards

if __name__ == "__main__":
    example_image_path = "media_images_eval_images_499_ef42de47b8ec98892954.jpg"
    example_image = Image.open(example_image_path)
    example_prompt = 'New York Skyline with "Hello World" written with fireworks on the sky'
    # Instantiate scorer
    scorer = OcrScorer(use_gpu=False)

    # Call scorer and print result
    reward = scorer([example_image], [example_prompt])
    print(f"OCR Reward: {reward}")