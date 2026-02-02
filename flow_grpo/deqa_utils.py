import torch
from transformers import AutoModelForCausalLM

def load_deqascore(device, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        "/mnt/workspace/tyz/A_MODELS/DeQA-Score-Mix3",
        attn_implementation="eager",
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    model.requires_grad_(False)
    
    @torch.no_grad()
    def compute_deqascore(images):
        score = model.score(images)
        score = score / 5
        score = [sc.item() for sc in score]
        return score

    return compute_deqascore