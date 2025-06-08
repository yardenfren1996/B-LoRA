import argparse

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from blora_utils import BLOCKS, filter_lora, scale_lora


class SinkhornOTAttnProcessor:
    """Attention processor using Sinkhorn Optimal Transport."""

    def __init__(self, n_iters: int = 20, eps: float = 1e-3):
        self.n_iters = n_iters
        self.eps = eps

    def _sinkhorn(self, log_scores):
        for _ in range(self.n_iters):
            log_scores = log_scores - torch.logsumexp(log_scores, dim=-1, keepdim=True)
            log_scores = log_scores - torch.logsumexp(log_scores, dim=-2, keepdim=True)
        return log_scores.exp()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        residual = hidden_states
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn_scores = torch.bmm(query, key.transpose(-1, -2)) * attn.scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = self._sinkhorn(attn_scores)

        hidden_states = torch.bmm(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states + residual


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, required=True, help="B-LoRA prompt"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="path to save the images"
    )
    parser.add_argument(
        "--content_B_LoRA", type=str, default=None, help="path for the content B-LoRA"
    )
    parser.add_argument(
        "--style_B_LoRA", type=str, default=None, help="path for the style B-LoRA"
    )
    parser.add_argument(
        "--content_alpha", type=float, default=1., help="alpha parameter to scale the content B-LoRA weights"
    )
    parser.add_argument(
        "--style_alpha", type=float, default=1., help="alpha parameter to scale the style B-LoRA weights"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=4, help="number of images per prompt"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                         vae=vae,
                                                         torch_dtype=torch.float16).to("cuda")

    # Get Content B-LoRA SD
    if args.content_B_LoRA is not None:
        content_B_LoRA_sd, _ = pipeline.lora_state_dict(args.content_B_LoRA)
        content_B_LoRA = filter_lora(content_B_LoRA_sd, BLOCKS['content'])
        content_B_LoRA = scale_lora(content_B_LoRA, args.content_alpha)
    else:
        content_B_LoRA = {}

    # Get Style B-LoRA SD
    if args.style_B_LoRA is not None:
        style_B_LoRA_sd, _ = pipeline.lora_state_dict(args.style_B_LoRA)
        style_B_LoRA = filter_lora(style_B_LoRA_sd, BLOCKS['style'])
        style_B_LoRA = scale_lora(style_B_LoRA, args.style_alpha)
    else:
        style_B_LoRA = {}

    # Merge B-LoRAs SD
    res_lora = {**content_B_LoRA, **style_B_LoRA}

    # Load
    pipeline.load_lora_into_unet(res_lora, None, pipeline.unet)

    # Replace attention processors with Sinkhorn OT
    OT_BLOCKS = ['up_blocks.0.attentions.0', 'up_blocks.0.attentions.1']
    for attn_processor_name, _ in pipeline.unet.attn_processors.items():
        if any(attn_processor_name.startswith(b) for b in OT_BLOCKS):
            attn_module = pipeline.unet
            for n in attn_processor_name.split('.')[:-1]:
                attn_module = getattr(attn_module, n)
            attn_module.set_processor(SinkhornOTAttnProcessor())

    # Generate
    images = pipeline(args.prompt, num_images_per_prompt=args.num_images_per_prompt).images

    # Save
    for i, img in enumerate(images):
        img.save(f'{args.output_path}/{args.prompt}_{i}.jpg')
