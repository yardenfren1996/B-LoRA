# Implicit Style-Content Separation using B-LoRA
<a href="https://B-LoRA.github.io/B-LoRA/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a> [![arXiv](https://img.shields.io/badge/arXiv-2403.14572-b31b1b.svg)](https://arxiv.org/abs/2403.14572)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yardenfren1996/B-LoRA/blob/main/B_LoRA_inference.ipynb) [![HuggingFace demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Yardenfren/B-LoRA)

![Teaser Image](docs/teaser_blora.png)

This repository contains the official implementation of the B-LoRA method, which enables implicit style-content separation of a single input image for various image stylization tasks. B-LoRA leverages the power of Stable Diffusion XL (SDXL) and Low-Rank Adaptation (LoRA) to disentangle the style and content components of an image, facilitating applications such as image style transfer, text-based image stylization, and consistent style generation.

## 🔧 21.5.2024: Important Update 🔧
There were some issues with the new versions of diffusers and PEFT that caused the fine-tuning process to not converge as quickly as desired. In the meantime, we have uploaded the original training script that we used in the paper.

Please note that we used a previous version of diffusers (0.25.0) and did not use PEFT.

## Getting Started

### Prerequisites
- Python 3.11.6+
- PyTorch 2.1.1+
- Other dependencies (specified in `requirements.txt`)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yardenfren1996/B-LoRA.git
   cd B-LoRA
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   (for windows 10 [here](https://github.com/yardenfren1996/B-LoRA/issues/6))

### Usage

1. **Training B-LoRAs**

   To train the B-LoRAs for a given input image, run:
   ```
   accelerate launch train_dreambooth_b-lora_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --instance_data_dir="<path/to/example_images>" \
    --output_dir="<path/to/output_dir>" \
    --instance_prompt="<prompt>" \
    --resolution=1024 \
    --rank=64 \
    --train_batch_size=1 \
    --learning_rate=5e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1000 \
    --checkpointing_steps=500 \
    --seed="0" \
    --gradient_checkpointing \
    --use_8bit_adam \
    --mixed_precision="fp16"
      ```
This will optimize the B-LoRA weights for the content and style and store them in  `output_dir`.
Parameters that need to replace  `instance_data_dir`, `output_dir`, `instance_prompt` (in our paper we use `A [v]`)


![Apps Image](docs/apps_method1.png)

2. **Inference**   

   For image stylization based on a reference style image (1), run:
   ```
   python inference.py --prompt="A <c> in <s> style" --content_B_LoRA="<path/to/content_B-LoRA>" --style_B_LoRA="<path/to/style_B-LoRA>" --output_path="<path/to/output_dir>"
   ```
   This will generate new images with the content of the first B-LoRA and the style of the second B-LoRA.
   Note that you need to replace `c` and `s` in the prompt according to the optimization prompt.

   For text-based image stylization (2), run:
   ```
   python inference.py --prompt="A <c> made of gold"" --content_B_LoRA="<path/to/content_B-LoRA>" --output_path="<path/to/output_dir>"
   ```
   This will generate new images with the content of the given B-LoRA and the style specified by the text prompt.

   For consistent style generation (3), run:
   ```
   python inference.py --prompt="A backpack in <s> style" --style_B_LoRA="<path/to/style_B-LoRA>" --output_path="<path/to/output_dir>"
   ```
   This will generate new images with the specified content and the style of the given B-LoRA.


   Several additional parameters that you can set in the `inference.py` file include:
   1. `--content_alpha`, `--style_alpha` for controlling the strength of the adapters.
   2. `--num_images_per_prompt` for specifying the number of output images.

   (For a111 and comfy see this [issue](https://github.com/yardenfren1996/B-LoRA/issues/7))

## Citation

If you use B-LoRA in your research, please cite the following paper:

```bibtex
@misc{frenkel2024implicit,
      title={Implicit Style-Content Separation using B-LoRA}, 
      author={Yarden Frenkel and Yael Vinker and Ariel Shamir and Daniel Cohen-Or},
      year={2024},
      eprint={2403.14572},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or suggestions, please feel free to open an issue or contact the authors at [yardenfren@gmail.com](mailto:yardenfren@gmail.com).
