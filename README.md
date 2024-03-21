# B-LoRA: Implicit Style-Content Separation for Image Stylization

This repository contains the implementation code for the B-LoRA method, which enables implicit style-content separation of a single input image for various image stylization tasks. B-LoRA leverages the power of Stable Diffusion XL (SDXL) and Low-Rank Adaptation (LoRA) to disentangle the style and content components of an image, facilitating applications such as image style transfer, text-based image stylization, and consistent style generation.

## Overview

Image stylization involves manipulating the visual appearance and texture (style) of an image while preserving its underlying objects, structures, and concepts (content). B-LoRA achieves this separation by jointly training two specific transformer blocks (referred to as B-LoRAs) within the SDXL model to reconstruct the given input image. By analyzing the architecture, we found that these two blocks capture the image's content and style, respectively, enabling their independent use for stylization tasks.

Key features of B-LoRA:

- **Implicit Style-Content Separation**: B-LoRA separates the style and content components of a single input image without explicit supervision.
- **Efficient Training**: Only two transformer blocks are optimized, reducing memory requirements and training time.
- **Flexible Stylization**: The learned B-LoRAs can be directly plugged into the SDXL model for various stylization tasks without additional training.
- **High-Quality Results**: B-LoRA achieves robust style transfer while preserving the content of challenging input images.

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- Other dependencies (specified in `requirements.txt`)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-repo/b-lora.git
   cd b-lora
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. **Training B-LoRAs**

   To train the B-LoRAs for a given input image, run:
   ```
   python train.py --input_image path/to/input/image.jpg
   ```
   This will optimize the LoRA weights for the content (`B-LoRA_content.pth`) and style (`B-LoRA_style.pth`) components of the input image.

2. **Image Stylization**

   For image stylization based on a reference style image, run:
   ```
   python stylize.py --content path/to/content/image.jpg --style path/to/style/image.jpg
   ```
   This will generate a new image with the content of the first image and the style of the second image.

   For text-based image stylization, run:
   ```
   python stylize.py --content path/to/content/image.jpg --text "A [content] made of gold"
   ```
   This will generate a new image with the content of the given image and the style specified by the text prompt.

   For consistent style generation, run:
   ```
   python generate.py --style path/to/style/image.jpg --text "A backpack in [style] style"
   ```
   This will generate a new image with the specified content and the style of the given image.

Please refer to the code documentation and examples for more details on usage and configuration.

## Citation

If you use B-LoRA in your research, please cite the following paper:

```bibtex
@inproceedings{b-lora2024,
  title={Implicit Style-Content Separation using B-LoRA},
  author={Anonymous},
  booktitle={ECCV},
  year={2024}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or suggestions, please feel free to open an issue or contact the authors at [yardenfren@gmail.com](mailto:yardenfren@gmail.com).
