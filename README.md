# StegaAttnGAN: Lightweight Cross-Attention Steganography

StegaAttnGAN is an end-to-end generative adversarial framework for **image steganography** — hiding discrete token sequences (e.g., text) inside natural images.  
It integrates:
- A Transformer-based **message encoder**
- A U-Net style **generator with multi-scale cross-attention fusion**
- An **8×8 spatial memory Transformer decoder**
- A lightweight **discriminator**
- A **stochastic noise layer** for robustness against compression and perturbations

The project was developed with the assistance of Generative AI and is released as **open source** for research and educational purposes.

---

## Features
- Hide short messages in small images (e.g., CIFAR-10, 32×32).
- Multi-scale cross-attention for adaptive message embedding.
- Curriculum training on message length and objective weighting.
- Robust decoding under common distortions (Gaussian noise, blur, down/up sampling, JPEG-like quantization).
- Open source and extensible for future research.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Andreas-UI/StegaAttnGAN.git
cd StegaAttnGAN
pip install -r requirements.txt
```

Requirements include:
* Python 3.10+
* PyTorch (CUDA recommended)
* torchvision
* tqdm
* numpy
* matplotlib

## Usage
### Training
To start training on CIFAR-10 with curriculum learning:
```bash
python training_example.py
```

This will:
* Train the model across message lengths (1 → max_msg_len).
* Alternate between text-focused and image-focused phases.
* Save checkpoints and log training progress.

### Inference
To run inference with a trained model (trained weights):
```bash
python inference_example.py
```

This will:
* Load a pretrained checkpoint.
* Encode a random message into a cover image.
* Decode the message from the stego image.
* Print accuracy and visualize cover/stego/residual images.

## Disclaimer

This project was developed with the help of Generative AI.
The author provides the code and documentation as-is and assumes no responsibility for errors, inaccuracies, or misuse. Use at your own risk.

## Contributing
Contributions are welcome!
Please open an issue or submit a pull request for:

* Bug fixes
* New features
* Improved documentation
* Additional datasets or evaluation pipelines
