# ProSDD

Official implementation of the paper:

**ProSDD: Learning Prosodic Representations for Speech Deepfake Detection against Expressive and Emotional Attacks**

[arXiv Paper](https://arxiv.org/abs/2604.13229) | [Project Website](https://prosdd.github.io/ProSDD_website/)

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/ProSDD/codes.git
cd codes
```

### Setup Environment

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate prosdd
```
### Pre-trained SSL Backbone

ProSDD uses [`facebook/wav2vec2-xls-r-300m`](https://huggingface.co/facebook/wav2vec2-xls-r-300m) as the pre-trained speech encoder for both Stage 1 and Stage 2.

The model is downloaded automatically through the Hugging Face `transformers` library when the training or evaluation scripts are run for the first time.

## Implementation Guidelines

### Pre-trained Checkpoints and Supervised Targets

Pre-trained checkpoints and supervised targets (speaker embeddings only) are available at the following link. Since the prosody embedding files are large, we provide the code used to extract the frame-level prosody embeddings.

Link:  
https://drive.google.com/drive/folders/1h250-Um5qWo-rpeOE6K_Gdeygy46k3xg?usp=sharing

### Available Checkpoints
1. ProSDD trained on ASVspoof 2019
2. ProSDD trained on ASVspoof 2024
3. Baselines trained on ASVspoof 2024: RawNet2; AASIST; XLSR-SLS  

These baselines are provided to help the community efficiently use the ASVspoof 2024 dataset.

### Evaluation Scores
We also release the evaluation scores for all provided checkpoints.

### Prosody Extraction Environment

Frame-level prosody embeddings are extracted using the [Masked Prosody Model](https://huggingface.co/cdminix/masked_prosody_model).

Create and activate the prosody extraction environment:

```bash
conda env create -f environment_prosody.yml
conda activate prosody_extraction
```

The speaker embeddings used as supervised targets are available in the Google Drive folder linked above.

### Running the Code

The repository provides separate scripts for each stage of the ProSDD pipeline:

- `extract_Prosody.py`: Extract frame-level prosody embeddings
- `main_stage1real.py`: Train Stage 1 using bonafide speech
- `main_stage2realfake.py`: Train Stage 2 using bonafide and spoofed speech
- `main_eval.py`: Evaluate a trained ProSDD checkpoint
