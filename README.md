# Custom Document OCR Training Pipeline with CRNN

## Overview
This repository contains a custom Optical Character Recognition (OCR) training pipeline that leverages a Convolutional Recurrent Neural Network (CRNN) architecture. The pipeline was trained by combining synthtic data and real world data from documents

## Features
- **Custom CRNN Architecture**: Combines CNN and RNN layers for feature extraction and sequential text recognition.
- **Flexible Training**: Supports training on both synthetic and real-world datasets.
- **Evaluation Metrics**: Custom CER (Character Error Rate) for evaluation of model's performance
- **Preprocessing Pipelines**: Handles data resizing, normalization, and augmentation.

## Prerequisites

Ensure you have the dependencies installed by using the requirements.txt file

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Model Architecture

The CRNN architecture consists of:
1. **Convolutional Layers**: Extract spatial features from input images.
2. **Recurrent Layers**: Capture sequential dependencies in features.
3. **CTC Loss**: Handles alignment between predictions and ground truth.

### Diagram:
![image](https://github.com/user-attachments/assets/6faeb540-1133-4bd7-9c18-bb725e4ff4b4)


## Results
Results are tabulated in a txt file named 'eval_res.txt' where the model achieved around 96% accuracy for a completely unseen dataset

## Future Improvements

- Incorporating attention mechanisms for improved recognition.
- Adding more diversity to the dataset to perform well with the edge cases

## Acknowledgments

- [CRNN Paper](https://arxiv.org/abs/1507.05717)
