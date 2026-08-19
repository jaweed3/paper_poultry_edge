# Poultry Fecal Disease Detection — Edge ML

Characterization of lightweight CNN architectures with INT8 quantization for poultry fecal disease detection on edge devices.

## Models

| Model | Test Acc | FP32 Size | INT8 Size | FP32 Latency |
|-------|----------|-----------|-----------|--------------|
| MobileNetV2 | 97.83% | 8.48 MB | 2.30 MB | 1.85 ms |
| ShuffleNetV2 | 97.38% | 4.89 MB | 1.47 MB | 3.17 ms |
| EfficientNet-B0 | 98.35% | 15.30 MB | 4.16 MB | 3.50 ms |

Benchmark: Intel i5-12400F CPU, ONNX Runtime, 200 runs.

## Dataset

Machuve et al. (2022) — 8,770 images, 4 classes (Coccidiosis, Healthy, NCD, Salmonellosis).

## Project Structure

```
├── train_pipeline.py          # Training + ONNX export + INT8 PTQ + Grad-CAM
├── eval_int8.py               # FP32 vs INT8 accuracy evaluation
├── eval_int8_batched.py       # Batched version (faster)
├── models/
│   ├── fp32/                  # PyTorch weights (.pth)
│   ├── onnx/                  # ONNX FP32 models
│   └── ptq/                   # ONNX INT8 quantized models
├── results/
│   ├── experiment_results.json
│   ├── int8_real_accuracy.json
│   └── int8_real_accuracy_lab.json
├── figures/                   # Training curves, confusion matrices, Grad-CAM, Pareto
├── paper/                     # LaTeX source + figures
└── data/images/               # Dataset (train/val/test split)
```

## Reproduce

```bash
pip install torch torchvision onnx onnxruntime numpy matplotlib seaborn tqdm Pillow pytorch_grad_cam
python train_pipeline.py
```

## Key Finding

INT8 PTQ on CPU introduces 6.4–30.1× latency overhead due to dequantization costs. Quantization does not always improve inference speed on CPU without hardware acceleration (VNNI/Tensor Cores).

## License

Research use only.
