# 🚀 ROCm PyTorch Training

Deep Learning training scripts optimized for AMD ROCm GPUs.

## Features

- 🎮 Full AMD ROCm 5.7+ support
- ⚡ Mixed precision training (FP16/BF16)
- 📈 Multi-GPU distributed training
- 📊 TensorBoard integration
- 💾 Automatic checkpointing

## Supported AMD GPUs

| GPU Series | Architecture | ROCm Support |
|------------|--------------|--------------|
| RX 7900 XTX | RDNA 3 | ✅ Full |
| RX 7900 XT | RDNA 3 | ✅ Full |
| RX 6900 XT | RDNA 2 | ✅ Full |
| RX 6800 XT | RDNA 2 | ✅ Full |
| Instinct MI250 | CDNA 2 | ✅ Full |
| Instinct MI210 | CDNA 2 | ✅ Full |

## Quick Start

```bash
# Install ROCm
wget https://repo.radeon.com/rocm/install_5.7.1.sh
sudo bash install_5.7.1.sh

# Install PyTorch with ROCm
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7

# Clone and run
git clone https://github.com/dickcongwe/rocm-pytorch-training.git
cd rocm-pytorch-training
python train.py --model resnet50 --dataset imagenet --epochs 100
```

## Training Examples

### Image Classification
```bash
python train.py --task classification --model efficientnet_b0 --dataset cifar100
```

### Object Detection
```bash
python train.py --task detection --model yolov8 --dataset coco
```

### NLP Fine-tuning
```bash
python train.py --task nlp --model bert-base --dataset sst2
```

## Performance

Benchmarked on AMD Radeon RX 7900 XTX:

| Model | Batch Size | Throughput | GPU Util |
|-------|------------|------------|----------|
| ResNet-50 | 64 | 1,245 img/s | 95% |
| EfficientNet-B0 | 128 | 2,100 img/s | 92% |
| BERT-base | 32 | 450 seq/s | 88% |

## License

MIT
