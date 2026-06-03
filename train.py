#!/usr/bin/env python3
"""
ROCm PyTorch Training Script
Optimized for AMD GPUs with ROCm support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import argparse
import time
import os


def get_model(name, num_classes):
    models_map = {
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'efficientnet_b0': models.efficientnet_b0,
        'vgg16': models.vgg16,
    }
    model = models_map[name](weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.FakeData(size=10000, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    
    # Model
    model = get_model(args.model, num_classes=1000).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"\nTraining {args.model} for {args.epochs} epochs")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        start = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        elapsed = time.time() - start
        throughput = len(train_dataset) / elapsed
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Time: {elapsed:.1f}s | "
              f"Throughput: {throughput:.0f} img/s")
    
    # Save
    torch.save(model.state_dict(), f'{args.model}_final.pth')
    print(f"\nModel saved to {args.model}_final.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50', choices=['resnet50','resnet101','efficientnet_b0','vgg16'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    train(args)
