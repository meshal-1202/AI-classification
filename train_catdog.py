# train_catdog.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

# ==========================================
# 1️⃣ Paths for firmware outputs
# ==========================================
FIRMWARE_INC_DIR = "../firmware/include"
FIRMWARE_LUT_DIR = "../firmware/lut"

os.makedirs(FIRMWARE_INC_DIR, exist_ok=True)
os.makedirs(FIRMWARE_LUT_DIR, exist_ok=True)

# ==========================================
# 2️⃣ TinyCatDogNet Model (3 conv layers)
# ==========================================
class TinyCatDogNet(nn.Module):
    def __init__(self):
        super(TinyCatDogNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(32 * 4 * 4, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32*4*4)
        x = self.classifier(x)
        return x

# ==========================================
# 3️⃣ CIFAR-10 Cat/Dog dataloaders
# ==========================================
def get_cat_dog_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    def filter_dataset(dataset):
        indices = [i for i, label in enumerate(dataset.targets) if label in [3,5]]  # 3=Cat,5=Dog
        dataset.targets = [0 if dataset.targets[i]==3 else 1 for i in indices]
        dataset.data = dataset.data[indices]
        return dataset

    trainset = filter_dataset(trainset)
    testset = filter_dataset(testset)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)  # small batch for test images

    return trainloader, testloader

# ==========================================
# 4️⃣ Export functions: weights and LUTs
# ==========================================
def quantize_to_int8(tensor):
    max_val = tensor.abs().max().item()
    scale = 127.0 / max_val if max_val != 0 else 1.0
    quantized = torch.round(tensor * scale).clamp(-128,127).to(torch.int8)
    return quantized, scale

def export_weights_h(model, filepath):
    with open(filepath,'w') as f:
        f.write("// [GENERATED] INT8 model weights\n")
        f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")

        for name,param in model.named_parameters():
            clean_name = name.replace('.', '_')
            q_param, scale = quantize_to_int8(param.detach().cpu())
            flat_data = q_param.flatten().numpy()
            f.write(f"// Layer: {name} | Scale: {scale:.6f}\n")
            f.write(f"const int8_t {clean_name}[{len(flat_data)}] = {{\n    ")
            for i,val in enumerate(flat_data):
                f.write(f"{val}, ")
                if (i+1)%16==0:
                    f.write("\n    ")
            f.write("\n};\n\n")
        f.write("#endif // WEIGHTS_H\n")
    print(f"[*] Saved weights to {filepath}")

def export_test_images_h(testloader, filepath):
    # Historical export path kept for reference; current Renode evaluation uses
    # scripts/export_eval_dataset.py to build an external image blob instead.
    images, labels = next(iter(testloader))
    with open(filepath,'w') as f:
        f.write("// [GENERATED] Test images as C arrays\n")
        f.write("#ifndef TEST_IMAGES_H\n#define TEST_IMAGES_H\n\n")
        f.write("#include <stdint.h>\n\n")
        for i in range(10):
            img_q8 = (images[i].flatten() * 256).to(torch.int16).numpy()
            label_name = "Dog" if labels[i]==1 else "Cat"
            f.write(f"// Image {i}: {label_name}\n")
            f.write(f"const int16_t test_img_{i}[1024] = {{\n    ")
            for j,val in enumerate(img_q8):
                f.write(f"{val}, ")
                if (j+1)%16==0:
                    f.write("\n    ")
            f.write("\n};\n")
            f.write(f"const int expected_label_{i} = {labels[i].item()};\n\n")
        f.write("#endif // TEST_IMAGES_H\n")
    print(f"[*] Saved test images to {filepath}")

def export_luts(out_dir):
    entries = 1024
    x_min, x_max = -8.0, 8.0
    x = np.linspace(x_min, x_max, entries)
    relu = np.maximum(0,x)
    sigmoid = 1/(1+np.exp(-x))
    relu_q8 = np.clip(np.round(relu*256),-32768,32767).astype(np.int16)
    sigmoid_q8 = np.clip(np.round(sigmoid*256),-32768,32767).astype(np.int16)

    def write_lut(filename, array_name, data):
        filepath = os.path.join(out_dir, filename)
        with open(filepath,'w') as f:
            f.write(f"// [GENERATED] {array_name} LUT (1024 entries, Q8.8)\n")
            f.write(f"// Range: {x_min} to {x_max}\n")
            f.write(f"#ifndef {array_name.upper()}_H\n#define {array_name.upper()}_H\n\n")
            f.write("#include <stdint.h>\n\n")
            f.write(f"const int16_t {array_name}[{entries}] = {{\n    ")
            for i,val in enumerate(data):
                f.write(f"{val}, ")
                if (i+1)%16==0:
                    f.write("\n    ")
            f.write("\n};\n\n")
            f.write(f"#endif // {array_name.upper()}_H\n")
        print(f"[*] Saved {filename} to {filepath}")

    write_lut("relu_lut.h","relu_lut",relu_q8)
    write_lut("sigmoid_lut.h","sigmoid_lut",sigmoid_q8)

# ==========================================
# 5️⃣ Main Training + Export
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    net = TinyCatDogNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    trainloader, testloader = get_cat_dog_dataloaders()
    epochs = 100  # max epochs for good accuracy (~75-80%)
    best_acc = 0.0

    print("[*] Starting Training...")
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data,1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
        acc = 100*correct/total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(trainloader):.3f} - Val Accuracy: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(),"best_catdog.pth")
            print(f"    --> Saved new best_catdog.pth (Accuracy: {best_acc:.2f}%)")

    print(f"[*] Training finished. Best Validation Accuracy: {best_acc:.2f}%")

    # Export firmware files
    print("\n[*] Generating Firmware Files...")
    net.load_state_dict(torch.load("best_catdog.pth"))
    net.eval()
    export_weights_h(net, os.path.join(FIRMWARE_INC_DIR,"weights.h"))
    export_luts(FIRMWARE_LUT_DIR)
    print("[*] Renode evaluation images are exported separately via scripts/export_eval_dataset.py")
    print("[*] All tasks completed successfully!")

if __name__ == "__main__":
    main()
