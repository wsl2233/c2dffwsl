import sys
print("=" * 60)
print("C2DFF-Net Environment Check")
print("=" * 60)
print(f"Python Version: {sys.version}")
print()

# Check basic packages
required_packages = [
    ('numpy', 'NumPy'),
    ('matplotlib', 'Matplotlib'),
    ('PIL', 'Pillow'),
]

print("Checking basic packages:")
for pkg, name in required_packages:
    try:
        module = __import__(pkg)
        print(f"[OK] {name} installed (version: {getattr(module, '__version__', 'unknown')})")
    except ImportError:
        print(f"[FAIL] {name} not installed")

print()

# Check deep learning packages
print("Checking deep learning packages:")
deep_learning_packages = [
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'),
    ('ultralytics', 'Ultralytics YOLO'),
    ('cv2', 'OpenCV'),
]

for pkg, name in deep_learning_packages:
    try:
        module = __import__(pkg)
        print(f"[OK] {name} installed (version: {getattr(module, '__version__', 'unknown')})")
    except ImportError:
        print(f"[FAIL] {name} not installed")

print()
print("=" * 60)

# Check if PyTorch has CUDA support
try:
    import torch
    print("PyTorch CUDA Info:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        print(f"  Current GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    pass

print("=" * 60)
print("Environment check completed!")
print("=" * 60)