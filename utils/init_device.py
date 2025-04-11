import torch
import sys

def init_cuda():
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.cuda.set_device(0)
            torch.cuda.init()
            print(f"CUDA Initialized. Device: {torch.cuda.get_device_name(0)}")
            return device
        else:
            print("CUDA not available. Using CPU.")
            return torch.device('cpu')
    except Exception as e:
        print(f"CUDA initialization error: {e}")
        sys.exit(1)