import platform
import os
import subprocess
import torch
import psutil

def cmd(c):
    try:
        return subprocess.check_output(c, shell=True).decode().strip()
    except:
        return "N/A"

print("="*45)
print("EXPERIMENTAL SETUP")
print("="*45)

# OS
print(f"OS              : {platform.system()} {platform.release()}")
print(f"Kernel          : {platform.version()}")

# CPU
print(f"CPU             : {platform.processor()}")
print(f"Physical Cores  : {psutil.cpu_count(logical=False)}")
print(f"Logical Cores   : {psutil.cpu_count(logical=True)}")

# RAM
ram = psutil.virtual_memory().total / (1024 ** 3)
print(f"RAM             : {ram:.2f} GB")

# Python
print(f"Python Version  : {platform.python_version()}")

# Environment
print(f"Virtual Env     : {os.environ.get('CONDA_DEFAULT_ENV', 'venv / system')}")

# PyTorch
print(f"PyTorch         : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")
print(f"MPS Available   : {torch.backends.mps.is_available()}")

if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("GPU             : Apple Silicon (MPS)")
else:
    print("GPU             : CPU only")

print("="*45)
