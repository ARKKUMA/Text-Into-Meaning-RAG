import os
import subprocess
import platform

def has_nvidia_gpu():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def main():
    if platform.system() == "Darwin":
        cmd = "pip install torch torchvision torchaudio"
    elif has_nvidia_gpu():
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        
    os.system(cmd)
    
    if os.path.exists("requirements.txt"):
        os.system("pip install -r requirements.txt")

if __name__ == "__main__":
    main()