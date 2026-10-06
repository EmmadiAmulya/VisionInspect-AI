import os
import subprocess
import sys

categories = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper"
]

python_exe = os.path.join("venv", "Scripts", "python.exe")

for cat in categories:
    print(f"==========================================")
    print(f"Training category: {cat}")
    print(f"==========================================")
    
    # Run the unified training script for the category, including calibration
    cmd = [python_exe, "ml/train_category.py", "--category", cat, "--calibrate"]
    
    # We pipe stdout and stderr so they appear in the task logs
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    
    if process.returncode != 0:
        print(f"Error: Training failed for {cat} with exit code {process.returncode}")
    else:
        print(f"Successfully trained {cat}")

print("All categories completed!")
