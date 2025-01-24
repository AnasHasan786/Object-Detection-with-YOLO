import os
from datetime import datetime

def create_output_dirs(base_dir="../outputs"): 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dirs = {
        "images": os.path.join(base_dir, f"images"),
        "visualizations": os.path.join(base_dir, f"visualizations"),
        "models": os.path.join(base_dir, f"models"),
    }

    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)

    return output_dirs

if __name__ == "__main__":
    output_dirs = create_output_dirs()
    print("Output directories created:")
    for name, path in output_dirs.items():
        print(f"{name}: {path}")
