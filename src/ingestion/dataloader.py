import os
import shutil
import kagglehub
import pandas as pd

# Define your project's local data directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

def download_f1_data() -> str:
    """
    Downloads the F1 dataset via kagglehub and ensures
    it's available locally inside the project's data directory.
    """
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Fetching F1 dataset...")
    # Download/locate dataset via cache
    cache_path = kagglehub.dataset_download("atharvranjan/formula-1-world-championship-1950-present")

    # Optional: Copy files to your local project directory if you want them explicitly there
    # ( kagglehub caches them, but copying ensures they live in your 'src/data' folder)
    for filename in os.listdir(cache_path):
        src_file = os.path.join(cache_path, filename)
        dst_file = os.path.join(DATA_DIR, filename)
        if os.path.isfile(src_file) and not os.path.exists(dst_file):
            shutil.copy(src_file, dst_file)

    print(f"Dataset ready at: {DATA_DIR}")
    return DATA_DIR

def load_csv(file_name: str) -> pd.DataFrame:
    """Helper to load any F1 CSV file directly into a pandas DataFrame."""
    data_path = download_f1_data()
    file_path = os.path.join(data_path, file_name)
    return pd.read_csv(file_path)
