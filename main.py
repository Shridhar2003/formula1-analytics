from src.ingestion.dataloader import load_csv

def main():
    print("Starting F1 Data Loader Test...")
    
    # Test loading a core dataset file, e.g., circuits.csv
    file_name = "circuits.csv"
    
    try:
        print(f"Fetching and loading {file_name}...")
        df = load_csv(file_name)
        
        print("\n[SUCCESS] Data loaded successfully!")
        print(f"Dataset Shape: {df.shape}")
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")

if __name__ == "__main__":
    main()

