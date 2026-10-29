import os
from src.ingestion.dataloader import load_csv
from src.retrospective.elo import F1EloEngine

def main():
    print("Starting F1 Data Loader & Elo Rating Engine Test...")
    
    file_name = "circuits.csv"
    try:
        print(f"Fetching and loading {file_name}...")
        df = load_csv(file_name)
        print("\n[SUCCESS] Data loaded successfully!")
        print(f"Dataset Shape: {df.shape}")
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        return

    print("\n--------------------------------------------------")
    print("Initializing F1 Elo Rating Engine...")
    
    elo_engine = F1EloEngine(base_rating=1000, k_factor=32, car_weight=0.55)
    
    try:
        print("Processing all available seasons race-by-race...")
        seasons_df = load_csv("seasons.csv")
        for year in sorted(seasons_df['year'].unique()):
            elo_engine.process_season(year)
        
        # Define the elo_results directory path
        elo_results_dir = os.path.join("data", "elo_results")
        
        # Export top driver history and everyone's individual Elo rating history
        elo_engine.export_highest_rated_history(os.path.join(elo_results_dir, "highest_rated_drivers.csv"))
        elo_engine.export_all_drivers_history(os.path.join(elo_results_dir, "all_drivers_elo_history.csv"))
        
        # Load drivers mapping to display names
        drivers_df = load_csv("drivers.csv")
        driver_names = {
            row['driverId']: f"{row['forename']} {row['surname']}" 
            for _, row in drivers_df.iterrows()
        }
        
        print("\n--- Top 10 Drivers by All-Time Elo Rating ---")
        sorted_drivers = sorted(
            elo_engine.driver_ratings.items(), 
            key=lambda item: item[1], 
            reverse=True
        )[:10]
        
        for rank, (d_id, rating) in enumerate(sorted_drivers, 1):
            name = driver_names.get(d_id, f"Driver {d_id}")
            print(f"{rank}. {name}: {rating:.2f}")
            
    except Exception as e:
        print(f"\n[ERROR] Failed during Elo processing: {e}")

if __name__ == "__main__":
    main()