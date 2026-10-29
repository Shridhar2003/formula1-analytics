import os
import pandas as pd
from typing import List, Dict
from src.ingestion.dataloader import load_csv

class F1EloEngine:
    def __init__(self, base_rating: float = 1000, k_factor: float = 32, car_weight: float = 0.55) -> None:
        self.base_rating = base_rating
        self.k_factor = k_factor
        self.car_weight = car_weight
        
        self.driver_ratings: Dict[int, float] = {}
        self.constructor_ratings: Dict[int, float] = {}
        
        # History logs
        self.highest_rated_history: List[Dict[str, any]] = []
        self.all_drivers_history: List[Dict[str, any]] = []

    def get_driver_rating(self, driver_id: int) -> float:
        return self.driver_ratings.get(driver_id, self.base_rating)

    def get_constructor_rating(self, constructor_id: int) -> float:
        return self.constructor_ratings.get(constructor_id, self.base_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def process_season(self, year: int) -> None:
        print(f"Processing Elo ratings for the {year} season...")
        
        races_df = load_csv("races.csv")
        results_df = load_csv("results.csv")
        
        season_races = races_df[races_df['year'] == year].sort_values('round')
        
        for _, race in season_races.iterrows():
            race_id = race['raceId']
            race_results = results_df[results_df['raceId'] == race_id]
            
            if race_results.empty:
                continue
                
            self._process_race(race_id, race_results)

    def _process_race(self, race_id: int, race_results: pd.DataFrame) -> None:
        valid_results = race_results.copy()
        valid_results['positionOrder'] = pd.to_numeric(valid_results['positionOrder'], errors='coerce')
        valid_results = valid_results.sort_values('positionOrder')
        
        drivers_in_race = valid_results.to_dict('records')
        n = len(drivers_in_race)
        
        if n < 2:
            return

        driver_deltas = {d['driverId']: 0.0 for d in drivers_in_race}
        constructor_deltas = {}

        for i in range(n):
            for j in range(i + 1, n):
                driver_a = drivers_in_race[i]
                driver_b = drivers_in_race[j]
                
                id_a, const_a = driver_a['driverId'], driver_a['constructorId']
                id_b, const_b = driver_b['driverId'], driver_b['constructorId']
                
                d_rat_a = self.get_driver_rating(id_a)
                d_rat_b = self.get_driver_rating(id_b)
                c_rat_a = self.get_constructor_rating(const_a)
                c_rat_b = self.get_constructor_rating(const_b)
                
                total_a = d_rat_a + (self.car_weight * c_rat_a)
                total_b = d_rat_b + (self.car_weight * c_rat_b)
                
                exp_a = self.expected_score(total_b, total_a)
                exp_b = 1 - exp_a
                
                weight_multiplier = 1.5 if const_a == const_b else 1.0
                
                delta_a = self.k_factor * weight_multiplier * (1.0 - exp_a)
                delta_b = self.k_factor * weight_multiplier * (0.0 - exp_b)
                
                driver_deltas[id_a] += delta_a / (n - 1)
                driver_deltas[id_b] += delta_b / (n - 1)
                
                constructor_deltas[const_a] = constructor_deltas.get(const_a, 0.0) + (delta_a / (n - 1))
                constructor_deltas[const_b] = constructor_deltas.get(const_b, 0.0) + (delta_b / (n - 1))

        # Apply rating updates
        for d_id, delta in driver_deltas.items():
            self.driver_ratings[d_id] = self.get_driver_rating(d_id) + delta
            
        for c_id, delta in constructor_deltas.items():
            self.constructor_ratings[c_id] = self.get_constructor_rating(c_id) + delta

        # Log highest rated driver
        if self.driver_ratings:
            top_driver_id = max(self.driver_ratings, key=self.driver_ratings.get)
            top_rating = self.driver_ratings[top_driver_id]
            
            self.highest_rated_history.append({
                "raceId": race_id,
                "highest_rated_driver": top_driver_id,
                "elo_rating": round(top_rating, 2)
            })

        # Log all drivers' ratings snapshot
        for d_id, rating in self.driver_ratings.items():
            self.all_drivers_history.append({
                "raceId": race_id,
                "driverId": d_id,
                "elo_rating": round(rating, 2)
            })

    def export_highest_rated_history(self, output_path: str) -> None:
        if not self.highest_rated_history:
            return
        df = pd.DataFrame(self.highest_rated_history)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Exported highest rated driver history to: {output_path}")

    def export_all_drivers_history(self, output_path: str) -> None:
        if not self.all_drivers_history:
            print("No driver history to export.")
            return
        df = pd.DataFrame(self.all_drivers_history)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Exported all drivers Elo history to: {output_path}")