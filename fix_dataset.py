import pandas as pd

def fix_dataset():
    csv_file_path = r"C:\Users\Vedav\OneDrive\Desktop\veda AI\EV_Predictive_Maintenance_Dataset_15min.csv"
    
    try:
        print(f"Loading {csv_file_path}...")
        df = pd.read_csv(csv_file_path)
        
        # Clean columns just in case
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        
        print("Injecting real-world physics into the dataset...")
        # Logical Failure Conditions: Hot Motor, Dead Battery, or Hot Battery
        condition = (df['motor_temperature'] > 90) | (df['soh'] < 0.60) | (df['battery_temperature'] > 55)
        
        # Overwrite the random failure probabilities with our logical ones
        df['failure_probability'] = condition.astype(int)
        
        print(f"Fixed! Created {df['failure_probability'].sum()} realistic failure scenarios out of {len(df)} rows.")
        
        # Save the fixed data back to the CSV
        df.to_csv(csv_file_path, index=False)
        print("Dataset successfully overwritten with logical data!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fix_dataset()