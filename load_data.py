import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv

def load_csv_to_postgres():
    # Load environment variables from .env file
    load_dotenv()

    # 1. Database Connection Configuration
    # Replace 'YOUR_PASSWORD' with your actual pgAdmin password
    db_password = os.environ.get('DB_PASSWORD')
    if not db_password:
        raise ValueError("DB_PASSWORD environment variable is missing. Please add it to your .env file.")
        
    encoded_password = quote_plus(db_password)
    
    # Create the connection engine
    engine = create_engine(f'postgresql://postgres:{encoded_password}@localhost:5432/postgres')

    # 2. Load the data using your exact Windows file path
    # The 'r' before the string tells Python to read the backslashes literally
    csv_file_path = r"C:\Users\Vedav\OneDrive\Desktop\veda AI\EV_Predictive_Maintenance_Dataset_15min.csv" 
    print(f"Loading data from: {csv_file_path}...")
    
    try:
        # Read the CSV file into memory
        df = pd.read_csv(csv_file_path)
        
        # Clean column names (lowercase, no spaces)
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        
        # 3. Push the data to the PostgreSQL database
        print("Pushing data to the database... This might take a minute.")
        df.to_sql('vehicle_telemetry', engine, if_exists='replace', index=False)
        
        print("Success! The telemetry data is now live in your PostgreSQL database.")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file at {csv_file_path}.")
    except Exception as e:
         print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    load_csv_to_postgres()

 