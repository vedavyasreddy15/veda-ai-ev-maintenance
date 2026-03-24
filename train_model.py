import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score
import joblib

def train_and_evaluate():
    csv_file_path = r"C:\Users\Vedav\OneDrive\Desktop\veda AI\EV_Predictive_Maintenance_Dataset_15min.csv"
    
    print(f"Loading dataset from {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file_path}")
        return

    # Clean columns to match your database format
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]

    # Define the 6 features the AI agent uses, and the target we want to predict
    features = ['battery_temperature', 'battery_voltage', 'motor_temperature', 'motor_rpm', 'soc', 'soh']
    X = df[features]
    y = df['failure_probability']

    # Split data: 80% for training the model, 20% for testing/evaluating it
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize Random Forest
    # class_weight='balanced' forces the model to heavily penalize missing a failure! (Maximizes Recall)
    print("Training Random Forest model (Optimizing for Recall)...")
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)

    # In our agent.py, we trigger an alert if the risk is > 10%. 
    # Let's evaluate our model using that exact same threshold to maximize Recall!
    print("Evaluating model with a 10% risk threshold...")
    y_probs = rf_model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > 0.10).astype(int)

    # Calculate metrics
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("📊 MODEL EVALUATION METRICS")
    print("="*40)
    print(f"🎯 RECALL:   {recall * 100:.2f}% (Most Important!)")
    print(f"✅ ACCURACY: {accuracy * 100:.2f}%")
    print("="*40)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the trained model so agent.py can use it!
    joblib.dump(rf_model, "rf_model.pkl")
    print("\n🚀 Model successfully saved as rf_model.pkl!")

if __name__ == "__main__":
    train_and_evaluate()