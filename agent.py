import os
import joblib
import pandas as pd
from urllib.parse import quote_plus
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.tools import tool

# Load the trained Machine Learning model
try:
    ml_model = joblib.load('rf_model.pkl')
except Exception as e:
    print("Warning: Could not load rf_model.pkl. Did you run train_model.py?")
    ml_model = None

@tool
def predict_failure_probability(battery_temperature: float, battery_voltage: float, motor_temperature: float, motor_rpm: float, soc: float, soh: float) -> str:
    """
    Predicts the EV battery failure probability using a Machine Learning model.
    Input requires exact numeric values for: battery_temperature, battery_voltage, motor_temperature, motor_rpm, soc, and soh.
    """
    input_data = pd.DataFrame([[battery_temperature, battery_voltage, motor_temperature, motor_rpm, soc, soh]], 
                              columns=['battery_temperature', 'battery_voltage', 'motor_temperature', 'motor_rpm', 'soc', 'soh'])
    
    # Get the exact probability of failure (class 1 is the second element)
    failure_prob = ml_model.predict_proba(input_data)[0][1]
    
    # Failures are rare! If the statistical probability is higher than 10%, we flag it as high risk!
    return f"HIGH RISK of failure detected! (Risk Score: {failure_prob * 100:.1f}%)" if failure_prob > 0.1 else f"LOW RISK of failure. Vehicle is healthy. (Risk Score: {failure_prob * 100:.1f}%)"

@tool
def send_alert_email(customer_email: str, risk_score: str) -> str:
    """
    Sends an emergency alert email to a customer whose vehicle is at high risk.
    Use this tool when a vehicle is diagnosed with a HIGH RISK of failure, or when the user asks to notify the customer.
    """
    # Mocking the email sending process for the portfolio
    email_body = f"""
    Subject: URGENT: Lucid Vehicle Health Alert
    To: {customer_email}

    Dear Lucid Owner,

    Our Predictive Maintenance AI has detected a critical anomaly in your vehicle's telemetry. 
    Your current failure risk score is {risk_score}. 
    
    Please bring your vehicle to the nearest Lucid Service Center immediately so we can make Lucid, Lucid again.

    Best,
    Veda AI Engineering Team
    """
    print(f"\n[Email Server] Sending email...\n{email_body}\n[Email Server] Email Sent Successfully!\n")
    return f"Alert email successfully sent to {customer_email}."

@tool
def send_bulk_alert_emails(num_vehicles: int, alert_reason: str) -> str:
    """
    Sends bulk emergency alert emails to multiple customers whose vehicles are at high risk.
    Use this tool when the user asks to notify a group of owners or a count of high-risk vehicles.
    """
    email_body = f"""
    Subject: URGENT: Lucid Fleet Health Alert
    
    Dear Customer,
    Our Predictive Maintenance AI has detected a critical anomaly in your vehicle's telemetry.
    Reason: {alert_reason}
    
    Please bring your vehicle to the nearest Lucid Service Center immediately.
    """
    print(f"\n[Email Server] Dispatching {num_vehicles} bulk emails...\n{email_body}\n[Email Server] All Bulk Emails Sent Successfully!\n")
    return f"Bulk alert emails successfully sent to {num_vehicles} customers."

def setup_database_connection():
    # 1. Database Connection Configuration (matching load_data.py)
    db_user = 'postgres'
    db_password = 'Veda@1512'
    encoded_password = quote_plus(db_password)
    db_host = 'localhost'
    db_port = '5432'
    db_name = 'postgres'
    
    # Connection string for SQLAlchemy/LangChain
    db_uri = f"postgresql+psycopg2://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
    
    try:
        # Initialize the LangChain SQLDatabase wrapper
        db = SQLDatabase.from_uri(db_uri)
        print(f"Successfully connected! Available tables for the AI: {db.get_usable_table_names()}")
        return db
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

if __name__ == "__main__":
    db = setup_database_connection()
    if db:
        print("\nInitializing Veda AI Agent...")
        
        # NOTE: Put your PAID Google Gemini API key here!
        os.environ["GOOGLE_API_KEY"] = "AIzaSyCnbGUNv7jjHHBTSp_B_hTD7DdTNG58jDs"

        # Using 2.5 Flash without pacing since you are on the paid tier!
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=5)
        
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="tool-calling",
            extra_tools=[predict_failure_probability, send_alert_email, send_bulk_alert_emails],
            verbose=True # We want to see the AI's "Thought Process" in the terminal!
        )
        
        # Let's test it!
        print("\n--- Testing Veda AI ---")
        question = "Find the average battery_temperature, battery_voltage, motor_temperature, motor_rpm, soc, and soh from the vehicle_telemetry table. Then use those averages to predict the failure probability of a vehicle with those average stats."
        print(f"User: {question}")
        response = agent_executor.invoke({"input": question})
        
        # Clean up Gemini's raw output format
        raw_output = response['output']
        if isinstance(raw_output, list):
            clean_output = "".join([item.get('text', '') if isinstance(item, dict) else str(item) for item in raw_output])
        else:
            clean_output = raw_output
            
        print(f"\nVeda AI: {clean_output}")