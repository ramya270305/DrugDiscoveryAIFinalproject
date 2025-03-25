import csv
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

def log_prediction(prediction_data):
    """
    Log prediction data to a CSV file.
    
    Args:
        prediction_data (dict): Dictionary containing prediction results
    """
    log_file = 'data/logs.csv'
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(log_file)
    
    try:
        with open(log_file, 'a', newline='') as csvfile:
            fieldnames = [
                'timestamp', 'smiles', 'molecular_weight', 'logP', 
                'num_hydrogen_donors', 'num_hydrogen_acceptors',
                'molar_refractivity', 'topological_polar_surface_area',
                'toxicity_probability', 'toxicity_class', 
                'lipinski_violations', 'druglikeness_score'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(prediction_data)
        
        logger.info(f"Prediction logged successfully at {datetime.now()}")
    
    except Exception as e:
        logger.error(f"Error logging prediction: {str(e)}")

def get_prediction_history():
    """
    Retrieve prediction history from the log file.
    
    Returns:
        list: List of dictionaries containing prediction data
    """
    log_file = 'data/logs.csv'
    history = []
    
    if not os.path.isfile(log_file):
        return history
    
    try:
        with open(log_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                history.append(row)
        
        logger.info(f"Retrieved {len(history)} prediction records")
    
    except Exception as e:
        logger.error(f"Error retrieving prediction history: {str(e)}")
    
    return history
