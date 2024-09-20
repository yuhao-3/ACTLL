import pandas as pd
import numpy as np

def prepare_eICU():
    path = 'data/eICU/'
    
    # Load time series data
    train_x = pd.read_csv(f"{path}" + "train/timeseries.csv")
    val_x = pd.read_csv(f"{path}" + "val/timeseries.csv")
    test_x = pd.read_csv(f"{path}" + "test/timeseries.csv")

    # Load labels data
    train_y = pd.read_csv(f"{path}" + "train/labels.csv")
    val_y = pd.read_csv(f"{path}" + "val/labels.csv")
    test_y = pd.read_csv(f"{path}" + "test/labels.csv")
    
    # Print shapes of the datasets for debugging
    print(f"Shape of train_x: {train_x.shape}")
    print(f"Shape of val_x: {val_x.shape}")
    print(f"Shape of test_x: {test_x.shape}")
    
    print(f"Shape of train_y: {train_y.shape}")
    print(f"Shape of val_y: {val_y.shape}")
    print(f"Shape of test_y: {test_y.shape}")
    
    # Combine the datasets (train, val, test)
    X = pd.concat([train_x, val_x, test_x], axis=0)
    y = pd.concat([train_y, val_y, test_y], axis=0)
    
    # Extract the 'actualhospitalmortality' column as the label, and keep 'patient' column for matching
    y_outcome = y[['patient', 'actualhospitalmortality']]
    
    # Group by patient_id
    grouped = X.groupby('patient')

    # Initialize lists to store reshaped data and labels
    reshaped_data = []
    reshaped_labels = []
    
    # Counter to track progress
    patient_count = 0
    
    for patient_id, group in grouped:
        # Increment the patient counter
        patient_count += 1
        
        # Print progress every 5000 patients
        if patient_count % 500 == 0:
            print(f"Has processed {patient_count} patients now!")
        
        # Ensure that the number of rows is a multiple of 24
        n_rows = group.shape[0]
        n_24_hour_blocks = n_rows // 24
        
        # Only keep full 24-hour blocks
        if n_24_hour_blocks > 0:
            # Reshape the group into 24-hour blocks with 29 features (excluding 'patient' and 'time')
            reshaped_group = group.iloc[:n_24_hour_blocks * 24, 2:].values.reshape(-1, 24, 29)
            reshaped_data.append(reshaped_group)
            
            # Get the label for the patient and repeat it for each 24-hour block
            patient_label = y_outcome[y_outcome['patient'] == patient_id].iloc[0]['actualhospitalmortality']
            reshaped_labels.append([patient_label] * n_24_hour_blocks)  # Repeat label for each block
    
    # Convert lists into arrays
    X_final = np.concatenate(reshaped_data, axis=0)
    y_final = np.concatenate(reshaped_labels, axis=0)
    
    # Print final shape of the processed data
    print(f"Final shape of X: {X_final.shape}")  # Should be (xx, 24, 29)
    print(f"Final shape of y: {y_final.shape}")
    
    
    return X_final,y_final


if __name__ == "__main__":
    X_final, y_final = prepare_eICU()