#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensor Data Processing Script
This script processes HAR and PAMAP2 datasets, creating windowed observation data
and saving the results as NetCDF files.
"""

import xarray as xr
import numpy as np
import pandas as pd

def main():
    # Define file paths for both datasets
    har_file_path = "./har_data_reformatted.csv"
    pamap_file_path = "./PAMAP2_D50.csv"

    # Choose which file to use (change this to switch between datasets)
    use_pamap = False  # Set to True for PAMAP2, False for HAR
    file_path = pamap_file_path if use_pamap else har_file_path

    # Load dataset
    print(f"Loading {'PAMAP2' if use_pamap else 'HAR'} dataset from: {file_path}")
    df = pd.read_csv(file_path)

    # Apply transformation only for PAMAP2 data
    if use_pamap:
        print("PAMAP2 dataset detected, applying activity ID transformation")
        print("Before mapping:")
        print(df["activity_id"].value_counts().sort_index())
        
        # Add 100 to all activity ids to avoid map clash
        activity_aligner = 100
        df["activity_id"] += activity_aligner
        
        print("During mapping:")
        print(df["activity_id"].value_counts().sort_index())
        
        # Define activity mapping for PAMAP2
        activity_map = {
            101: 6,  # LAYING
            102: 4,  # SITTING
            103: 5,  # STANDING
            104: 1,  # WALKING
            112: 2,  # WALKING_UPSTAIRS
            113: 3   # WALKING_DOWNSTAIRS
        }
        
        # Remap concerned activity IDs
        df["activity_id"] = df["activity_id"].apply(lambda x: activity_map[x] if x in activity_map else x)
        
        print("After mapping:")
        print(df["activity_id"].value_counts().sort_index())
    else:
        print("HAR dataset detected, no transformation needed")
        print("Activity IDs:")
        print(df["activity_id"].value_counts().sort_index())

    # Define window size
    WINDOW_SIZE = 128

    # Define sensor columns
    sensor_columns = [col for col in df.columns if col not in ["time_stamp", "activity_id", "id"]]

    # Create observation windows
    windows, participant_ids = create_observations(df, WINDOW_SIZE)

    # Convert to xarray format
    obs_count = len(windows)
    coords = {
        "observation_number": np.arange(obs_count),
        "timestamp": np.arange(WINDOW_SIZE),
        "participant_id": ("observation_number", np.array(participant_ids))
    }

    data_vars = {
        "activity_id": (
            ["observation_number", "timestamp"],
            np.array([w["activity_id"].values for w in windows])
        )
    }

    # Add sensor data
    for sensor in sensor_columns:
        data_vars[sensor] = (
            ["observation_number", "timestamp"],
            np.array([w[sensor].values for w in windows])
        )

    # Create xarray dataset
    dataset = xr.Dataset(data_vars, coords=coords)

    # Process data and create final dataset
    final_dataset = add_majority_activity(dataset)

    activity_label_map = {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING"
    }

    # Add metadata
    final_dataset.attrs = {
        "date_created": pd.Timestamp.now().isoformat(),
        "input_data_source": "pamap_windowed.csv" if use_pamap else "har_windowed.csv",
        "team_name": "CS67-2",
        "description": "Windowed sensor data with majority activity labels",
        "window_size": WINDOW_SIZE,
        "activity_label_map": str(activity_label_map)
    }

    # Count the number of occurrences of each majority activity
    activity_counts = final_dataset.majority_activity.to_series().value_counts().sort_index()

    # Display the activity counts
    print("Activity counts (after majority filtering):")
    print(activity_counts)

    # Save dataset as NetCDF with appropriate filename
    output_filename = "pamap_windowed.nc" if use_pamap else "har_windowed.nc"
    final_dataset.to_netcdf(output_filename)
    print(f"Dataset saved as: {output_filename}")

    # Display dataset info
    print(final_dataset)
    
    return final_dataset


def create_observations(df, window_size):
    """
    Create observation windows from time series data with 50% overlap
    
    Args:
        df: Pandas DataFrame containing sensor data
        window_size: Size of each window in number of samples
        
    Returns:
        tuple: (list of window DataFrames, list of participant IDs)
    """
    observations = []
    participant_ids = []  # Track participant IDs
    step_size = window_size // 2  # 50% overlap

    for participant_id, group in df.groupby("id"):
        group = group.sort_values("time_stamp")  # Ensure chronological order

        for i in range(0, len(group) - window_size + 1, step_size):
            window = group.iloc[i : i + window_size]
            if len(window) == window_size:
                observations.append(window)
                participant_ids.append(participant_id)

    print(f"\nTotal windows created: {len(observations)}")
    return observations, participant_ids


def add_majority_activity(dataset, threshold=0.5, valid_activities=None):
    """
    Calculate majority activity for each observation window
    
    Args:
        dataset: xarray Dataset containing windowed observations
        threshold: Minimum fraction of window that must contain the majority activity
        valid_activities: List of valid activity IDs to include
        
    Returns:
        xarray Dataset: Filtered dataset with majority activity coordinates
    """
    if valid_activities is None:
        valid_activities = [1, 2, 3, 4, 5, 6]

    majority_activities = []
    valid_observations = []
    window_size = dataset.dims['timestamp']

    for obs in range(dataset.dims['observation_number']):
        activity_values = dataset.activity_id.sel(observation_number=obs).values
        unique, counts = np.unique(activity_values, return_counts=True)
        majority_activity = unique[np.argmax(counts)]

        if (counts.max() / window_size) >= threshold and majority_activity in valid_activities:
            majority_activities.append(majority_activity)
            valid_observations.append(obs)

    filtered_dataset = dataset.isel(observation_number=valid_observations)
    filtered_dataset = filtered_dataset.assign_coords(
        majority_activity=("observation_number", np.array(majority_activities))
    )
    return filtered_dataset


if __name__ == "__main__":
    main()