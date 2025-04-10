#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAR Dataset Converter
This script converts the UCI HAR Dataset to a format with timestamps 
and combined accelerometer/gyroscope data
"""

import pandas as pd
import numpy as np
import os

def convert_har_dataset(base_path):
    """
    Convert the UCI HAR Dataset to the format with timestamps and combined accelerometer/gyroscope data

    Args:
        base_path: Path to the UCI HAR Dataset folder

    Returns:
        DataFrame in the desired format
    """
    print(f"Loading data from {base_path}...")

    # Load training data
    # Subject IDs
    train_subjects = pd.read_csv(os.path.join(base_path, 'train', 'subject_train.txt'),
                                header=None, names=['subject_id'])

    # Activity labels
    train_y = pd.read_csv(os.path.join(base_path, 'train', 'y_train.txt'),
                         header=None, names=['activity_id'])

    # Load raw inertial signals for training set
    train_acc_x = pd.read_csv(os.path.join(base_path, 'train', 'Inertial Signals', 'total_acc_x_train.txt'),
                             header=None, delim_whitespace=True)
    train_acc_y = pd.read_csv(os.path.join(base_path, 'train', 'Inertial Signals', 'total_acc_y_train.txt'),
                             header=None, delim_whitespace=True)
    train_acc_z = pd.read_csv(os.path.join(base_path, 'train', 'Inertial Signals', 'total_acc_z_train.txt'),
                             header=None, delim_whitespace=True)

    train_gyro_x = pd.read_csv(os.path.join(base_path, 'train', 'Inertial Signals', 'body_gyro_x_train.txt'),
                              header=None, delim_whitespace=True)
    train_gyro_y = pd.read_csv(os.path.join(base_path, 'train', 'Inertial Signals', 'body_gyro_y_train.txt'),
                              header=None, delim_whitespace=True)
    train_gyro_z = pd.read_csv(os.path.join(base_path, 'train', 'Inertial Signals', 'body_gyro_z_train.txt'),
                              header=None, delim_whitespace=True)

    print("Loading test data...")

    # Load test data
    # Subject IDs
    test_subjects = pd.read_csv(os.path.join(base_path, 'test', 'subject_test.txt'),
                               header=None, names=['subject_id'])

    # Activity labels
    test_y = pd.read_csv(os.path.join(base_path, 'test', 'y_test.txt'),
                        header=None, names=['activity_id'])

    # Load raw inertial signals for test set
    test_acc_x = pd.read_csv(os.path.join(base_path, 'test', 'Inertial Signals', 'total_acc_x_test.txt'),
                            header=None, delim_whitespace=True)
    test_acc_y = pd.read_csv(os.path.join(base_path, 'test', 'Inertial Signals', 'total_acc_y_test.txt'),
                            header=None, delim_whitespace=True)
    test_acc_z = pd.read_csv(os.path.join(base_path, 'test', 'Inertial Signals', 'total_acc_z_test.txt'),
                            header=None, delim_whitespace=True)

    test_gyro_x = pd.read_csv(os.path.join(base_path, 'test', 'Inertial Signals', 'body_gyro_x_test.txt'),
                             header=None, delim_whitespace=True)
    test_gyro_y = pd.read_csv(os.path.join(base_path, 'test', 'Inertial Signals', 'body_gyro_y_test.txt'),
                             header=None, delim_whitespace=True)
    test_gyro_z = pd.read_csv(os.path.join(base_path, 'test', 'Inertial Signals', 'body_gyro_z_test.txt'),
                             header=None, delim_whitespace=True)

    print("Transforming data...")

    # Combine train and test data
    subjects = pd.concat([train_subjects, test_subjects], ignore_index=True)
    activities = pd.concat([train_y, test_y], ignore_index=True)

    all_data = []

    sampling_rate = 50  # Hz as mentioned in documentation
    time_between_samples = 1 / sampling_rate  # in seconds

    def process_dataset(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, subjects_df, activities_df, start_idx=0):
        data_points = []

        for i in range(len(subjects_df)):
            subject_id = subjects_df.iloc[i]['subject_id']
            activity_id = activities_df.iloc[i]['activity_id']

            # Get the window of 128 readings for this observation
            for j in range(128):  # 128 readings per window
                # Calculate timestamp (approximation)
                # We are just making sure each window is unique and we will do the windowing, when we run the script for training data window
                timestamp = (start_idx + i) * 2.56 + j * time_between_samples

                data_point = {
                    'time_stamp': round(timestamp, 2),
                    'activity_id': activity_id,
                    'id': subject_id,
                    'chest_3D_acceleration_16_x': acc_x.iloc[i, j],
                    'chest_3D_acceleration_16_y': acc_y.iloc[i, j],
                    'chest_3D_acceleration_16_z': acc_z.iloc[i, j],
                    'chest_3D_gyroscope_x': gyro_x.iloc[i, j],
                    'chest_3D_gyroscope_y': gyro_y.iloc[i, j],
                    'chest_3D_gyroscope_z': gyro_z.iloc[i, j]
                }
                data_points.append(data_point)

        return data_points

    # Process train data
    print("Processing train data...")
    train_data_points = process_dataset(train_acc_x, train_acc_y, train_acc_z,
                                        train_gyro_x, train_gyro_y, train_gyro_z,
                                        train_subjects, train_y)

    # Process test data
    print("Processing test data...")
    test_data_points = process_dataset(test_acc_x, test_acc_y, test_acc_z,
                                       test_gyro_x, test_gyro_y, test_gyro_z,
                                       test_subjects, test_y, start_idx=len(train_subjects))

    # Confirming lengths before merging
    print(f"Number of training data points: ({len(train_data_points)}, {len(train_data_points[0])})")
    print(f"Number of test data points: {len(test_data_points)}, {len(test_data_points[0])}")

    # Combine all data points
    all_data = train_data_points + test_data_points

    print(f"Total number of data points after merging: {len(all_data)}, {len(all_data[0])}")

    # Convert to DataFrame
    print("Creating DataFrame...")
    output_df = pd.DataFrame(all_data)
    return output_df


def main():
    """Main function to execute the HAR dataset conversion"""
    # Use local path for the dataset
    base_path = './UCI HAR Dataset'  # Dataset folder expected in current directory
    
    # Check if the path exists
    if not os.path.exists(base_path):
        print(f"ERROR: Dataset not found at {base_path}")
        print("Please make sure the UCI HAR Dataset folder is in the current directory")
        return
    
    # Convert the dataset
    output_df = convert_har_dataset(base_path)
    
    # Save to CSV in the current directory
    output_path = './har_data_reformatted.csv'
    output_df.to_csv(output_path, index=False)
    print(f"Converted data saved to {output_path}")
    print(f"Sample of converted data:\n{output_df.head()}")


if __name__ == "__main__":
    main()