import csv

def soft_remove_columns(input_csv_path, output_csv_path, columns_to_remove):
    """
    Reads a CSV file, excludes specified columns, and writes the result to a new CSV.

    :param input_csv_path: Path to the original CSV file
    :param output_csv_path: Path to save the filtered CSV file
    :param columns_to_remove: List of column names to exclude
    """
    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        # Determine columns to keep
        columns_to_keep = [col for col in reader.fieldnames if col not in columns_to_remove]

        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=columns_to_keep)
            writer.writeheader()

            for row in reader:
                filtered_row = {col: row[col] for col in columns_to_keep}
                writer.writerow(filtered_row)

if __name__ == "__main__":
    input_file = "data/binary_features_log.csv"     # Replace with your input CSV path
    output_file = "data/no_pitch_train_data.csv"   # Replace with your desired output CSV path
    cols_to_remove = ["pitch_mean", "pitch_std", "rms_std", "zcr_std", "centroid_std","flatness_std", "mfcc1_std", "mfcc2_std", "mfcc3_std", "mfcc4_std", "mfcc5_std", "mfcc6_std", "mfcc7_std", "mfcc8_std", "mfcc9_std", "mfcc10_std", "mfcc11_std","mfcc12_std", "mfcc13_std",]  # Replace with the actual columns you want to exclude

    soft_remove_columns(input_file, output_file, cols_to_remove)
    print(f"Filtered CSV saved to {output_file}")
