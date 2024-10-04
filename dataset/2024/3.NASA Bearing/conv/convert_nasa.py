import os
import sys
from datetime import datetime

def get_datetime_from_filename(filename):
    parts = filename.split('.')

    # Fill with '00' if seconds part is missing
    if len(parts) == 5:
        parts.append('00')

    date_str = '.'.join(parts)

    try:
        return datetime.strptime(date_str, "%Y.%m.%d.%H.%M.%S")
    except ValueError:
        print(f"Warning: Unable to parse date from filename: {filename}")
        return None

def process_files(directory, column_prefix):
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    data_files = [f for f in all_files if get_datetime_from_filename(f) is not None]
    sorted_files = sorted(data_files, key=lambda f: get_datetime_from_filename(f) or datetime.min)

    for file in sorted_files:
        file_datetime = get_datetime_from_filename(file)
        epoch_time = int(file_datetime.timestamp()) if file_datetime else "Not available"
        epoch_time = epoch_time * 1000000000

        file_path = os.path.join(directory, file)
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as datafile:
                # Read data rows and output (only output up to 5 rows)
                for row_num, line in enumerate(datafile, 1):
                    row = line.strip().split('\t')
                    for i, value in enumerate(row, 1):
                        print(f"{column_prefix}{i},{epoch_time},{value}")

                    epoch_time = epoch_time + 48828

        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <column_prefix> [directory_path]")
        sys.exit(1)

    column_prefix = sys.argv[1]
    directory_path = sys.argv[2] if len(sys.argv) > 2 else "."

    process_files(directory_path, column_prefix)
