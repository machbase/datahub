import csv
import datetime
import os

def process_csv_files(file_name, base_time):
    results = []

    for file_name in file_list:
        # Get the part of the file name without the extension
        file_prefix = os.path.splitext(file_name)[0]

        # Dynamically create column names
        columns = ['time', f'{file_prefix}_normal', f'{file_prefix}_type1', f'{file_prefix}_type2', f'{file_prefix}_type3']

        # Read the CSV file
        with open(file_name, 'r') as csv_file:
            reader = csv.reader(csv_file)

            # Process each row
            for row in reader:
                # Convert the time value to float and add it to base_time
                time = int(base_time + (float(row[0]) * 1000000000))

                # Process each column
                for i, value in enumerate(row[1:], start=1):
                    results.append(f"{columns[i]},{time},{value}")

    return results

# List of files to process
file_list = ['g1_sensor1.csv', 'g1_sensor2.csv', 'g1_sensor3.csv', 'g1_sensor4.csv', 'g2_sensor1.csv', 'g2_sensor2.csv', 'g2_sensor3.csv', 'g2_sensor4.csv']  # Update with actual file names

# Process files and get results
epoch_2024 = int(datetime.datetime(2024, 1, 1).timestamp())
base_time = epoch_2024 * 1000000000  # Multiply by 1 billion (convert to nano)
for file_name in file_list:
    output = process_csv_files(file_name, base_time)

# Print results
print("NAME,TIME,VALUE")
for line in output:
    print(line)