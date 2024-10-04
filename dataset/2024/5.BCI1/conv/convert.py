import sys
from datetime import datetime, timedelta

def parse_and_print_files(sensor_prefix, filename1, filename2=None):
    # Set the starting time to January 1, 2024
    current_time = int(datetime(2024, 1, 1).timestamp() * 1000000000)

    with open(filename1, 'r') as file1:
        file2 = open(filename2, 'r') if filename2 else None
        batch_count = 0
        try:
            while True:
                batch = []

                # Read 64 lines
                batch_time = current_time
                # If the second file is provided, read and print from that file
                if file2:
                    second_file_line = file2.readline().strip()
                    if second_file_line:
                        print(f"{sensor_prefix}-answer,{batch_time},{second_file_line}")
                        # print(f"Line {batch_count} from the second file: {second_file_line}")
                    else:
                        # print(f"Reached the end of the second file.")
                        file2.close()
                        file2 = None

                for _ in range(64):
                    line = file1.readline().strip()
                    if not line:
                        # Exit if end of file is reached
                        if not batch:
                            return
                        break
                    # Convert each line into a list of numbers
                    numbers = [float(x) for x in line.split()]
                    if len(numbers) != 3000:
                        print(f"Error: Line does not contain 3000 numbers. Actual count: {len(numbers)}")
                        return
                    batch.append(numbers)

                # Output the batch
                # print(f"Batch {batch_count + 1}")
                for i in range(3000):
                    for sensor in range(len(batch)):
                        if i < len(batch[sensor]):
                            # Convert epoch time to nanoseconds (multiply by 1 billion)
                            print(f"{sensor_prefix}-{sensor},{current_time},{batch[sensor][i]:.7f}")
                    current_time += 1000000

                current_time += 57 * 1000000000
                batch_count += 1
                # print(f"Batch {batch_count} completed")

        finally:
            if file2:
                file2.close()

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python script.py <sensor_prefix> <source> [answer]")
        sys.exit(1)
    sensor_prefix = sys.argv[1]
    filename1 = sys.argv[2]
    filename2 = sys.argv[3] if len(sys.argv) == 4 else None
    parse_and_print_files(sensor_prefix, filename1, filename2)