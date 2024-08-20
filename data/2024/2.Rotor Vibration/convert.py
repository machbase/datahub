import csv
import datetime
import os

def process_csv_files(file_name, base_time):
    results = []

    for file_name in file_list:
        # 파일 이름에서 확장자를 제외한 부분을 가져옴
        file_prefix = os.path.splitext(file_name)[0]

        # 동적으로 컬럼명 생성
        columns = ['time', f'{file_prefix}_normal', f'{file_prefix}_type1', f'{file_prefix}_type2', f'{file_prefix}_type3']

        # CSV 파일 읽기
        with open(file_name, 'r') as csv_file:
            reader = csv.reader(csv_file)

            # 각 행을 처리
            for row in reader:
                # time 값을 float로 변환하고 base_time에 더함
                time = int(base_time + (float(row[0]) * 1000000))

                # 각 컬럼에 대해 처리
                for i, value in enumerate(row[1:], start=1):
                    results.append(f"{columns[i]},{time},{value}")

    return results

# 처리할 파일 리스트
file_list = ['g1_sensor1.csv', 'g1_sensor2.csv', 'g1_sensor3.csv', 'g1_sensor4.csv', 'g2_sensor1.csv', 'g2_sensor2.csv', 'g2_sensor3.csv', 'g2_sensor4.csv']  # 실제 파일 이름으로 수정해주세요

# 파일 처리 및 결과 얻기
epoch_2024 = int(datetime.datetime(2024, 1, 1).timestamp())
base_time = epoch_2024 * 1000000  # 100만 추가

for file_name in file_list:
    output = process_csv_files(file_name, base_time)

# 결과 출력
for line in output:
    print(line)
