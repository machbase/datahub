import sys
from datetime import datetime, timedelta

def parse_and_print_files(sensor_prefix, filename1, filename2=None):
    # 시작 시간을 2024년 1월 1일로 설정
    current_time = int(datetime(2024, 1, 1).timestamp() * 1000000000)

    with open(filename1, 'r') as file1:
        file2 = open(filename2, 'r') if filename2 else None
        batch_count = 0
        try:
            while True:
                batch = []

                # 64개의 라인을 읽음
                batch_time   = current_time;
                # 두 번째 파일이 제공된 경우에만 해당 파일에서 라인을 읽고 출력
                if file2:
                    second_file_line = file2.readline().strip()
                    if second_file_line:
                        print(f"{sensor_prefix}-answer,{batch_time},{second_file_line}")
                        #print(f"두 번째 파일의 라인 {batch_count}: {second_file_line}")
                    else:
                        #print(f"두 번째 파일의 끝에 도달했습니다.")
                        file2.close()
                        file2 = None

                for _ in range(64):
                    line = file1.readline().strip()
                    if not line:
                        # 파일의 끝에 도달하면 종료
                        if not batch:
                            return
                        break
                    # 각 라인을 숫자 리스트로 변환
                    numbers = [float(x) for x in line.split()]
                    if len(numbers) != 3000:
                        print(f"오류: 라인에 3000개의 숫자가 없습니다. 실제 개수: {len(numbers)}")
                        return
                    batch.append(numbers)

                # 배치 출력
                #print(f"배치 {batch_count + 1}")
                for i in range(3000):
                    for sensor in range(len(batch)):
                        if i < len(batch[sensor]):
                            # epoch 시간을 나노초 단위로 변환 (10억을 곱함)
                            print(f"{sensor_prefix}-{sensor},{current_time},{batch[sensor][i]:.7f}")
                    current_time += 1000000

                current_time += 57 * 1000000000
                batch_count += 1
                #print(f"배치 {batch_count} 완료")

        finally:
            if file2:
                file2.close()

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("사용법: python script.py <sensor_prefix> <source> [answer]")
        sys.exit(1)
    sensor_prefix = sys.argv[1]
    filename1 = sys.argv[2]
    filename2 = sys.argv[3] if len(sys.argv) == 4 else None
    parse_and_print_files(sensor_prefix, filename1, filename2)
