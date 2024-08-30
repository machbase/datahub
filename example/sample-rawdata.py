import requests
import csv
import io

# 요청할 URL
url = "http://127.0.0.1:5654/db/tql/datahub/api/v1/select-rawdata.tql"

# URL에 포함할 파라미터
params = {
    "target": "name,time,value",
    "table": "home",
    "name": "'TAG-pressure','TAG-dewPoint'",
    "start": "2016-01-01 14:00:00",
    "end": "2017-01-01 14:00:10",
    "limit": 10
}


# GET 요청 보내기
response = requests.get(url, params=params)

# 응답 확인
if response.status_code == 200:
 # CSV 데이터를 메모리 내 파일 객체로 변환
    csv_data = io.StringIO(response.text)

    # CSV 데이터 파싱
    reader = csv.reader(csv_data)

    # 파싱된 데이터를 리스트로 저장
    data = list(reader)

    # 첫 번째 줄은 헤더일 수 있으므로 분리
    header = data[0]
    rows = data[1:]

    # 출력 확인
    print(header)
    for row in rows:
        print(row)
else:
    print(f"Request failed with status code {response.status_code}")
