import requests
import csv
import io

import pandas as pd
from urllib.parse import quote

# 파라미터 설정
# tag table 에서 가져올 데이터 컬럼 선택
target = 'name,time,value'
# tag table 이름 설정
table = 'rotor'
# tag name 설정
name = "'g1_sensor1_normal','g2_sensor1_normal'"
# 시작 시간 설정
start = '2024-01-01 00:00:00 000:000:000'
# 끝 시간 설정
end = '2024-01-01 00:01:00 010:000:000'
# 데이터 로드 개수 제한
limit =''
# 시간 포멧 설정
timeformat = 'Default'

# 시간 포멧 변경
# URL 인코딩
start_ = quote(start)
end_ = quote(end)

encoded_url = quote(name, safe=":/")


df = pd.read_csv(f'http://127.0.0.1:5654/db/tql/datahub/common/select-rawdata.tql?target={target}&table={table}&name={encoded_url}&start={start_}&end={end_}&limit={limit}&timeformat={timeformat}')

print(df)
