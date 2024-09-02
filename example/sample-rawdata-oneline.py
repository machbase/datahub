import requests
import csv
import io

import pandas as pd
from urllib.parse import quote

# 파라미터 설정
# tag table 에서 가져올 데이터 컬럼 선택
target = 'name,time,value'
# tag table 이름 설정
table = 'home'
# tag name 설정
name = "'TAG-precipProbability','TAG-pressure'"
# 시작 시간 설정
start = '2016-01-01 14:00:00'
# 끝 시간 설정
end = '2016-01-02 15:00:00'
# 데이터 로드 개수 제한
limit =''
# 시간 포멧 설정
timeformat = 'Default'


params = [
    {
        "target": "name,time,value",
        "table": "home",
        "name": "'TAG-pressure','TAG-dewPoint'",
        "start": "2016-01-01 14:00:00",
        "end"  : "2017-01-01 14:00:10",
        "limit": 10
    },
    {
        "target": "name,time,value",
        "table": "home",
        "name": "'TAG-dewPoint'",
        "start": "'2016-01-01 14:00:00'",
        "end"  : "'2017-01-01 14:00:10'",
        "limit": 10
    },
    {
        "target": "name,time,value",
        "table": "home",
        "name": "'TAG-pressure','TAG-dewPoint'",
        "start": "TO_DATE('2016-01-01 14:00:00.000000000', 'YYYY-MM-DD HH24:MI:SS.mmmuuunnn')",
        "end":   "TO_DATE('2016-01-01 14:00:18.999999999', 'YYYY-MM-DD HH24:MI:SS.mmmuuunnn')",
        "limit": 30
    },
    {
        "target": "name,time,value",
        "table": "home",
        "name": "'TAG-pressure','TAG-dewPoint'",
        "start": "1451624400000000000",
        "end":   "1451624407000000000",
        "limit": 30
    },
]


for query in params:

    # 시간 포멧 변경
    # URL 인코딩
    encoded_url = quote(query["name"], safe=":/")
    start_ = quote(query["start"])
    end_ = quote(query["end"])

    df = pd.read_csv(f'http://127.0.0.1:5654/db/tql/datahub/api/v1/select-rawdata.tql?target={query["target"]}&table={query["table"]}&name={encoded_url}&start={start_}&end={end_}&limit={query["limit"]}&timeformat=Default')

    print(df)
