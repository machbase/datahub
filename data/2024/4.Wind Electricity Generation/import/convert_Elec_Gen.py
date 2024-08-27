import pandas as pd

# 파일리스트
data = ['./hk1.csv','./hk2.csv','./ss.csv']

# 데이터 로드 
df = pd.read_csv(data[0])
df1 = pd.read_csv(data[1])
df2 = pd.read_csv(data[2])

# 데이터 컬럼 수정 
df2.columns = ['time', 'ss_1', 'ss_2', 'ss_3', 'ss_4', 'ss_5', 'ss_6', 'ss_7', 'ss_8', 'ss_9', 'ss_10']

# 데이터 변환
df = df.melt(id_vars=['time'], var_name='name', value_name='value')
df1 = df1.melt(id_vars=['time'], var_name='name', value_name='value')
df2 = df2.melt(id_vars=['time'], var_name='name', value_name='value')

# 데이터 병합
df_ = pd.concat([df,df1,df2], axis=0).reset_index(drop=True)

# time 열을 datetime 형식으로 변환
df_['time'] = pd.to_datetime(df_['time'], format='mixed')

# 데이터 컬럼 순서 변경
df_ = df_[['name','time','value']]
df_.columns = ['NAME', 'TIME', 'VALUE']

# 결측값 제거
df_ = df_.dropna().reset_index(drop=True)

# 기준 시간 (epoch time) -> UTC 시간 변환 
epoch = pd.Timestamp('1970-01-01', tz='UTC')

# 타임 포멧 변경
df_['TIME'] = pd.to_datetime(df_['TIME'], format='%Y-%m-%d %H:%M:%S')
df_['TIME'] = df_['TIME'].dt.tz_localize('Asia/Seoul')
df_['TIME'] = df_['TIME'].apply(lambda x: int((x - epoch).total_seconds() * 1_000_000_000))

# 통합 데이터프레임 저장
df_.to_csv('./datahub-2024-04-wind_elec_gen1.csv', index=False)