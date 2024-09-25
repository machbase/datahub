# 필요 라이브러리 호출
import pandas as pd

# 데이터 로드
df_h1 = pd.read_csv('./ETTh1.csv')
df_h2 = pd.read_csv('./ETTh2.csv')
df_m1 = pd.read_csv('./ETTm1.csv')
df_m2 = pd.read_csv('./ETTm2.csv')

# 데이터 컬럼 수정
df_h1.columns = ['time', 'HUFL_h1', 'HULL_h1', 'MUFL_h1', 'MULL_h1', 'LUFL_h1', 'LULL_h1', 'OT_h1']
df_h2.columns = ['time', 'HUFL_h2', 'HULL_h2', 'MUFL_h2', 'MULL_h2', 'LUFL_h2', 'LULL_h2', 'OT_h2']
df_m1.columns = ['time', 'HUFL_m1', 'HULL_m1', 'MUFL_m1', 'MULL_m1', 'LUFL_m1', 'LULL_m1', 'OT_m1']
df_m2.columns = ['time', 'HUFL_m2', 'HULL_m2', 'MUFL_m2', 'MULL_m2', 'LUFL_m2', 'LULL_m2', 'OT_m2']

# 데이터 변환
df_h1 = df_h1.melt(id_vars=['time'], var_name='name', value_name='value')
df_h2 = df_h2.melt(id_vars=['time'], var_name='name', value_name='value')
df_m1 = df_m1.melt(id_vars=['time'], var_name='name', value_name='value')
df_m2 = df_m2.melt(id_vars=['time'], var_name='name', value_name='value')

# 데이터 병합
df_total = pd.concat([df_h1, df_h2, df_m1, df_m2], axis=0).reset_index(drop=True)

# time 열을 datetime 형식으로 변환
df_total['time'] = pd.to_datetime(df_total['time'], format='mixed')

# 데이터 컬럼 순서 변경
df_total = df_total[['name','time','value']]
df_total.columns = ['NAME', 'TIME', 'VALUE']

# 결측값 제거
df_total = df_total.dropna().reset_index(drop=True)

# 기준 시간 (epoch time) -> UTC 시간 변환 
epoch = pd.Timestamp('1970-01-01', tz='UTC')

# 타임 포멧 변경
df_total['TIME'] = pd.to_datetime(df_total['TIME'], format='%Y-%m-%d %H:%M:%S')
df_total['TIME'] = df_total['TIME'].dt.tz_localize('Asia/Seoul')
df_total['TIME'] = df_total['TIME'].apply(lambda x: int((x - epoch).total_seconds() * 1_000_000_000))

# 통합 데이터프레임 저장
df_total.to_csv('./datahub-2024-06-elec-transformer.csv', index=False)