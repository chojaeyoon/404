# 시스템 품질 변화로 인한 사용자 불편 예지 AI 경진대회

404
# 공동 작업으로 정리 대기중!!

작업자 : 조재윤, 최명진, 표유진

2021 데이콘, 시스템 품질 변화로 인한 사용자 불편 예지 AI 경진대회

비식별화 된 시스템 기록(로그 및 수치 데이터)을 분석하여 시스템 품질 변화로 사용자에게 불편을 야기하는 요인을 진단

평가 산식 : AUC (사용자로부터 불만 접수가 일어날 확률 예측)

## 폴더 설명

#### EDA_Preprocessing
##### EDA ipynb 파일들이 모여있음

##### 간단한 datetime 전환부터
    def time_split(data):
      data['time2'] = pd.to_datetime(data['time'], format= "%Y%m%d%H%M%S")
      data['time2'] = data['time2'].dt.strftime('%Y-%m-%d %H:%M:%S')
      data.drop(['time'], axis = 1, inplace = True)
      data.rename(columns = {'time2' : 'time'}, inplace = True)
      return data
##### 불만 고객과 만족 고객의 T 검정
    # T 검정을 하기 위해서 두 표본이 등분산인지 확인할 필요가 있음
    # 따라서 등분산 검정을 진행
    cols = list(train_qui2_tr.iloc[:,1:-2].columns)

    r_list = []

    for col in cols:
      result = stats.levene(train_qui2_tr[col], train_qui2_fls[col])
      r_list.append(list(result))
    #print('LeveneResult(F) : %.3f \np-value : %.3f' % (lresult))

    df = pd.DataFrame(r_list, index = cols, columns = ['통계값', 'P_value'])
    df['등분산성'] = df['P_value'].apply(lambda x: '불만족' if x < 0.05 else '만족')
    df
    
##### 사용시간 패턴 등등
![image](https://user-images.githubusercontent.com/76254564/108595774-f53afc80-73c4-11eb-8dde-28f7ac748564.png)

#### Auto_ML 폴더
##### 효율적인 진행을 위해 Auto ML(Pycaret) Library 활용, 여기서 효과적인 파생변수, 접근방식, 모델 선정을 진행하여 개별 모델 생성
#### MK1 폴더
#####
