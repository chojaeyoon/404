# 시스템 품질 변화로 인한 사용자 불편 예지 AI 경진대회

# 404_not_found


작업자 : 조재윤, 최명진, 표유진

2021 데이콘, 시스템 품질 변화로 인한 사용자 불편 예지 AI 경진대회

비식별화 된 시스템 기록(로그 및 수치 데이터)을 분석하여 시스템 품질 변화로 사용자에게 불편을 야기하는 요인을 진단

평가 산식 : AUC (사용자로부터 불만 접수가 일어날 확률 예측)

## 폴더 설명

### EDA_Preprocessing
##### EDA ipynb 파일들이 모여있음
##### Raw_data의 사이즈가 큰 관계로(합 3000만건 이상) 연산이 오래 걸리는 feature_engineering 미리 진행하여 csv 저장, 예측에 

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

### Auto_ML 폴더
##### 효율적인 진행을 위해 Auto ML(Pycaret) Library 활용, 여기서 효과적인 파생변수, 접근방식, 모델 선정을 진행하여 개별 모델 생성
#### MK1 폴더
##### 초기모델, datetime 전환과 오류 별 발생횟수, 가장 기본적인 변수를 포함하여 model build
![image](https://user-images.githubusercontent.com/76254564/108595989-6a0e3680-73c5-11eb-92ab-b22e502846a3.png)

#### MK2 폴더
##### Outlier 제거 시도
##### 각 퀼리티 별 발생 횟수 합계와 시간변수 추가

    def cos_stats(dataset):
      dataset['hour'] = dataset['time'].dt.hour
      data = cos_time(dataset)[['user_id', 'cos_time', 'sin_time']]
      cols = []
      temp = data.groupby(by = 'user_id')

      a = temp['cos_time'].mean()
      cols += ['cos_mean']
      b = temp['cos_time'].std()
      cols += ['cos_std']

      c = temp['sin_time'].mean()
      cols += ['sin_mean']
      d = temp['sin_time'].std()
      cols += ['sin_std']

      df = pd.concat([a,b,c,d], axis = 1) 
      df.columns = cols
      return df

#### MK3 폴더
##### 소비자의 사용패턴을 연속적인 시계열로 전환, 비슷한 사용패턴을 보이는 고객들을 군집화 시도
![image](https://user-images.githubusercontent.com/76254564/108596145-8068c200-73c6-11eb-8b97-0a64b689c9f1.png)
    # 시계열 자료를 활용해 군집화.
    ks = KMeans(n_clusters = 7, random_state = 42)
    ks.fit(train_time_series_6h)
    train_time_series_6h['Group'] = ks.predict(train_time_series_6h)
    test_time_series_6h['Group'] = ks.predict(test_time_series_6h)
    # train_time_series_6h['problems'] = train_time_series_6h.index.isin(prob_list)
    
#### MK4 폴더
##### 그간 시도되었던 접근 중에서 AUC 향상에 기여하는 방법론을 모아서 중간 정리

#### MK5 폴더
##### LGBM, Catboost, XGBoost를 선정하여 각 모델을 튜닝, AUC score를 기준으로 제작할 모델 선정, 간단한 stacking 시도
    fit_params={"eval_set" : [(X_val2, y_val2)], "eval_metric" : 'auc', 'verbose': 100}

    from scipy.stats import randint as sp_randint
    from scipy.stats import uniform as sp_uniform

    param_test ={'num_leaves': sp_randint(6, 50), 
                 'min_child_samples': sp_randint(10, 500), 
                 'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                 'subsample': sp_uniform(loc=0.2, scale=0.8), 
                 'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                 'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                 'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
                 
#### MK6 폴더
##### Dacon notebook 토론에 논의되었던 파생변수를 참고하여 새로운 파생변수 시도, 총 761개의 변수를 통한 예측
![image](https://user-images.githubusercontent.com/76254564/108596179-c02fa980-73c6-11eb-89e1-8797acd28469.png)


### references 폴더
##### Dacon에 공유되었던 추가 파생변수들, 본 팀의 코드에 맞게 변형, 새로운 변수만 가져오는 과정

### sv_Reports
##### Sweetviz library를 활용한 EDA와 시각화
![image](https://user-images.githubusercontent.com/76254564/108596263-346a4d00-73c7-11eb-8c7d-92152acb2f99.png)
