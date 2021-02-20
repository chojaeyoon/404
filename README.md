# 시스템 품질 변화로 인한 사용자 불편 예지 AI 경진대회

404
# 공동 작업으로 정리 대기중!!

작업자 : 조재윤, 최명진, 표유진

2021 데이콘, 시스템 품질 변화로 인한 사용자 불편 예지 AI 경진대회

비식별화 된 시스템 기록(로그 및 수치 데이터)을 분석하여 시스템 품질 변화로 사용자에게 불편을 야기하는 요인을 진단

평가 산식 : AUC (사용자로부터 불만 접수가 일어날 확률 예측)

#### 모델링은 정리 작업 진행 중,
### EDA만 readme 형태로 우선 정리

##### 퀼리티 로그의 값을 카테고리 변수로 전환 / 각 카테고리의 임계치 설정
    def quality_categorize(dataset):
      data = dataset.copy()
      for key in tqdm(qualities):
        intervals = qualities.get(key)
        for i in range(len(intervals)):
          if intervals[i] == intervals[-1]:
            data[key][data[key] >= intervals[i]] =  intervals[i]
          # elif intervals[i+1] - intervals[i] == 1:
          #   continue
          else:
            data[key][(data[key] >= intervals[i]) & (data[key] < intervals[i+1])] = intervals[i]
      return data

##### 왼쪽이 처리 천, 오른쪽이 처리 후
![image](https://user-images.githubusercontent.com/76254564/107886563-e2977200-6f43-11eb-831a-7b225bf9f7a6.png)
##### 펌웨어 버전 정보와 모델 버전 정보에 관한 분석 진행

#### 파생 변수 추가
##### 가장 빈번한 오류 2가지를 각 id의 파생변수로 추가
    def quality_frequency(dataset):
      data = dataset.drop(['time'], axis = 1)
      data = data.fillna(-10)
      frequencies = pd.DataFrame()
      #ids = [10000, 24997]
      ids = list(set(data.user_id.values))
      ids.sort()
      for id in tqdm(ids):
        fre_list = []
        temp = data[data['user_id'] == id]
        for i in range(1,11):
          fre = temp.iloc[1:,i].value_counts()
          one = fre.index[0]
          try:
            two = fre.index[1]
          except:
            two = 'None'  
          fre_list.extend([one, two])
        most = pd.DataFrame(fre_list).transpose()
        frequencies = pd.concat([frequencies, most], axis = 0)
      frequencies.columns = freq_list
      frequencies.index = ids
      return frequencies
      
##### 오류 누적 횟수를 일일 단위로 누적하여 변수 추가
    def days_sum_quality(data):
        #ids = [10000, 24997]
        ids = list(set(data.user_id.values))
        ids.sort()
        ids_err = list()
        for id in tqdm(ids):
            so_far = 0
            id_err = list()
            dat = data[data['user_id'] == id]
            for days in range(begins, ends + 1):
                # day_errs = data[(data['time'].dt.dayofyear == days) & (data['user_id'] == id)].shape[0]/12
                day_errs = dat[dat['time'].dt.dayofyear == days].shape[0]/12
                so_far += day_errs
                id_err.extend([so_far])
            ids_err.append(id_err)
        err_sums = pd.DataFrame(ids_err, index = ids, columns=q_list)
        return err_sums
        
##### 추가로 각 id 별 오류코드 보고 횟수 / 시간대 별 오류 발생과 불만접수의 상관관계 / 각종 카테고리 변수를 확률 변수로 치환하여 AUC 변동 확인
    id_error = train_err3[['user_id','errtype']].values
    error = np.zeros((15000,42))
    for person_idx, err in tqdm(id_error):
        # person_idx - 10000 위치에 person_idx, errtype에 해당하는 error값을 +1
        error[person_idx - 10000,err - 1] += 1
    ###########################################################################    
    def cos_stats(data):
      ids = list(set(data['user_id']))
      ids.sort()
      stats = []
      for id in tqdm(ids):
        temp = data[data.user_id == id]
        if len(temp) > 1:
          a = temp.cos_time.mean()
          b = temp.cos_time.std()
          c = temp.cos_time.mode().values[0]
          t_list = [a,b,c]
        else:
          t_list = [0,0,0]
        stats.append(t_list)
      df = pd.DataFrame(stats, columns=['mean', 'std', 'freq'], index=ids)
      return df
    ##############################################################################
    def probability(train, test)
      cols = list(train.loc[:,'quality_1':'quality_12'].columns)

      train['problems'] = train.user_id.isin(prob_list)
      train_tr = train[train['problems']]
      train_fls = train[train['problems'] != True]


      for col in tqdm(cols):
        temp_ori = train[col].value_counts()
        temp_tr = train_tr[col].value_counts()
        temp_fls = train_fls[col].value_counts()

        ori = pd.DataFrame(temp_ori)
        tr = pd.DataFrame(temp_tr)

        probs = tr/ori
        for ind in list(probs.index):
          train[col][train[col] == ind] = probs.loc[ind,:].values[0]
          test[col][test[col] == ind] = probs.loc[ind,:].values[0]
      return train, test
      
##### 업데이트 여부가 불만접수에 큰 영향을 미치므로 업데이트 여부 변수 추가
    def update(dataset):

      data = dataset.copy()
      ids = list(set(data.user_id))
      ids.sort()
      print('1/3')
      data.replace({'fwver': np.nan}, {'fwver': '00.00.0000'}, inplace = True)
      data.replace({'fwver': '8.5.3'}, {'fwver': '08.05.3000'}, inplace = True)
      data.replace({'fwver': '10'}, {'fwver': '10.00.0000'}, inplace = True)
      data['fwver'] = data['fwver'].str.replace('\.','', regex = True)
      print('2/3')
      data = data.astype({'fwver': 'int'})
      fw_list = data.fwver.unique()
      data2 = data.groupby(by = 'user_id').mean()
      print('3/3')
      data2['fw_update'] = data2['fwver'].isin(fw_list)
      data2['fw_update'] = data2['fw_update'].astype(int)
      df = data2.loc[:,['fw_update']]
      return df
      
    def multi_model(dataset):
      data = dataset.copy()
      #ids = [x for x in range(10000,12000)]
      ids = data.user_id.unique()
      ids.sort()

      models = []

      for id in tqdm(ids):
        temp = data[data['user_id'] == id]
        mul = len(temp.model_nm.unique())
        if mul == 1:
          models.append('one')
        else:
          models.append('multi')

      df = pd.DataFrame(models, index = ids, columns = ['multi_model'])

      return df
      
##### 추가된 변수 scaling, 카테고리 변수 encoding, 파라미터 최적화, stacking과 ensemble의 AUC 비교, kfold를 활용한 cv 등 추가 진행
