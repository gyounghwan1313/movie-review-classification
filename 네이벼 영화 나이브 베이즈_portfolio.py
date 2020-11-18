### 라이브러리 import
from konlpy.tag import Okt # 한글 형태소 분석기
import re # 텍스트 편집
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer # 문자를 벡터화
from sklearn.naive_bayes import MultinomialNB # 나이브베이즈 분류기
from sklearn.metrics import accuracy_score # 모델의 accuracy
from sklearn.model_selection import train_test_split # 데이터 분리
from sklearn.metrics import confusion_matrix, classification_report # 모델의 평가 도구
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier #앙상블 모델
from sklearn.tree import DecisionTreeClassifier # 의사결정 나무
import matplotlib.pylab as plt # 그래프
from matplotlib import font_manager, rc # 그래프 설정

okt=Okt()
# plt 한글깨짐 방지
font_name=font_manager.FontProperties(fname="c:/windows/fonts/malgun.ttf").get_name()
rc('font',family=font_name)

### 데이터 불러오기
data =pd.read_csv("c:/data/나이브베이지안_프로젝트/movie_200.csv",encoding='euc-kr')
data=data[pd.notnull(data.review)]
data = data[["review","star"]] #리뷰와 평점만 추출

plt.hist(data.star, color="gray")
plt.xlabel("영화 평점",fontsize=15)
plt.ylabel("Count",fontsize=15)


data.star=data.star.apply(lambda x: 1 if x>=7 else 0) # 긍정(고평점) / 부정(저평점) 분류



### Naive model A
x_train, x_test, y_train, y_test = train_test_split(data.review,data.star,test_size=.2,random_state=11,stratify=data.star)

len(x_train) #25526
len(x_test) #6382


# 단어 변환 함수 생성
def replace_word(x,y,word):
    a = [re.sub(x,y,i) for i in list(word)]
    return pd.Series(a)

# 단어 전처리 함수 생성
def word_chage(data_set):
    data_set = replace_word('영화', ' ', data_set)
    data_set = replace_word('배우', ' ', data_set)
    data_set = replace_word('주인공', ' ', data_set)
    data_set = replace_word('요', '요. ', data_set)
    data_set = replace_word('니다', '니다. ', data_set)
    data_set = replace_word('재밋', '재밌', data_set)
    data_set = replace_word('재미있', '재밌있다.', data_set)
    data_set = replace_word('\w{0,10}최고\w{0,10}', '최고급', data_set)
    data_set = replace_word('\w{0,10}쓰레기\w{0,10}', '최하급', data_set)
    data_set = replace_word('\w{0,10}후회없\w{0,10}', '재미있다.', data_set)
    data_set = replace_word('안보면\s후회\w{0,10}', '재미있다.', data_set)
    data_set = replace_word('\w{0,10}후회안\w{0,10}', '재미있다.', data_set)
    data_set = replace_word('후회할뻔\w{0,10}', '재미있다.', data_set)
    data_set = replace_word('졸려\w{0,10}', '최하급 ', data_set)
    data_set = replace_word('잤\w{0,10}', '최하급 ', data_set)
    data_set = replace_word('좋았을텐데\w{0,10}', '아쉽다.', data_set)
    data_set = replace_word('\s0점\w{0,10}', '최하급 ', data_set)
    data_set = replace_word('\s1점\w{0,10}', '최하급 ', data_set)
    data_set = replace_word('\s2점\w{0,10}', '최하급 ', data_set)
    data_set = replace_word('\s3점\w{0,10}', '최하급 ', data_set)
    data_set = replace_word('\s4점\w{0,10}', '최하급 ', data_set)
    data_set = replace_word('10자\w{0,10}', ' ', data_set)
    data_set = replace_word('실망\w{0,10}', ' 실망 ', data_set)
    data_set = replace_word('\w{0,10}지루한\s?\w{0,10}', ' 지루 ', data_set)
    data_set = replace_word('불면증\w{0,10}', ' 지루 ', data_set)
    data_set = replace_word('\w{0,10}루즈\s?\w{0,10}', ' 지루 ', data_set)
    data_set = replace_word('\w{0,10}노잼\w{0,10}', ' 최하급 ', data_set)
    data_set = replace_word('\w{0,10}꿀잼\w{0,10}', ' 재미있다. ', data_set)
    data_set = replace_word('\w{0,10}비추천\w{0,10}', ' 최하급 ', data_set)
    data_set = replace_word('\w{0,10}지루하지\s+\w{0,10}', ' 재미있다. ', data_set)
    data_set = replace_word('\w{0,10}지루하지않고\w{0,10}', ' 재미있다. ', data_set)
    return data_set

x_train=word_chage(x_train) #단어전처리

contnet_token=[okt.morphs(i) for i in x_train] #토큰화

# 한 글자 이상인 단어들을 다시 문장의 형태로 이어붙이기
contents_vectorize=[]
for i in contnet_token:
    txt =''
    for j in i:
        if len(j)>1:
            txt = txt + ' ' + j
    contents_vectorize.append(txt)
contents_vectorize[1] #예시

cv = CountVectorizer() #문자를 벡터화하기 위한 인스턴스
nb = MultinomialNB() # 나이브 베이즈를 위한 인스턴스

## 문자를 벡터화
x_train = cv.fit_transform((list(contents_vectorize)))

## 모델에 fit
nb.fit(x_train,y_train)



## vaidation dataset 정제
x_test=word_chage(x_test)

contnet_token=[okt.morphs(i) for i in x_test]

contents_vectorize=[]
for i in contnet_token:
    txt =''
    for j in i:
        if len(j)>1:
            txt = txt + ' ' + j
    contents_vectorize.append(txt)

x_test = cv.transform(list(contents_vectorize)) #벡터화
y_predict =  nb.predict(x_test) # 예측 값 생성

## 모델의 평가
# 정확도
print(accuracy_score(y_test,y_predict)) # 0.870103415857098
print(pd.crosstab(y_test,y_predict)) # 혼동행렬
print(classification_report(y_test,y_predict))




## 20개 영화로 모델 test

#데이터 불러오기
new_test = pd.read_csv("c:/data/나이브베이지안_프로젝트/movie_20_test.csv",encoding="euc-kr")
new_test=new_test[pd.notnull(new_test.review)]
new_test = new_test[["review","star"]]
new_test.star=new_test.star.apply(lambda x: 1 if x>=7 else 0)

new_x_test = new_test.review
new_y_test = new_test.star

new_x_test=word_chage(new_x_test)

contnet_token=[okt.morphs(i) for i in new_x_test]

contents_vectorize=[]
for i in contnet_token:
    txt =''
    for j in i:
        if len(j)>1 :
            txt = txt + ' ' + j
    contents_vectorize.append(txt)
contents_vectorize[1]


new_x_test = cv.transform(list(contents_vectorize)) #단어를 벡터화
y_predict =  nb.predict(new_x_test) # 예측값 생성

##모델의 평가
accuracy_score(new_y_test,y_predict) #0.8262948207171315
pd.crosstab(new_y_test,y_predict)
print(classification_report(new_y_test,y_predict))




######################################################################################


### Naive model B
# train data를 세개로 나누기
train, test =train_test_split(data,test_size=.2,random_state=11,stratify=data.star)
len(train) #25526
len(test) # 6382

train_1, train_1_1 =train_test_split(train,test_size=.67,random_state=11,stratify=train.star)
len(train_1) #8423
len(train_1_1) #17103

train_2, train_3 =train_test_split(train_1_1,test_size=.5,random_state=11,stratify=train_1_1.star)
len(train_2) #8551
len(train_3) #8552


## train_1 dataset으로 모델1을 생성
x_train_1 = train_1.review
y_train_1 = train_1.star

x_train_1=word_chage(x_train_1)

contnet_token=[okt.morphs(i) for i in x_train_1]

contents_vectorize=[]
for i in contnet_token:
    txt =''
    for j in i:
        if len(j)>1:
            txt = txt + ' ' + j
    contents_vectorize.append(txt)

cv1 = CountVectorizer() # train_1에 대한 인스턴스 생성
nb1 = MultinomialNB() # train_1에 대한 인스턴스 생성

x_train_1_trans = cv1.fit_transform((list(contents_vectorize))) # 문자를 벡터화

nb1.fit(x_train_1_trans,y_train_1) #모델이 fit

##train_1 데이터를 스스로 평가하기
x_train_1_self = cv1.transform(list(contents_vectorize))
y_predict_1 = nb1.predict(x_train_1_self)

print(accuracy_score(y_train_1,y_predict_1))
print(pd.crosstab(y_train_1,y_predict_1))
print(classification_report(y_train_1,y_predict_1))

#train_1에서 오분류 고르기
y_fail_1= pd.Series(y_train_1[y_train_1!=y_predict_1].values)
x_fail_1 = pd.Series(x_train_1[list(y_train_1!=y_predict_1)].values)
fail_1_index = pd.Series(y_train_1[y_train_1!=y_predict_1].index)

df_fail_1=pd.DataFrame({"review":list(x_fail_1),"star":list(y_fail_1)},index=fail_1_index) # 오분류된 데이터


## 오분류된 데이터를 train_2데이터와 합치기
train_2 = pd.concat([train_2,df_fail_1])

## train_2 dataset으로 모델2을 생성
x_train_2 = train_2.review
y_train_2 = train_2.star

x_train_2=word_chage(x_train_2)
contnet_token=[okt.morphs(i) for i in x_train_2]

contents_vectorize=[]
for i in contnet_token:
    txt =''
    for j in i:
        if len(j)>1:
            txt = txt + ' ' + j
    contents_vectorize.append(txt)

cv2 = CountVectorizer() # train_2에 대한 인스턴스 생성
nb2 = MultinomialNB() # train_2에 대한 인스턴스 생성

x_train_2_trans = cv2.fit_transform((list(contents_vectorize)))

nb2.fit(x_train_2_trans,y_train_2)

## train_2 데이터를 스스로 평가하기
x_train_2_self = cv2.transform(list(contents_vectorize))
y_predict_2 = nb2.predict(x_train_2_self)

print(accuracy_score(y_train_2,y_predict_2))
print(pd.crosstab(y_train_2,y_predict_2))
print(classification_report(y_train_2,y_predict_2))

## train_2에서 오분류 고르기
y_fail_2= pd.Series(y_train_2[y_train_2!=y_predict_2].values)
x_fail_2 = pd.Series(x_train_2[list(y_train_2!=y_predict_2)].values)
fail_2_index = pd.Series(y_train_2[y_train_2!=y_predict_2].index)

df_fail_2=pd.DataFrame({"review":list(x_fail_2),"star":list(y_fail_2)},index=fail_2_index) #train_2에서 오분류된 데이터

## train_3번과 오분류 데이터 합치기
train_3 = pd.concat([train_3,df_fail_2])

#train_3로 모델 3생성

x_train_3 = train_3.review
y_train_3 = train_3.star

x_train_3=word_chage(x_train_3)
contnet_token=[okt.morphs(i) for i in x_train_3]

contents_vectorize=[]
for i in contnet_token:
    txt =''
    for j in i:
        if len(j)>1:
            txt = txt + ' ' + j
    contents_vectorize.append(txt)

cv3 = CountVectorizer() # train_3에 대한 인스턴스 생성
nb3 = MultinomialNB() # train_3에 대한 인스턴스 생성


x_train_3_trans = cv3.fit_transform((list(contents_vectorize)))

nb3.fit(x_train_3_trans,y_train_3) # 모델3에 fit

#train_3 데이터를 스스로 평가하기
x_train_3_self = cv3.transform(list(contents_vectorize))
y_predict_3 = nb3.predict(x_train_3_self)

print(accuracy_score(y_train_3,y_predict_3))
print(pd.crosstab(y_train_3,y_predict_3))
print(classification_report(y_train_3,y_predict_3))




## validation datset으로 모델 평가하기
x_test = test.review
y_test = test.star
y_test=y_test.reset_index(drop=True)

x_test=word_chage(x_test)

contnet_token=[okt.morphs(i) for i in x_test]

contents_vectorize=[]
for i in contnet_token:
    txt =''
    for j in i:
        if len(j)>1:
            txt = txt + ' ' + j
    contents_vectorize.append(txt)
contents_vectorize[1]


# 모델 1로 1차 분류
x_test_1_trans = cv1.transform(list(contents_vectorize))
y_predict_test_1 = nb1.predict(x_test_1_trans)

print(accuracy_score(y_test,y_predict_test_1)) #0.8497336258226261
print(pd.crosstab(y_test,y_predict_test_1))
print(classification_report(y_test,y_predict_test_1))

# 모델 2로 2차 분류
x_test_2_trans = cv2.transform(list(contents_vectorize))
y_predict_test_2 =  nb2.predict(x_test_2_trans)

print(accuracy_score(y_test,y_predict_test_2)) # 0.8486367909746161
print(pd.crosstab(y_test,y_predict_test_2))
print(classification_report(y_test,y_predict_test_2))

# 모델 3로 3차 분류
x_test_3_trans = cv3.transform(list(contents_vectorize))
y_predict_test_3 =  nb3.predict(x_test_3_trans)

print(accuracy_score(y_test,y_predict_test_3)) #0.8470698840488875
print(pd.crosstab(y_test,y_predict_test_3))
print(classification_report(y_test,y_predict_test_3))


## 각 모델별로 분류된 것을 데이터 프레임으로 저장
result=pd.DataFrame({'nb1':list(y_predict_test_1),'nb2':list(y_predict_test_2),'nb3':list(y_predict_test_3)})
result.info()
result.head()

result["sum"]=result.nb1+result.nb2+result.nb3 # 3개의 모델의 분류여부를 sum

result.loc[(result["sum"]==2) | (result["sum"]==1),"sum"] # 1 또는 2인 데이터는 1002
result["sum"][result["sum"]==2] # 2인 데이터는 553
result["sum"][result["sum"]==1] # 1인 데이터는 449
result["sum"][result["sum"]==3] # 3인 데이터는 4318
result["sum"][result["sum"]==0] # 0인 데이터는 1062

# 2이상인 데이터를 긍정으로 분류, 그렇지 않은 데이터는 부정으로 분류
result["pred"]=result["sum"].apply(lambda x: 1 if x>=2 else 0)
result.head()

# 다수결방식으로 정한 긍정/부정을 라벨에 대해 성능 평가
print(accuracy_score(y_test,result["pred"])) #0.8569413976809778
print(pd.crosstab(y_test,result["pred"]))
print(classification_report(y_test,result["pred"]))


## 모델2,3에 가중치를 부여하여 긍정/부정 분류
result_prob=pd.DataFrame({'nb1_1':pd.Series(nb1.predict_proba(x_test_1_trans).tolist()).apply(lambda x:x[0]).tolist(),
                          'nb1_2':pd.Series(nb1.predict_proba(x_test_1_trans).tolist()).apply(lambda x:x[1]).tolist(),
                          'nb2_1':pd.Series(nb2.predict_proba(x_test_2_trans).tolist()).apply(lambda x:x[0]).tolist(),
                          'nb2_2':pd.Series(nb2.predict_proba(x_test_2_trans).tolist()).apply(lambda x:x[1]).tolist(),
                          'nb3_1':pd.Series(nb3.predict_proba(x_test_3_trans).tolist()).apply(lambda x:x[0]).tolist(),
                          'nb3_2':pd.Series(nb3.predict_proba(x_test_3_trans).tolist()).apply(lambda x:x[1]).tolist()})
result_prob.head() # 각 모델별로 각 리뷰가 긍정/부정으로 분류될 확률값을 계산

result_prob.iloc[:,[2,3]]=result_prob.iloc[:,[2,3]]*1.1 # 모델2의 값에 1.1가중치를 부여
result_prob.iloc[:,[4,5]]=result_prob.iloc[:,[4,5]]*1.2 # 모델3의 값에 1.2가중치를 부여

# 가중치를 바탕으로 모델1,2,3의 확률을 더해 최종 확률값을 계산
result_prob["prob_0"]=result_prob.iloc[:,[0,2,4]].sum(axis=1)
result_prob["prob_1"]=result_prob.iloc[:,[1,3,5]].sum(axis=1)
result_prob[["prob_0","prob_1"]]

# 긍정/부정에 대한 라벨 생성
pred_prob=[]
for i in range(len(result_prob)):
    if result_prob.iloc[i,6] > result_prob.iloc[i,7]:
        x=0
    else :
        x=1
    pred_prob.append(x)

# 모델 평가
print(accuracy_score(y_test,pred_prob)) # 0.8607019743027264
print(pd.crosstab(y_test,pd.Series(pred_prob)))
print(classification_report(y_test,pd.Series(pred_prob)))


##############################################################################################################





### Naive model C
x_train, x_test, y_train, y_test = train_test_split(data.review,data.star,test_size=.2,random_state=11,stratify=data.star)

y_train=y_train.reset_index(drop=True)

x_train=word_chage(x_train) # 단어 전처리

contnet_token=[okt.morphs(i) for i in x_train] #토큰화

contents_vectorize=[]
for i in contnet_token:
    txt =''
    for j in i:
        if len(j)>1:
            txt = txt + ' ' + j
    contents_vectorize.append(txt)

cv = CountVectorizer() # 단어 벡터화에 대한 인스턴스 생성
nb = MultinomialNB() # 모델에 대한 인스턴스 생성


x_train_trans = cv.fit_transform((list(contents_vectorize))) # 단어 벡터화
nb.fit(x_train_trans,y_train) # 모델 fit


# 스스로 모델을 평가
x_train_1_self = cv.transform(list(contents_vectorize))
y_predict_1 = nb.predict(x_train_1_self)

print(accuracy_score(y_train,y_predict_1))
print(pd.crosstab(y_train,y_predict_1))
print(classification_report(y_train,y_predict_1))


## 리뷰 중 긍정/부정 확률이 애매한 리뷰들을 다시 학습

y_predict_1_prob = nb.predict_proba(x_train_1_self) # 각 리뷰에 대해 긍정/부정 분류 확률을 저장
result_train1_prob = pd.DataFrame({'n0':pd.Series(y_predict_1_prob.tolist()).apply(lambda x:x[0]).tolist(),'n1':pd.Series(y_predict_1_prob.tolist()).apply(lambda x:x[1]).tolist()},index=x_train.index)
result_train1_prob # 데이터 프레임 형태로 긍정/부정 확률을 저장

# |긍정-부정| <=0.5인 리뷰들을 선별
abs_prob=[]
for i in range(len(result_train1_prob)):
    if abs(result_train1_prob.iloc[i,0] - result_train1_prob.iloc[i,1]) <= .05:
        x=1
    else:
        x=0
    abs_prob.append(x)

result_train1_prob["abs_prob"]=abs_prob
result_train1_prob # 모호한 리뷰인지에 대한 컬럼을 추가

#모호한 리뷰와 그렇지 않은 리뷰를 나눠서 dataframe형태로 저장
index_sim=result_train1_prob[result_train1_prob["abs_prob"]==1].index.tolist()

#모호하지 않은 리뷰들
x_train_nosim=x_train.drop(index_sim, axis=0)
y_train_nosim=y_train.drop(index_sim, axis=0)

#모호한 리뷰들
x_train_sim=x_train[index_sim]
y_train_sim=y_train[index_sim]

y_pre_pred=pd.Series(y_predict_1).drop(index_sim, axis=0) # 모호한 리뷰들의 실제 라벨값을 저장


## 모호한 리뷰에 대해서 학습하여 모델 2생성

x_train_sim=word_chage(x_train_sim)
contnet_token=[okt.morphs(i) for i in x_train_sim]

contents_vectorize=[]
for i in contnet_token:
    txt =''
    for j in i:
        if len(j)>1:
            txt = txt + ' ' + j
    contents_vectorize.append(txt)

cv10 = CountVectorizer() # 모호한 리뷰 단어를 벡터화하는 인스턴스
nb10 = MultinomialNB() # 모호한 리뷰를 분류하는 인스턴스


x_train_sim_trans = cv10.fit_transform((list(contents_vectorize))) # 문자를 벡터화

nb10.fit(x_train_sim_trans,y_train_sim) # 모델에fit


# 모호한 리뷰들을 학습한 모델을 스스로 평가
x_train_10_self = cv10.transform(list(contents_vectorize))
y_predict_10 = nb10.predict(x_train_10_self)

print(accuracy_score(y_train_sim,y_predict_10))
print(pd.crosstab(y_train_sim,y_predict_10))
print(classification_report(y_train_sim,y_predict_10))

## 모호하지 않은 리뷰와 모호한 리뷰를 예측한 값에 대해 평가
final_label=pd.concat([y_train_nosim,y_train_sim])
final_pred=pd.concat([y_pre_pred,pd.Series(y_predict_10)])

print(accuracy_score(final_pred,final_label)) #0.9270155919454673
print(pd.crosstab(final_pred,final_label))
print(classification_report(final_pred,final_label))


## validation dataset으로 모델 평가
y_test = y_test.reset_index(drop=True)

x_test=word_chage(x_test)

contnet_token=[okt.morphs(i) for i in x_test]

contents_vectorize=[]
for i in contnet_token:
    txt =''
    for j in i:
        if len(j)>1:
            txt = txt + ' ' + j
    contents_vectorize.append(txt)
contents_vectorize[1]

x_test_trans = cv.transform(list(contents_vectorize)) # 단어를 벡터화
y_predict_test =  nb.predict(x_test_trans) # 모델에 적용

print(accuracy_score(y_test,y_predict_test))

## 모호한 리뷰들을 선별
y_predict_test_prob = nb.predict_proba(x_test_trans)
result_test_prob = pd.DataFrame({'n0':pd.Series(y_predict_test_prob.tolist()).apply(lambda x:x[0]).tolist(),'n1':pd.Series(y_predict_test_prob.tolist()).apply(lambda x:x[1]).tolist()},index=x_test.index)

abs_prob=[]
for i in range(len(result_test_prob)):
    if abs(result_test_prob.iloc[i,0] - result_test_prob.iloc[i,1]) < .05:
        x=1
    else:
        x=0
    abs_prob.append(x)


result_test_prob["abs_prob"]=abs_prob
index_sim=result_test_prob[result_test_prob["abs_prob"]==1].index.tolist()

# 모호하지 않은 리뷰들
x_test_nosim=x_test.drop(index_sim, axis=0)
y_test_nosim=y_test.drop(index_sim, axis=0)

# 모호한 리뷰들
x_test_sim=x_test[index_sim]
y_test_sim=y_test[index_sim]

# 모호란 리뷰들의 실제 라벨값
y_pre_pred=pd.Series(y_predict_test).drop(index_sim, axis=0)



## 모호한 리뷰들을 모델2로 예측
x_test_sim=word_chage(x_test_sim)

contnet_token=[okt.morphs(i) for i in x_test_sim]

contents_vectorize=[]
for i in contnet_token:
    txt =''
    for j in i:
        if len(j)>1:
            txt = txt + ' ' + j
    contents_vectorize.append(txt)
contents_vectorize[1]

x_test_sim_trans = cv10.transform(list(contents_vectorize))
y_predict_sim_test =  nb10.predict(x_test_sim_trans) # 모델에 적용

print(accuracy_score(y_test_sim,y_predict_sim_test)) #모호한 리뷰들의 accuarcy값
print(pd.crosstab(y_test_sim,y_predict_sim_test))
print(classification_report(y_test_sim,y_predict_sim_test))

# 모호란 리뷰와 모호하지 않은 리뷰들을 합쳐 validation dataset의 최종 평가
final_label=pd.concat([y_test_nosim,y_test_sim])
final_pred=pd.concat([y_pre_pred,pd.Series(y_predict_sim_test)])

# 최종 모델의 평가
print(accuracy_score(final_pred,final_label)) # 0.8699467251645252
print(pd.crosstab(final_pred,final_label))
print(classification_report(final_pred,final_label))




## 20개의 영화 test dataset으로 테스트
new_test = pd.read_csv("c:/data/나이브베이지안_프로젝트/movie_20_test.csv",encoding="euc-kr")

new_test=new_test[pd.notnull(new_test.review)]
new_test = new_test[["review","star"]]
new_test=new_test.reset_index(drop=True)
new_test.star=new_test.star.apply(lambda x: 1 if x>=7 else 0)

new_x_test = new_test["review"]
new_y_test = new_test["star"]


new_x_test=word_chage(new_x_test)
contnet_token=[okt.morphs(i) for i in new_x_test]

contents_vectorize=[]
for i in contnet_token:
    txt =''
    for j in i:
        if len(j)>1 :
            txt = txt + ' ' + j
    contents_vectorize.append(txt)
contents_vectorize[1]

new_x_test_trans = cv.transform(list(contents_vectorize))
y_predict_test =  nb.predict(new_x_test_trans)

accuracy_score(new_y_test,y_predict_test)
pd.crosstab(new_y_test,y_predict_test)
print(classification_report(new_y_test,y_predict_test))


## 모호한 리뷰들 분류
y_predict_test_prob = nb.predict_proba(new_x_test_trans)
result_test_prob = pd.DataFrame({'n0':pd.Series(y_predict_test_prob.tolist()).apply(lambda x:x[0]).tolist(),'n1':pd.Series(y_predict_test_prob.tolist()).apply(lambda x:x[1]).tolist()})

abs_prob=[]
for i in range(len(result_test_prob)):
    if abs(result_test_prob.iloc[i,0] - result_test_prob.iloc[i,1]) < .05:
        x=1
    else:
        x=0
    abs_prob.append(x)

result_test_prob["abs_prob"]=abs_prob
index_sim=result_test_prob[result_test_prob["abs_prob"]==1].index.tolist()

# 모호하지 않은 리뷰들
new_x_test_nosim=new_x_test.drop(index_sim, axis=0)
new_y_test_nosim=new_y_test.drop(index_sim, axis=0)

# 모호한 리뷰들
x_test_sim=new_x_test[index_sim]
y_test_sim=new_y_test[index_sim]

# 모호한 리뷰들에 대한 실제 label 값
y_pre_pred=pd.Series(y_predict_test).drop(index_sim, axis=0)



## 모호한 리뷰들에 대해 모델2로 예측
x_test_sim=word_chage(x_test_sim)

contnet_token=[okt.morphs(i) for i in x_test_sim]

contents_vectorize=[]
for i in contnet_token:
    txt =''
    for j in i:
        if len(j)>1:
            txt = txt + ' ' + j
    contents_vectorize.append(txt)

x_test_sim_trans = cv10.transform(list(contents_vectorize))
y_predict_sim_test =  nb10.predict(x_test_sim_trans) # 모델에 적용

# 모호한 리뷰들의 예측값 평가
print(accuracy_score(y_test_sim,y_predict_sim_test))
print(pd.crosstab(y_test_sim,y_predict_sim_test))
print(classification_report(y_test_sim,y_predict_sim_test))

## 모호한 리뷰와 그렇지 않은 리뷰를 합쳐 모델의 최종 예측값 생성과 평가
final_label=pd.concat([new_y_test_nosim,y_test_sim])
final_pred=pd.concat([y_pre_pred,pd.Series(y_predict_sim_test)])

print(accuracy_score(final_pred,final_label)) #0.8270916334661355
print(pd.crosstab(final_pred,final_label))
print(classification_report(final_pred,final_label))



#############################################################################################################



## 랜덤포레스트
x_train, x_test, y_train, y_test = train_test_split(data.review,data.star,test_size=.2,random_state=11,stratify=data.star)

y_train=y_train.reset_index(drop=True)

x_train=word_chage(x_train)

contnet_token=[okt.morphs(i) for i in x_train]

contents_vectorize=[]
for i in contnet_token:
    txt =''
    for j in i:
        if len(j)>1:
            txt = txt + ' ' + j
    contents_vectorize.append(txt)

cv = CountVectorizer()
nb = MultinomialNB()

x_train_trans = cv.fit_transform((list(contents_vectorize)))

# 모델생성
rfmodel=RandomForestClassifier(criterion="entropy",max_depth=400,n_estimators=100,oob_score=True)

rfmodel.fit(x_train_trans,y_train)
rfmodel.score(x_train_trans,y_train)

# validation dataset으로 평가
x_test=word_chage(x_test)

contnet_token=[okt.morphs(i) for i in x_test]

contents_vectorize=[]
for i in contnet_token:
    txt =''
    for j in i:
        if len(j)>1:
            txt = txt + ' ' + j
    contents_vectorize.append(txt)


x_test = cv.transform(list(contents_vectorize))

y_predict = rfmodel.predict(x_test) # 예측

# 랜덤포레스트 모델 평가
print(accuracy_score(y_test,y_predict)) #0.80
print(pd.crosstab(y_test,y_predict))
print(classification_report(y_test,y_predict))




## 부스팅
tree_model=DecisionTreeClassifier(criterion='entropy',max_depth=500)
tree_model.fit(x_train_trans,y_train)
tree_model.score(x_train_trans,y_train)
y_predict = tree_model.predict(x_test)

print(accuracy_score(y_test,y_predict)) #0.85
print(pd.crosstab(y_test,y_predict))
print(classification_report(y_test,y_predict))

# 부스팅 모델 생성
boost_model = AdaBoostClassifier(base_estimator=tree_model,n_estimators=500)
boost_model.fit(x_train_trans,y_train)
boost_model.score(x_train_trans,y_train)
y_predict = boost_model.predict(x_test)

# 부스팅 모델 평가
print(accuracy_score(y_test,y_predict)) #0.85
print(pd.crosstab(y_test,y_predict))
print(classification_report(y_test,y_predict))


## 배깅

# 배깅 모델 생성
bagg_model=BaggingClassifier(base_estimator=tree_model,n_estimators=100)
bagg_model.fit(x_train_trans,y_train)
bagg_model.score(x_train_trans,y_train)
y_predict = bagg_model.predict(x_test)

# 배깅 모델 평가
print(accuracy_score(y_test,y_predict)) #0.85
print(pd.crosstab(y_test,y_predict))
print(classification_report(y_test,y_predict))

