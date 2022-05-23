from konlpy.tag import Okt # 형태소 분석에 사용할 konlpy 패키지의 Okt 클래스를 임포트하고 okt
okt = Okt()
import re # 정규식을 사용하기 위해 re 모듈을 임포트
# Train용 데이터셋과 Test용 데이터 셋 분리
# 1. 예측력을 높이기 위해 수집된 데이터를 학습용과 테스트 용으로 분리하여 진행
# 2. 보통 20~30%를 테스트용으로 분리해 두고 테스트

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(text, score , test_size=0.2, random_state=0)
print(len(train_x), len(train_y), len(test_x), len(test_y))

from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(tokenizer=okt.morphs, ngram_range=(1,2), min_df=3, max_df=0.9)
tfv.fit(train_x)
tfv_train_x = tfv.transform(train_x)
print(tfv_train_x)

from sklearn.linear_model import LogisticRegression # 이진 분류 알고리즘
from sklearn.model_selection import GridSearchCV # 하이퍼 파라미터 최적화

clf = LogisticRegression(random_state=0)
params = {'C': [15, 18, 19, 20, 22]}
grid_cv = GridSearchCV(clf, param_grid=params, cv=3, scoring='accuracy', verbose=1)
grid_cv.fit(tfv_train_x, train_y)

# 최적의 평가 파라미터는 grid_cv.best_estimator_에 저장됨
print(grid_cv.best_params_, grid_cv.best_score_)# 가장 적합한 파라메터, 최고 정확도 확인

tfv_test_x = tfv.transform(test_x)
# test_predict = grid_cv.best_estimator_.score(tfv_test_x,test_y)
test_predict = grid_cv.best_estimator_.predict(tfv_test_x)
from sklearn.metrics import accuracy_score
print('감성 분류 모델의 정확도 : ',round(accuracy_score(test_y, test_predict), 3))

input_text = '딱히 대단한 재미도 감동도 없는데 ~! 너무 과대 평과된 영화 중 하나'
#입력 텍스트에 대한 전처리 수행
input_text = re.compile(r'[ㄱ-ㅣ가-힣]+').findall(input_text)
input_text = [" ".join(input_text)]
# 입력 텍스트의 피처 벡터화
st_tfidf = tfv.transform(input_text)

# 최적 감성 분석 모델에 적용하여 감성 분석 평가
st_predict = grid_cv.best_estimator_.predict(st_tfidf)

#예측 결과 출력
if(st_predict == 0):
    print('예측 결과: ->> 부정 감성')
else :
    print('예측 결과: ->> 긍정 감성')
