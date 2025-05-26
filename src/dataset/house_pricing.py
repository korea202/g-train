import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.utils import config

class HousePricingDataset:
    def __init__(self, df, scaler=None, label_encoders=None):
        self.df = df
        self.features = None
        self.labels = None
        self.scaler = scaler
        self.label_encoders = label_encoders
        self._preprocessing()

    def _preprocessing(self):

        target_nm = 'target'
        # 타겟 및 피처 정의
        self.labels = self.df[target_nm].to_numpy()
        features = self.df.drop(columns=[target_nm], axis=1).to_numpy()

        # 피처 스케일링
        if self.scaler:
            self.features = self.scaler.transform(features)
        else:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features)


    @property
    def features_dim(self):
        return self.features.shape[1]

    """ @property
    def num_classes(self):
        return len(self.label_encoder.classes_) """

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def read_dataset():
    
    return pd.read_csv(config.HOUSE_PRICING_DATA)


# 사용 컬럼 셋팅 
def set_columns(df:pd.DataFrame) -> pd.DataFrame:
    
    columns_to_drop = [
        #'계약년', 
        #'계약월', 
        #'계약일', 
        '중개사소재지',
        #시군구랑 시는 삭제합니다.
        '시군구', 
        '시', 
        '번지',
        #'k-전체세대수',
        'k-전체동수', 
        'k-연면적', 
        'k-주거전용면적', 
        'k-관리비부과면적',
        '경비비관리형태', 
        '청소비관리형태',
        #'k-사용검사일-사용승인일',
        '단지승인일', 
        '사용허가여부',
        'k-단지분류(아파트,주상복합등등)', 
        'k-세대타입(분양형태)',
        'k-관리방식', 
        'k-복도유형', 
        #'k-난방방식',
        '세대전기계약방법',
        '기타/의무/임대/임의=1/2/3/4', 
        'k-팩스번호', 
        '관리비 업로드',
        '단지신청일', 
        '등기신청일자', 
        '거래유형',
        'k-사용검사일-사용승인일', 
        'k-수정일자', 
        '고용보험관리번호', 
        'k-시행사', 
        '신축여부',
        'k-전용면적별세대현황(60㎡이하)', 
        'k-전용면적별세대현황(60㎡~85㎡이하)'
    ]

    # 존재하지 않는 컬럼이 있을 경우 오류를 방지하기 위해 errors='ignore' 옵션 사용 가능
    df.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')

    numerical_list=[] # 숫자형 변수를 담아두기 위한 빈 List 생성
    categorical_list=[] # 문자형 변수를 담다주기 위한 빈 List 생성

    target_nm = 'target'

    for i in df.drop(columns=[target_nm], axis=1).columns : # for문을 활용하여 DataFrame에 있는 Col(컬럼) list들을 하나씩 루프(loop)를 수행함
        if df[i].dtypes == 'O' : # dtypes == 'O'라는 말은 'Object'의 줄임말로서 자료형 형태(문자형)를 나타냄
            categorical_list.append(i) # 상위 if문 조건에서 Col의 타입이 자료형인 경우에는 문자형 변수 List에 해당 내용을 추가함
        else :
            numerical_list.append(i) # 그렇지 않은 경우 숫자형 변수 List에 루프(loop)를 돌고있는 col 명을 추가함
    
    print("숫자형 변수:", numerical_list) # 최종적으로 숫자형 변수 List를 출력
    print("문자형 변수:", categorical_list) # 최종적으로 문자형 변수 List를 출력

    label_encoders = {}
    
    for col in categorical_list:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str)) 
        label_encoders[col] = encoder

    return df, label_encoders 


def split_dataset(df):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
    return train_df, val_df, test_df


def get_datasets(scaler=None, label_encoders=None):

    #1. 데이터 읽기
    df = read_dataset()

    #2. 컬럼 셋팅
    df, label_encoders = set_columns(df)

    #3 .데이타 분할
    train_df, val_df, test_df = split_dataset(df)

    train_dataset = HousePricingDataset(train_df, scaler, label_encoders)
    val_dataset = HousePricingDataset(val_df, scaler=train_dataset.scaler, label_encoders=train_dataset.label_encoders)
    test_dataset = HousePricingDataset(test_df, scaler=train_dataset.scaler, label_encoders=train_dataset.label_encoders)
    return train_dataset, val_dataset, test_dataset
