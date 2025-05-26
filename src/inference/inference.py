import os
import sys
import glob
import pickle

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.utils import config, utils
from src.model.house_price_predictor import HousePricePredictor
from src.dataset.house_pricing import HousePricingDataset, get_datasets

def make_inference_df(data):

    columns = "본번 부번 아파트명 전용면적(㎡) 계약일 층 건축년도 도로명 k-전화번호 k-난방방식 k-전체세대수 k-건설사(시공사) k-85㎡~135㎡이하" \
                " 건축면적 주차대수 좌표X 좌표Y target 구 동 계약년 계약월 강남여부 버스정류장유무 지하철역유무".split()
    return pd.DataFrame(data=data, columns=columns)

def model_validation(model_path):
    original_hash = utils.read_hash(model_path)
    current_hash = utils.calculate_hash(model_path)
    if original_hash == current_hash:
        print("validation success")
        return True
    else:
        return False

def load_checkpoint():
    target_dir = os.path.join(config.MODELS_DIR, HousePricePredictor.name)
    models_path = os.path.join(target_dir, "*.pkl")
    latest_model = glob.glob(models_path)[-1]

    if model_validation(latest_model):
        with open(latest_model, "rb") as f:
            checkpoint = pickle.load(f)
        return checkpoint    
    else:
        raise FileExistsError("Not found or invalid model file")        

def init_model(checkpoint):
    model = HousePricePredictor(**checkpoint["model_params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    scaler = checkpoint.get("scaler", None)
    label_encoders = checkpoint.get("label_encoders", None)
    return model, scaler, label_encoders


def inference(model, scaler, label_encoders, data:np.array, batch_size=1):
    if data.size > 0:
        df = make_inference_df(data)

        print(df.head())
        dataset = HousePricingDataset(df, scaler=scaler, label_encoders=label_encoders)

    else:
        _, _, dataset = get_datasets(scaler=scaler, label_encoders=label_encoders)

    loss, predictions = model.test(dataset)
    print(loss, predictions)
    return predictions  

if __name__ == '__main__':

    load_dotenv()
    checkpoint = load_checkpoint()
    model, scaler, label_encoders = init_model(checkpoint)
    
    #data = np.array([1, 1092073, 4508, 7.577, 1204.764])
    data = np.array([])
    recommend = inference(model, scaler, label_encoders, data=np.array([]), batch_size=64)
    print("-------------------------------------")
    print(recommend)