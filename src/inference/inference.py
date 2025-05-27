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

    checkpoint = load_checkpoint()
    model, scaler, label_encoders = init_model(checkpoint)

    
    data = [
        [3.00000000e+01, 1.00000000e+00, 9.38000000e+02, 1.14840000e+02,
        9.00000000e+00, 1.50000000e+01, 2.00000000e+03, 1.80400000e+03,
        1.35000000e+02, 1.00000000e+00, 9.76000000e+02, 3.27000000e+02,
        2.40000000e+02, 0.00000000e+00, 1.14000000e+03, 1.26900834e+02,
        3.75296467e+01, 1.37000000e+05, 1.90000000e+01, 5.30000000e+01,
        2.02300000e+03, 4.00000000e+00, 1.00000000e+00, 1.00000000e+00,
        1.00000000e+00],
        [1.31700000e+03, 0.00000000e+00, 4.28800000e+03, 8.48600000e+01,
        2.70000000e+01, 1.10000000e+01, 2.01100000e+03, 5.11700000e+03,
        1.18400000e+03, 0.00000000e+00, 3.45221705e+02, 7.80000000e+01,
        7.49002584e+01, 0.00000000e+00, 3.37051163e+02, 1.26829869e+02,
        3.75107697e+01, 3.57000000e+04, 1.80000000e+01, 1.85000000e+02,
        2.01200000e+03, 4.00000000e+00, 0.00000000e+00, 1.00000000e+00,
        0.00000000e+00],
        [5.40000000e+01, 7.00000000e+00, 3.47500000e+03, 5.73300000e+01,
        1.30000000e+01, 3.00000000e+00, 1.99700000e+03, 6.71400000e+03,
        1.18400000e+03, 0.00000000e+00, 6.78993095e+02, 7.80000000e+01,
        5.68888889e+01, 2.77557996e+03, 7.74420590e+02, 1.27013688e+02,
        3.75157083e+01, 5.00000000e+04, 1.40000000e+01, 2.56000000e+02,
        2.01100000e+03, 6.00000000e+00, 1.00000000e+00, 1.00000000e+00,
        1.00000000e+00]
    ]

    data = np.array(data)

    price = inference(model, scaler, label_encoders, data, batch_size=64)
    print(price)
