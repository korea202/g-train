import os
import sys
from dotenv import load_dotenv, dotenv_values

# 환경변수 읽기
if (python_path := dotenv_values().get('PYTHONPATH')) and python_path not in sys.path: sys.path.append(python_path)


#필수 라이브러리 정리
import fire
from icecream import ic
import wandb

import numpy as np
import pandas as pd

from src.utils.constant import Models
from src.dataset.house_pricing import get_datasets
from src.inference.inference import (load_checkpoint, init_model, inference)


def run_train(model_name, batch_size=1, num_epochs=1):
    # 모델명 체크
    Models.validation(model_name)

    # 데이터 불러오기
    train_dataset, val_dataset, test_dataset = get_datasets()

    # 딥러닝용
    model_params = {
        "input_dim": train_dataset.features_dim,
        "num_classes": None,
        "hidden_dim": 64,
    }

    # 모델 생성 HousePricePredictor
    model_class = Models[model_name.upper()].value  # Models -> HOUSE_PRICE_PREDICTOR = HousePricePredictor
    model = model_class(**model_params, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)

    train_loss = model.train_lgbm()
    val_loss, _ = model.evaluate()
    test_loss, predictions = model.test()

    print("train_loss=", train_loss)
    print("val_loss=", val_loss)
    print("test_loss=", test_loss)

    model.save_model(model_params, num_epochs, train_loss, train_dataset.scaler, train_dataset.label_encoders) 

def run_inference(data=None, batch_size=64):

    checkpoint = load_checkpoint()
    model, scaler, label_encoders = init_model(checkpoint)

    if data is None:
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

    price = inference(model, scaler, label_encoders, data, batch_size)
    print(price)

if __name__ == '__main__':  # python main.py

    fire.Fire({
        "train": run_train,  # python main.py train --model_name house_price_predictor
        "inference": run_inference, # python inference/inference.py
    })