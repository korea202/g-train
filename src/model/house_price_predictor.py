import os
import pickle
import datetime

import numpy as np
import lightgbm as lgbm

from src.dataset.house_pricing import HousePricingDataset
from src.utils import config, utils

class HousePricePredictor:
    name = "house_price_predictor"  

    def __init__(self, input_dim=None, num_classes=None, hidden_dim=None, train_dataset=None, val_dataset=None, test_dataset=None):
        
        self. model = lgbm.LGBMRegressor(
                boosting_type='gbdt',
                colsample_bytree=0.8219308825575806,
                learning_rate=0.23722302135412585, 
                max_bin=255,
                min_child_samples=47, 
                #n_estimators=11719,
                n_estimators=4000, 
                n_jobs=-1,
                num_leaves=23, 
                reg_alpha=1.650103578274446,
                reg_lambda=1.3622126795875413, 
                verbose=-1, 
                device='gpu')
    
        self.train_dataset = train_dataset
        self.val_dataset =  val_dataset
        self.test_dataset = test_dataset

    def load_state_dict(self, state_dict):
        self.model = state_dict   

    def train_lgbm(self):
        dataset = self.train_dataset
        evals_result_history = {}

        self.model.fit(
            dataset.features, 
            dataset.labels,
            eval_set=[(dataset.features, dataset.labels)], 
            eval_names=['train'],
            eval_metric='mse', 
            callbacks=[lgbm.early_stopping(stopping_rounds=50), 
                        lgbm.log_evaluation(period=100, show_stdv=True),
                        lgbm.record_evaluation(evals_result_history)] )
        
        train_mse_history = evals_result_history.get('train', {}).get('l2', [])
        
        return train_mse_history[-1]


    def evaluate(self, dataset=None):
        
        dataset =  self.val_dataset  if dataset is None else dataset

        # 검증 데이터 예측 및 loss 출력
        predictions = self.model.predict(dataset.features)
        loss = np.mean((predictions - dataset.labels) ** 2)
        
        return loss, predictions

    def test(self, dataset=None):

        dataset =  self.test_dataset  if dataset is None else dataset

        # 테스트 데이터 예측 및 loss 출력
        predictions = self.model.predict(dataset.features)
        loss = np.mean((predictions - dataset.labels) ** 2)
        
        return loss, predictions
    
    def save_model(self, model_params, epoch, loss, scaler, label_encoders):
        save_dir = os.path.join(config.MODELS_DIR, self.name) 
        os.makedirs(save_dir, exist_ok=True)
        
        current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")  # 250521142130
        dst = os.path.join(save_dir, f"E{epoch}_T{current_time}.pkl")

        save_data = {
            "epoch": epoch,
            "model_params": model_params,
            # 머신러닝은 모델, 딥러닝은 가중치 저장
            "model_state_dict": self.model,
            "loss": loss,
            "scaler": scaler,
            "label_encoders": label_encoders,
        }

        with open(dst, "wb") as f:
            pickle.dump(save_data, f)
            utils.save_hash(dst)
        print(f"Model saved to {dst}")

