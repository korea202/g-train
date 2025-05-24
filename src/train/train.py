import numpy as np

def train(model, train_loader):
    total_loss = 0
    for features, labels in train_loader:
        predictions = model.forward(features)
        labels = labels.reshape(-1, 1)
        loss = np.mean((predictions - labels) ** 2)
        
        model.backward(features, labels, predictions)

        total_loss += loss

    return total_loss / len(train_loader)


def train(model, train_loader, val_loader):
    total_loss = 0
    for features, labels in train_loader:
        model.fit(features, labels,
            eval_set=[(X_val, y_val)], 
            eval_metric='rmse', 
            categorical_feature='auto', 
            callbacks=[lgbm.early_stopping(stopping_rounds=50), 
                        lgbm.log_evaluation(period=100, show_stdv=True)] )


        # 검증 데이터 예측 및 RMSE 계산
        y_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        rmse_list.append(rmse)
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

        # 평균 RMSE 출력
        print("평균 RMSE:", np.mean(rmse_list))

        if SAVE_MODEL == True: save_model(model)
        return model

def test():
    print('im test')