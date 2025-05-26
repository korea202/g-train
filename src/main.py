import os
import sys

import fire
import wandb
from icecream import ic
from dotenv import load_dotenv, dotenv_values

if (python_path := dotenv_values().get('PYTHONPATH')) and python_path not in sys.path: sys.path.append(python_path)


#1. 데이터 읽기
df = load_data()

#2. 컬럼 셋팅
df = set_columns(df)

#3. 데이터 셋팅
test_input = None

# 제출파일 만들때
if(TRAIN_MODE == False): 
    test_input = load_test()

#train_input, val_input, train_target, val_target, test_input = set_data(df, 0.2, test_input)
train_input, val_input, train_target, val_target, test_input = set_data(df, 0.1, test_input)

#4. 모델 학습
#model = load_model() if TRAIN_MODE == False else train_lgbm_kfold(train_input, train_target, 5)
#model = load_model() if TRAIN_MODE == False else train_lgbm(train_input, val_input, train_target, val_target)
#model = load_model() if TRAIN_MODE == False else train_flaml(train_input, train_target)
model = load_model()

#5. 테스트/etc
print_ex("트레이닝 점수=", model.score(train_input, train_target))
print_ex("검증 점수=", model.score(val_input, val_target))

# feature_importances_
if SHOW_FEATURE_IMPORTANCE == True: show_feature_importance(model, df)

if SHOW_PERMUTATION == True: show_permutation(model, )


if(TRAIN_MODE == False):
    
    y_pred= model.predict(test_input)
    print(y_pred.shape)
    # 예측 결과 일부 확인
    print(f"예측 값: {y_pred[:10]}")
    df_result = pd.DataFrame({'target': np.array(y_pred).astype(int)})
    df_result.to_csv(SAVE_DIR, index=False)
else:

    # 검증 데이터 예측 및 RMSE 출력
    y_pred = model.predict(val_input)
    rmse = np.sqrt(np.mean((val_target - y_pred) ** 2))
    print_ex(f"RMSE: {rmse:.4f}")

    # 예측 결과 일부 확인
    print(f"예측 값: {y_pred[:10]}")
    print(f"실제 값: {val_target[:10]}")

    if SHOW_HITMAP == True: show_hitmap(df)
    if SHOW_SCATTER == True: show_scatter(val_target, y_pred)


send_kakao_message("작업이 완료되었습니다.\n" + work_msg)    