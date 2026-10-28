# Divorce Dataset을 활용한 이혼 위험도 예측 모델

부부관계 설문 데이터(21개 특성)로 이혼 여부를 예측하는 이진 분류 모델을 만들고,
**전처리 데이터셋 5종 × 모델 6종** 조합의 성능을 비교한 머신러닝 팀 프로젝트입니다.

- 부산대학교 데이터사이언스전문대학원 머신러닝 팀프로젝트 7팀 (강준호, 고권률, 김지윤, 오윤아)
- 2025.11

## 데이터

- **출처**: Kaggle Divorce Dataset (부부관계 설문조사 기반 정량 데이터)
- **크기**: 5,000 커플, 특성 21개 + 타깃 1개
- **타깃**: `divorced` (0: 관계 유지 3,009건 / 1: 이혼 1,991건)
- **특성 구성**: 인구·기본 정보, 재정, 사회 환경, 관계 적합성, 의사소통·감정, 행동·환경 요인
  - 범주형 5개: `education_level`, `employment_status`, `religious_compatibility`,
    `conflict_resolution_style`, `marriage_type` → 원핫 인코딩(`drop_first=True`)

> 데이터 파일은 저장소에 포함하지 않았습니다. Kaggle에서 내려받아 노트북과 같은 폴더에
> `divorce_df.csv`로 두고 실행하세요.

## EDA 요약

- 중복 행 없음
- 타깃과 수치형 변수의 상관계수가 모두 |r| < 0.08 → 단일 변수로 설명되지 않는 복잡한 문제
- 변수 간 상관계수도 모두 |r| ≤ 0.7이지만, VIF 기준으로는 `age_at_marriage`(21.2)가 심각한 수준이고
  `trust_score`, `communication_score`, `combined_income`, `social_support`, `financial_stress_level`이 5~10 사이

## 전처리 데이터셋 5종

| 데이터셋 | 구성 |
|---|---|
| D0 | 원본 |
| D1 | 이상치 제거: 월 갈등 횟수 6회 이상인데 신뢰도 10점인 비논리적 행 제거 (5,000 → 4,994행) |
| D2 | D1 + 특성공학: `money_attitude = combined_income × financial_stress_level` 생성 후 두 원본 열 제거 |
| D3 | D1 + 다중공선성 높은 `age_at_marriage` 제거 |
| D4 | D1 + 특성공학 + `age_at_marriage` 제거 |

## 모델

공통: train/test = 8:2 층화 분할(`random_state=42`), 클래스 불균형 보정(`class_weight='balanced'` 등),
**F1-Score 기준**으로 하이퍼파라미터 탐색.

| 모델 | 설정 |
|---|---|
| L1 로지스틱 회귀 | `LogisticRegressionCV`(saga, 5-fold, C 0.001~100000), 표준화 |
| L2 로지스틱 회귀 | `LogisticRegressionCV`(5-fold), 표준화 |
| LightGBM | Optuna 50 trials + early stopping 30 |
| CatBoost | Optuna 50 trials + early stopping 30 |
| XGBoost | Optuna 50 trials + early stopping 50, `scale_pos_weight` |
| RandomForest | Optuna 40 trials |

정확도는 클래스 편향이 있으면 오해를 부를 수 있어, Precision · Recall · **F1-Score** 중심으로 평가했습니다.

## 결과

### 테스트 F1-Score (모델 × 데이터셋)

| 모델 | D0 원본 | D1 이상치 제거 | D2 +특성공학 | D3 +VIF 열 제거 | D4 +특성공학 +VIF 열 제거 |
|---|---:|---:|---:|---:|---:|
| L1 로지스틱 | 0.5054 | 0.5216 | 0.5349 | 0.5268 | 0.5307 |
| L2 로지스틱 | 0.4921 | 0.5263 | 0.5349 | 0.5239 | 0.5354 |
| LightGBM | 0.5006 | 0.5303 | 0.5482 | 0.4932 | 0.5620 |
| CatBoost | 0.5040 | 0.5298 | 0.5333 | 0.4899 | **0.5650** |
| XGBoost | 0.5113 | 0.5039 | 0.5436 | 0.5345 | 0.5463 |
| RandomForest | 0.4876 | 0.5098 | 0.5158 | 0.5074 | 0.5077 |

### D4 데이터셋 모델별 상세 (테스트)

| 모델 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| L1 로지스틱 | 0.6016 | 0.5000 | 0.5653 | 0.5307 | 0.6228 |
| L2 로지스틱 | 0.6056 | 0.5044 | 0.5704 | 0.5354 | 0.6231 |
| LightGBM | 0.5365 | 0.4507 | 0.7462 | 0.5620 | 0.5958 |
| CatBoost | 0.5145 | 0.4393 | 0.7915 | **0.5650** | 0.5846 |
| XGBoost | 0.5876 | 0.4863 | 0.6231 | 0.5463 | 0.6189 |
| RandomForest | 0.5826 | 0.4788 | 0.5402 | 0.5077 | 0.6017 |

### 주요 변수

- **L1 로지스틱(원본)**: 외도 경험(`infidelity_occurred`), 가정폭력 이력(`domestic_violence_history`)이
  이혼 위험을 높이고, 의사소통 점수(`communication_score`)는 낮추는 방향으로 가장 큰 계수
- **CatBoost(D4, 최고 F1)**: `domestic_violence_history`(43.2), `infidelity_occurred`(31.7),
  `money_attitude`(25.2) 세 변수만 사용하고 나머지 25개 변수의 중요도는 0

## 결론

- 6개 모델 모두 원본(D0)보다 전처리 데이터에서 F1이 올랐고, 모든 모델이 **D2 또는 D4**에서 최고 성능
- 새로 만든 `money_attitude`(소득 × 재정 스트레스)는 CatBoost에서 세 번째로 중요한 변수로 쓰임
- 최고 F1은 **CatBoost + D4 (0.5650)**. 다만 Recall 0.79 / Precision 0.44로 이혼을 넓게 잡는 쪽에
  치우쳐 있고, Accuracy·ROC-AUC는 로지스틱 회귀(D4)가 더 높음
- 전체적으로 ROC-AUC 0.6 안팎으로, 주어진 설문 특성만으로는 이혼 여부를 강하게 구분하기 어려움

## 실행

```bash
pip install -r requirements.txt
jupyter notebook divorce_simple.ipynb
```

노트북은 EDA → 다중공선성 진단 → 데이터셋 생성 → 모델별(`*_result`: 성능, `*_coef`: 계수·중요도) 실행 순서로
구성되어 있으며, 마지막 셀에 테스트 F1-Score 정리가 있습니다.
