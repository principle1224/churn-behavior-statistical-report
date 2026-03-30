import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter


# ── 공통 전처리 로직 ──────────────────────────────────────────────────────────

def _clean_customers(customers):
    customers['Member Since'] = pd.to_datetime(customers['Member Since'])
    rate_col = customers['Subscription Rate'].astype(str).str.replace('$', '', regex=False).str.strip()
    customers['Subscription Rate'] = pd.to_numeric(rate_col, errors='coerce')
    customers['Cancellation Date'] = pd.to_datetime(customers['Cancellation Date'], errors='coerce')
    customers['Subscription Plan'] = customers['Subscription Plan'].fillna('Basic (Ads)')
    customers['Discount?'] = np.where(customers['Discount?'] == 'Yes', 1, 0)
    # 99.99 타입 오류 수정 (원본 데이터에만 존재)
    mask_typo = customers['Subscription Rate'] == 99.99
    if mask_typo.any():
        customers.loc[mask_typo, 'Subscription Rate'] = 9.99
    # Email: 접두어 제거 (원본 데이터 포맷)
    if customers['Email'].str.startswith('Email: ').any():
        customers['Email'] = customers['Email'].str.replace('Email: ', '', regex=False)
    customers['Cancelled'] = np.where(customers['Cancellation Date'].notna(), 1, 0)
    return customers


def _process_audio_and_history(listening_history, audio):
    # Pop Music → Pop 통일
    audio = audio.copy()
    audio['Genre'] = np.where(audio['Genre'] == 'Pop Music', 'Pop', audio['Genre'])

    # Audio ID 분리 (Song-101 → Type='Song', Audio ID=101)
    audio_clean = pd.DataFrame(audio['ID'].str.split('-').to_list()).rename(
        columns={0: 'Type', 1: 'Audio ID'})
    audio_clean['Audio ID'] = audio_clean['Audio ID'].astype(int)
    audio_all = pd.concat([audio_clean, audio], axis=1)

    df = listening_history.merge(audio_all, how='left', on='Audio ID')
    return audio_all, df


def _build_model_df(customers, df):
    number_of_sessions = (df.groupby('Customer ID')['Session ID']
                          .nunique()
                          .rename('Number of Sessions')
                          .reset_index())

    genres = (pd.concat([df['Customer ID'], pd.get_dummies(df['Genre'])], axis=1)
              .groupby('Customer ID').sum().reset_index())

    total_audio = (df.groupby('Customer ID')['Audio ID']
                   .count().rename('Total Audio').reset_index())

    df_audio = genres.merge(total_audio, how='left', on='Customer ID')

    model_df = customers[['Customer ID', 'Cancelled', 'Discount?']].copy()
    model_df = model_df.merge(number_of_sessions, how='left', on='Customer ID')

    if 'Pop' in df_audio.columns:
        model_df['Percent Pop'] = df_audio['Pop'] / df_audio['Total Audio'] * 100
    else:
        model_df['Percent Pop'] = 0.0

    podcast_cols = [c for c in ['Comedy', 'True Crime'] if c in df_audio.columns]
    if podcast_cols:
        model_df['Percent Podcasts'] = (df_audio[podcast_cols].sum(axis=1) /
                                        df_audio['Total Audio'] * 100)
    else:
        model_df['Percent Podcasts'] = 0.0

    model_df = model_df.fillna(0)
    return model_df


# ── 데이터 로더 ───────────────────────────────────────────────────────────────

def load_and_process_data():
    """원본 데이터 (30명) 로드 및 전처리"""
    customers = pd.read_csv('Data/maven_music_customers.csv')
    listening_history = pd.read_excel('Data/maven_music_listening_history.xlsx')
    audio = pd.read_excel('Data/maven_music_listening_history.xlsx', sheet_name=1)
    sessions = pd.read_excel('Data/maven_music_listening_history.xlsx', sheet_name=2)

    customers = _clean_customers(customers)
    audio_all, df = _process_audio_and_history(listening_history, audio)

    genres = (pd.concat([df['Customer ID'], pd.get_dummies(df['Genre'])], axis=1)
              .groupby('Customer ID').sum().reset_index())
    model_df = _build_model_df(customers, df)

    return customers, listening_history, audio_all, sessions, df, model_df, genres


def load_synthetic_data():
    """합성 데이터 (500명) 로드 및 전처리"""
    customers = pd.read_csv('Data/synthetic_customers.csv')
    listening_history = pd.read_excel('Data/synthetic_listening_history.xlsx', sheet_name=0)
    audio = pd.read_excel('Data/synthetic_listening_history.xlsx', sheet_name=1)
    sessions = pd.read_excel('Data/synthetic_listening_history.xlsx', sheet_name=2)

    customers = _clean_customers(customers)
    audio_all, df = _process_audio_and_history(listening_history, audio)

    genres = (pd.concat([df['Customer ID'], pd.get_dummies(df['Genre'])], axis=1)
              .groupby('Customer ID').sum().reset_index())
    model_df = _build_model_df(customers, df)

    return customers, listening_history, audio_all, sessions, df, model_df, genres


# ── 통계 분석 함수 ────────────────────────────────────────────────────────────

def run_chi_square_test(customers_df):
    """
    할인 여부 × 취소 여부 카이제곱 검정
    Returns: dict with chi2, p_value, dof, contingency_table, cramers_v
    """
    ct = pd.crosstab(customers_df['Discount?'], customers_df['Cancelled'],
                     rownames=['할인 여부'], colnames=['취소 여부'])
    ct.index = ['할인 없음 (0)', '할인 있음 (1)']
    ct.columns = ['유지 (0)', '취소 (1)']

    chi2, p_value, dof, expected = stats.chi2_contingency(ct)

    # Cramer's V (효과 크기)
    n = ct.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

    expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)

    return {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'contingency_table': ct,
        'expected_table': expected_df,
        'cramers_v': cramers_v,
        'significant': p_value < 0.05,
    }


def run_logistic_regression(model_df):
    """
    로지스틱 회귀: Cancelled ~ Discount? + Number of Sessions + Percent Pop + Percent Podcasts
    Returns: dict with features, coefficients, odds_ratios, ci_lower, ci_upper, p_values, accuracy
    """
    features = ['Discount?', 'Number of Sessions', 'Percent Pop', 'Percent Podcasts']
    feature_labels = ['할인 여부', '세션 수', 'Pop 비율 (%)', '팟캐스트 비율 (%)']

    df = model_df[features + ['Cancelled']].dropna()
    X = df[features].values
    y = df['Cancelled'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)

    # 피셔 정보 행렬로 표준오차 계산
    p_hat = model.predict_proba(X_scaled)[:, 1]
    W = p_hat * (1 - p_hat)
    H = X_scaled.T @ np.diag(W) @ X_scaled
    try:
        H_inv = np.linalg.inv(H)
        std_errs = np.sqrt(np.diag(H_inv))
    except np.linalg.LinAlgError:
        std_errs = np.ones(len(coefs)) * 0.1

    z_scores = coefs / std_errs
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    # 95% 신뢰구간 (오즈비 기준)
    ci_lower = np.exp(coefs - 1.96 * std_errs)
    ci_upper = np.exp(coefs + 1.96 * std_errs)

    accuracy = model.score(X_scaled, y)

    return {
        'features': feature_labels,
        'coefficients': coefs,
        'odds_ratios': odds_ratios,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_values': p_values,
        'accuracy': accuracy,
    }


def compute_survival_data(customers_df):
    """
    Kaplan-Meier 생존 분석용 데이터 계산
    T: 가입일 ~ 취소일 또는 관측 종료일 (일 단위)
    E: 취소 여부 (1=취소, 0=관측 중단)
    Returns: dict per segment with T, E arrays and KaplanMeierFitter objects
    """
    OBS_END = pd.Timestamp("2023-08-31")
    df = customers_df.copy()
    df['Member Since'] = pd.to_datetime(df['Member Since'])
    df['Cancellation Date'] = pd.to_datetime(df['Cancellation Date'])

    # 관측 기간 (일)
    df['T'] = df.apply(lambda r: (
        (r['Cancellation Date'] - r['Member Since']).days
        if pd.notna(r['Cancellation Date'])
        else (OBS_END - r['Member Since']).days
    ), axis=1).clip(lower=1)
    df['E'] = df['Cancelled']

    # 세그먼트 분류
    df['Segment'] = df.apply(lambda r: (
        'Premium 할인 ($7.99)' if r['Discount?'] == 1
        else ('Premium 정가 ($9.99)' if r['Subscription Rate'] == 9.99
              else 'Basic ($2.99)')
    ), axis=1)

    results = {}
    segments = ['Basic ($2.99)', 'Premium 정가 ($9.99)', 'Premium 할인 ($7.99)']
    colors = ['#2ca02c', '#1f77b4', '#d62728']

    for seg, color in zip(segments, colors):
        mask = df['Segment'] == seg
        sub = df[mask]
        if len(sub) == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub['T'], sub['E'], label=seg)

        timeline = kmf.timeline
        sf = kmf.survival_function_[seg].values
        ci_lower = kmf.confidence_interval_[f'{seg}_lower_0.95'].values
        ci_upper = kmf.confidence_interval_[f'{seg}_upper_0.95'].values
        median_survival = kmf.median_survival_time_

        results[seg] = {
            'timeline': timeline,
            'survival': sf,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'median': median_survival,
            'color': color,
            'n': len(sub),
        }

    return results
