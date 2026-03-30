"""
합성 데이터 생성 스크립트 (한 번만 실행)
원본 30명 데이터의 패턴을 유지하면서 500명으로 확대
"""
import numpy as np
import pandas as pd
from datetime import timedelta
import random

np.random.seed(42)
random.seed(42)

# ── 기본 설정 ────────────────────────────────────────────────────────────────

N = 500  # 생성할 고객 수

# 세그먼트 정의 (원본 비율 유지)
# Basic $2.99: 17/30 → 283명, Premium $9.99: 6/30 → 100명, Premium $7.99(할인): 7/30 → 117명
SEGMENTS = [
    {"plan": "Basic (Ads)",        "rate": 2.99,  "discount": "No",  "n": 283, "cancel_rate": 0.30},
    {"plan": "Premium (No Ads)",   "rate": 9.99,  "discount": "No",  "n": 100, "cancel_rate": 0.50},
    {"plan": "Premium (No Ads)",   "rate": 7.99,  "discount": "Yes", "n": 117, "cancel_rate": 0.857},
]

# 이름 풀
first_names = [
    "Aria", "Beatrice", "Benny", "Bobby", "Carol", "Chord", "Greta",
    "Harmony", "Jazz", "Kiki", "Lyric", "Melody", "Reed", "Rhythm",
    "Rock", "Sonata", "Symphony", "Tempo", "Cadence", "Clef",
    "Allegra", "Solo", "Forte", "Coda", "Treble", "Bass", "Tenor",
    "Alto", "Vivace", "Adagio"
]
last_names = [
    "Greene", "Keys", "Bell", "Bassett", "Dixon", "Saxton", "Sharp",
    "Kingbird", "Nash", "Coleman", "Hayes", "Franklin", "Campbell",
    "Beat", "Parks", "Rhodes", "Bass", "Saunders", "Flat", "Groove",
    "Heart", "Wallace", "Fitzgerald", "Murphy", "Drummond", "Singer",
    "Hunter", "Rivers", "Stone", "Moon", "Star", "Lake", "Brooks",
    "Wood", "Hill", "Lane", "Cross", "Reed", "Banks", "Fields"
]

email_adj = [
    "melodious", "rhythmic", "harmonic", "groovy", "jazzy", "musical",
    "lyrical", "tuneful", "beatful", "melodic", "sonorous", "vibrant",
    "soulful", "dynamic", "classic", "smooth", "funky", "chill",
    "vivid", "resonant", "acoustic", "electric", "digital", "indie",
    "lofi", "hipster", "retro", "modern", "classic", "ambient"
]
email_domains = ["email.com"] * 7 + ["email.edu"] * 3  # 70% .com, 30% .edu

# 오디오 라이브러리 (원본 그대로)
AUDIO_IDS = list(range(101, 113)) + list(range(201, 206))
AUDIO_TYPES = {**{i: "Song" for i in range(101, 113)}, **{i: "Podcast" for i in range(201, 206)}}
# 인기도 기반 가중치
AUDIO_WEIGHTS_RAW = {
    101: 52, 102: 48, 103: 71, 104: 46, 105: 50,
    106: 37, 107: 31, 108: 22, 109: 28, 110: 38,
    111: 28, 112: 20,
    201: 6,  202: 9,  203: 4,  204: 9,  205: 6
}
total_w = sum(AUDIO_WEIGHTS_RAW.values())
AUDIO_PROBS = [AUDIO_WEIGHTS_RAW[a] / total_w for a in AUDIO_IDS]

# ── 고객 데이터 생성 ──────────────────────────────────────────────────────────

def gen_unique_names(n):
    names = set()
    while len(names) < n:
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        names.add(f"{fn} {ln}")
    return list(names)

def gen_email(name):
    adj = random.choice(email_adj)
    part = name.lower().replace(" ", ".")
    domain = random.choice(email_domains)
    return f"Email: {adj}.{part.split('.')[0]}@{domain}"

rows = []
cid_pool = random.sample(range(5000, 20000), N)
cid_pool.sort()
all_names = gen_unique_names(N)
name_idx = 0

for seg in SEGMENTS:
    for i in range(seg["n"]):
        cid = cid_pool[name_idx]
        name = all_names[name_idx]
        email = gen_email(name)

        # 가입일: Basic/Premium 정가 = 3~5월 전체, 할인 = 5월 집중
        if seg["discount"] == "Yes":
            join_dates = pd.date_range("2023-05-01", "2023-05-31", freq="B")
        else:
            join_dates = pd.date_range("2023-03-13", "2023-05-14", freq="B")
            # 3월 가중치 높게
            weights = np.array([3 if d.month == 3 else (2 if d.month == 4 else 1)
                                 for d in join_dates], dtype=float)
            weights /= weights.sum()

        if seg["discount"] == "Yes":
            member_since = random.choice(list(join_dates))
        else:
            member_since = np.random.choice(join_dates, p=weights)

        # 취소 여부
        cancelled = np.random.rand() < seg["cancel_rate"]
        if cancelled:
            # 할인 고객: 16~34일 후 빠른 이탈, 정가 고객: 45~90일 후 이탈
            if seg["discount"] == "Yes":
                days_to_cancel = np.random.randint(16, 35)
            else:
                days_to_cancel = np.random.randint(45, 91)
            cancel_date = pd.Timestamp(member_since) + timedelta(days=int(days_to_cancel))
            cancel_date_str = f"{cancel_date.month}/{cancel_date.day}/{str(cancel_date.year)[2:]}"
        else:
            cancel_date_str = np.nan

        member_str = f"{pd.Timestamp(member_since).month}/{pd.Timestamp(member_since).day}/{str(pd.Timestamp(member_since).year)[2:]}"

        rows.append({
            "Customer ID": cid,
            "Customer Name": name,
            "Email": email,
            "Member Since": member_str,
            "Subscription Plan": seg["plan"],
            "Subscription Rate": f"${seg['rate']:.2f}",
            "Discount?": seg["discount"] if seg["discount"] == "Yes" else np.nan,
            "Cancellation Date": cancel_date_str,
        })
        name_idx += 1

customers_df = pd.DataFrame(rows)
customers_df.to_csv("Data/synthetic_customers.csv", index=False)
print(f"고객 데이터 생성 완료: {len(customers_df)}명")

# 세그먼트별 취소율 확인
for seg in SEGMENTS:
    mask = customers_df["Subscription Rate"] == f"${seg['rate']:.2f}"
    sub = customers_df[mask]
    cr = sub["Cancellation Date"].notna().sum() / len(sub)
    print(f"  {seg['plan']} ${seg['rate']:.2f}: {len(sub)}명, 취소율 {cr:.1%}")

# ── 청취 기록 생성 ──────────────────────────────────────────────────────────

customers_df["Member Since"] = pd.to_datetime(customers_df["Member Since"])
customers_df["Cancellation Date"] = pd.to_datetime(customers_df["Cancellation Date"])

listening_rows = []
session_rows = []
session_id = 200000
OBS_END = pd.Timestamp("2023-08-31")

for _, cust in customers_df.iterrows():
    cid = cust["Customer ID"]
    join_date = cust["Member Since"]
    end_date = cust["Cancellation Date"] if pd.notna(cust["Cancellation Date"]) else OBS_END
    end_date = min(pd.Timestamp(end_date), OBS_END)

    cancelled = pd.notna(cust["Cancellation Date"])

    # 세션 수: 취소 고객 1~15회, 활성 고객 10~50회
    if cancelled:
        n_sessions = np.random.randint(1, 16)
    else:
        n_sessions = np.random.randint(10, 51)

    # 세션 날짜 생성 (가입~종료 사이 균등 분포)
    active_days = (end_date - join_date).days
    if active_days <= 0:
        active_days = 1

    for _ in range(n_sessions):
        session_id += np.random.randint(1, 200)
        day_offset = np.random.randint(0, active_days)
        login_time = join_date + timedelta(days=int(day_offset),
                                           hours=int(np.random.randint(0, 24)))
        session_rows.append({"Session ID": session_id, "Session Log In Time": login_time})

        # 세션 내 트랙 수: 3~15
        n_tracks = np.random.randint(3, 16)
        for order in range(1, n_tracks + 1):
            audio_id = np.random.choice(AUDIO_IDS, p=AUDIO_PROBS)
            listening_rows.append({
                "Customer ID": cid,
                "Session ID": session_id,
                "Audio Order": order,
                "Audio ID": audio_id,
                "Audio Type": AUDIO_TYPES[audio_id],
            })

lh_df = pd.DataFrame(listening_rows)
sessions_df = pd.DataFrame(session_rows)

# 원본 오디오 라이브러리 불러와서 그대로 유지
audio_df = pd.read_excel("Data/maven_music_listening_history.xlsx", sheet_name=1)

# Excel 저장 (3 시트)
with pd.ExcelWriter("Data/synthetic_listening_history.xlsx", engine="openpyxl") as writer:
    lh_df.to_excel(writer, sheet_name="Listening History", index=False)
    audio_df.to_excel(writer, sheet_name="Audio", index=False)
    sessions_df.to_excel(writer, sheet_name="Sessions", index=False)

print(f"\n청취 기록 생성 완료: {len(lh_df):,}행")
print(f"세션 수: {len(sessions_df):,}개")
print(f"고객당 평균 세션 수: {len(sessions_df)/N:.1f}회")
print(f"\n합성 데이터 파일 저장 완료:")
print("  - Data/synthetic_customers.csv")
print("  - Data/synthetic_listening_history.xlsx")
