import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_processing import load_and_process_data

# 페이지 설정
import streamlit as st

st.set_page_config(
    page_title="이호진 통계 분석 포트폴리오",
    page_icon="🎵",
    layout="wide"
)

st.markdown("""
<style>
header {visibility: hidden;}
.block-container {padding-top: 0rem;}
</style>
""", unsafe_allow_html=True)

# 중앙 정렬된 헤더
st.markdown("""
<div style="text-align: center; padding: 40px 0;">
    <h1 style="font-size: 48px; margin-bottom: 10px;"> 이호진_통계분석 포트폴리오 제출</h1>
    <h3 style="color: #666; font-weight: normal; margin-bottom: 30px;"> 음악 앱 사용자 패턴 분석_고객 이탈 패턴 분석 리포트</h3>
    <div style="max-width: 700px; margin: 0 auto; font-size: 16px; line-height: 1.8;">
        <p>할인 프로모션 후 신규 고객은 늘었지만,<br>
        <strong style="color: #d62728;">할인 고객의 85.7%가 3개월 내 이탈</strong> (정가 고객 30.4%)</p>
        <p style="margin-top: 15px;">본 분석: "할인 고객이 더 빨리 떠나는 현상" 분석 및<br>
        실행 가능한 리텐션 전략을 도출</p>
    </div>
</div>
""", unsafe_allow_html=True)

# 커스텀 CSS
st.markdown("""
<style>
    .big-title {
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 20px;
        color: #666;
        margin-bottom: 40px;
    }
    .section-title {
        font-size: 32px;
        font-weight: bold;
        margin-top: 60px;
        margin-bottom: 20px;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 10px;
    }
    .insight {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
        font-size: 16px;
    }
    .metric-card {
        background-color: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 데이터 로드 및 전처리
@st.cache_data
def get_data():
    return load_and_process_data()

# 데이터 로드
customers, listening_history, audio_all, sessions, df, model_df, genres = get_data()

# ===== HERO SECTION =====
st.markdown('<p class="big-title">Music 앱 사용자 패턴 분석</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Music 앱 사용자 구독 취소 패턴 분석 및 예측_ 레포트</p>', unsafe_allow_html=True)

# ===== EXECUTIVE SUMMARY =====
st.markdown('<p class="section-title"> 핵심 요약</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# 할인 고객 취소율
discount_yes = customers[customers['Discount?']==1]
cancel_rate_discount = discount_yes.Cancelled.sum() / len(discount_yes)

# 비할인 고객 취소율
discount_no = customers[customers['Discount?']==0]
cancel_rate_no_discount = discount_no.Cancelled.sum() / len(discount_no)

# 전체 취소율
overall_cancel_rate = customers['Cancelled'].mean()

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #d62728;">{cancel_rate_discount*100:.1f}%</h3>
        <p>할인 받은 고객의 취소율</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #2ca02c;">{cancel_rate_no_discount*100:.1f}%</h3>
        <p>할인 없는 고객의 취소율</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #1f77b4;">{overall_cancel_rate*100:.1f}%</h3>
        <p>전체 평균 취소율</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="insight">
<strong> 핵심 인사이트</strong><br>
• 할인 제공 고객의 취소율이 일반 고객 대비 <strong>{cancel_rate_discount/cancel_rate_no_discount:.1f}배</strong> 높음<br>
• 세션 활동이 많을수록 취소 가능성 낮아짐<br>
• Pop 장르 선호도가 높은 고객이 가장 많음
</div>
""", unsafe_allow_html=True)

# ===== DATA OVERVIEW =====
st.markdown('<p class="section-title"> 데이터 개요</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("총 고객 수", f"{len(customers):,}명")

with col2:
    st.metric("취소한 고객", f"{customers['Cancelled'].sum()}명")

with col3:
    st.metric("총 청취 세션", f"{df['Session ID'].nunique():,}회")

with col4:
    st.metric("분석 기간", "3개월")

# ===== 고객 구성 =====
st.markdown('<p class="section-title"> 사용자 분포</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # 구독 플랜별 분포
    plan_counts = customers['Subscription Plan'].value_counts()
    fig = px.pie(
        values=plan_counts.values,
        names=plan_counts.index,
        title="구독 플랜 분포",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # 구독 요금별 고객 수
    rate_counts = customers['Subscription Rate'].value_counts().sort_index()
    fig = px.bar(
        x=rate_counts.index,
        y=rate_counts.values,
        title="구독 요금별 고객 수",
        labels={'x': '요금 ($)', 'y': '고객 수'},
        color=rate_counts.values,
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="insight">
<strong> 발견</strong> Premium 플랜($9.99) 고객이 가장 많으며, Basic 플랜은 광고 포함 버전이 주를 이룸
</div>
""", unsafe_allow_html=True)

# ===== 청취 행동 =====
st.markdown('<p class="section-title"> 사용자 음악 사용 패턴</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # 세션 수 분포
    session_data = df.groupby('Customer ID')['Session ID'].nunique()
    fig = px.histogram(
        session_data,
        nbins=30,
        title="사용자 별 청취 세션 수 분포",
        labels={'value': '세션 수', 'count': '사용자 수'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # 장르별 인기도
    genre_counts = df['Genre'].value_counts()
    fig = px.bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation='h',
        title="장르별 청취 횟수",
        labels={'x': '청취 횟수', 'y': '장르'},
        color=genre_counts.values,
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="insight">
<strong> 발견</strong> Pop 장르가 압도적으로 인기 있으며, 대부분 고객이 10~30회 세션을 가짐
</div>
""", unsafe_allow_html=True)

# ===== 취소 패턴 분석 =====
st.markdown('<p class="section-title"> 사용자 구독 취소 패턴 분석</p>', unsafe_allow_html=True)

# 할인 여부에 따른 취소율
cancel_comparison = pd.DataFrame({
    'Customer Type': ['할인 받음', '할인 안받음'],
    'Cancellation Rate': [cancel_rate_discount, cancel_rate_no_discount]
})

fig = px.bar(
    cancel_comparison,
    x='Cancellation Rate',
    y='Customer Type',
    orientation='h',
    title="할인 여부에 따른 취소율 비교",
    labels={'Cancellation Rate': '취소율', 'Customer Type': '고객 유형'},
    color='Cancellation Rate',
    color_continuous_scale='Reds',
    text=[f'{x:.1%}' for x in cancel_comparison['Cancellation Rate']]
)
fig.update_traces(textposition='outside')
st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""
<div class="insight">
<strong> 경고</strong> 할인을 받은 고객의 취소율이 {cancel_rate_discount:.1%}로, 
할인 없는 고객({cancel_rate_no_discount:.1%})보다 <strong>{cancel_rate_discount/cancel_rate_no_discount:.1f}배 높음</strong>
</div>
""", unsafe_allow_html=True)

# ===== 상관관계 분석 =====
st.markdown('<p class="section-title"> 변수 간 상관관계</p>', unsafe_allow_html=True)

# 상관관계 히트맵
corr_matrix = model_df.corr()

fig = px.imshow(
    corr_matrix,
    text_auto='.2f',
    aspect="auto",
    title="변수 간 상관계수 히트맵",
    color_continuous_scale='RdBu_r',
    zmin=-1,
    zmax=1
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="insight">
<strong>해석</strong><br>
• 세션 수와 취소 간 음의 상관관계 존재 (활동적일수록 취소 가능성 낮음)<br>
• 할인과 취소 간 강한 양의 상관관계 확인됨<br>
• Pop 선호도는 취소와 약한 음의 상관관계
</div>
""", unsafe_allow_html=True)

# ===== 핵심 변수 관계 =====
st.markdown('<p class="section-title"> 핵심 변수 관계 시각화</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # 할인 여부와 취소 관계
    fig = px.scatter(
        model_df, 
        x='Discount?', 
        y='Cancelled',
        color='Cancelled',
        title="할인 여부 vs 취소",
        labels={'Discount?': '할인 여부 (0=없음, 1=있음)', 'Cancelled': '취소 여부 (0=유지, 1=취소)'},
        color_continuous_scale='Reds',
        opacity=0.7
    )
    fig.update_traces(marker=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # 세션 수와 취소 관계
    fig = px.scatter(
        model_df, 
        x='Number of Sessions', 
        y='Cancelled',
        color='Cancelled',
        title="세션 수 vs 취소",
        labels={'Number of Sessions': '청취 세션 수', 'Cancelled': '취소 여부 (0=유지, 1=취소)'},
        color_continuous_scale='Reds',
        opacity=0.7
    )
    fig.update_traces(marker=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="insight">
<strong> 해석</strong><br>
• <strong>할인 여부</strong>: 할인 받은 고객(1) 대부분이 취소(노란색 점들이 위쪽 집중)<br>
• <strong>세션 수</strong>: 활동적인 고객(오른쪽)은 대부분 유지, 활동 적은 고객(왼쪽)은 취소 경향
</div>
""", unsafe_allow_html=True)


# ===== 세부 데이터 =====
st.markdown('<p class="section-title"> 원본 데이터 미리보기</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["고객 데이터", "청취 기록", "모델링 데이터"])

with tab1:
    st.dataframe(customers.head(20), use_container_width=True)

with tab2:
    st.dataframe(df.head(20), use_container_width=True)

with tab3:
    st.dataframe(model_df.head(20), use_container_width=True)

# ===== 액션 아이템 =====
st.markdown('<p class="section-title">해결방안 도출</p>', unsafe_allow_html=True)

st.markdown("""
<div style="background-color: #fff3cd; padding: 30px; border-radius: 10px; border-left: 5px solid #ffc107;">
    <h3 style="margin-top: 0;">제안 사항</h3>
    
    <h4>1 할인 정책 재검토</h4>
    <p>• 할인 고객의 높은 이탈률은 가격 민감 고객층 유입을 의미함<br>
    • 할인 대신 첫 달 무료 체험 또는 가치 기반 프로모션 고려</p>
    
    <h4>2 활동 기반 리텐션 전략</h4>
    <p>• 세션 수가 적은 고객에게 맞춤 플레이리스트/추천 제공<br>
    • 월 10회 미만 사용자에게 재참여(Re-engagement) 캠페인 실행</p>
    
    <h4>3 장르 기반 개인화</h4>
    <p>• Pop 선호 고객을 위한 신곡 알림 서비스<br>
    • 팟캐스트 청취자 대상 전용 콘텐츠 확대</p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #666; font-size: 14px;">
 Data Analysis Report | Maven Music Customer Insights | 2026
</p>
""", unsafe_allow_html=True)
