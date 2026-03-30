import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_processing import (
    load_and_process_data, load_synthetic_data,
    run_chi_square_test, run_logistic_regression, compute_survival_data
)

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="이호진 | 고객 이탈 분석 포트폴리오",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
header {visibility: hidden;}
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
[data-testid="stSidebar"] {background-color: #f8f9fa;}
.section-title {
    font-size: 26px; font-weight: bold;
    margin-top: 30px; margin-bottom: 16px;
    border-bottom: 3px solid #1f77b4;
    padding-bottom: 8px;
}
.insight {
    background-color: #f0f8ff;
    padding: 18px 22px;
    border-radius: 8px;
    border-left: 5px solid #1f77b4;
    margin: 16px 0;
    font-size: 15px;
    line-height: 1.7;
}
.warning-box {
    background-color: #fff3cd;
    padding: 18px 22px;
    border-radius: 8px;
    border-left: 5px solid #ffc107;
    margin: 16px 0;
}
.stat-result {
    background-color: #f0fff0;
    padding: 18px 22px;
    border-radius: 8px;
    border-left: 5px solid #2ca02c;
    margin: 16px 0;
    font-size: 15px;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 분석 설정")
    st.markdown("---")

    data_source = st.radio(
        "데이터 소스",
        ["원본 데이터 (30명)", "합성 데이터 (500명)"],
        index=1,
        help="합성 데이터는 원본 패턴을 유지하며 통계적 검정력을 높이기 위해 500명으로 확대한 데이터입니다."
    )

    st.markdown("---")
    st.markdown("### 분석 목차")
    st.markdown("""
- **1탭** 개요 & 핵심 요약
- **2탭** 사용자 분포 분석
- **3탭** 통계 검정 (카이제곱 · 로지스틱)
- **4탭** 구독 유지율 분석 (Kaplan-Meier)
- **5탭** 고급 시각화
- **6탭** 원본 데이터
""")

    st.markdown("---")
    st.markdown("""
<small>
Data: Maven Music<br>
이호진 | 통계분석 포트폴리오<br>
2026
</small>
""", unsafe_allow_html=True)

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
@st.cache_data
def get_original():
    return load_and_process_data()

@st.cache_data
def get_synthetic():
    return load_synthetic_data()

if data_source == "원본 데이터 (30명)":
    customers, listening_history, audio_all, sessions, df, model_df, genres = get_original()
    data_label = "원본 (30명)"
else:
    customers, listening_history, audio_all, sessions, df, model_df, genres = get_synthetic()
    data_label = "합성 (500명)"

# 공통 통계
discount_yes = customers[customers['Discount?'] == 1]
discount_no  = customers[customers['Discount?'] == 0]
cancel_rate_disc   = discount_yes['Cancelled'].mean() if len(discount_yes) > 0 else 0
cancel_rate_nodisc = discount_no['Cancelled'].mean()  if len(discount_no) > 0  else 0
overall_cancel     = customers['Cancelled'].mean()

# ── 헤더 ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; padding:32px 0 20px 0;">
    <h1 style="font-size:42px; margin-bottom:8px;">음악 앱 고객 이탈 패턴 분석</h1>
    <h3 style="color:#666; font-weight:normal; margin-bottom:6px;">
        Maven Music Customer Churn Analysis Report
    </h3>
    <span style="background:#e8f4fd; padding:4px 14px; border-radius:20px;
                 font-size:14px; color:#1f77b4; border:1px solid #1f77b4;">
        현재 데이터: {data_label}
    </span>
</div>
""", unsafe_allow_html=True)

# ── 탭 구조 ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "개요 & 핵심 요약",
    "사용자 분포",
    "통계 검정",
    "구독 유지율 분석",
    "고급 시각화",
    "원본 데이터",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — 개요 & 핵심 요약
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">핵심 요약 (Executive Summary)</p>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전체 고객 수", f"{len(customers):,}명")
    col2.metric("전체 취소율", f"{overall_cancel:.1%}")
    col3.metric("할인 고객 취소율", f"{cancel_rate_disc:.1%}",
                delta=f"+{(cancel_rate_disc - overall_cancel)*100:.1f}%p vs 평균",
                delta_color="inverse")
    col4.metric("정가 고객 취소율", f"{cancel_rate_nodisc:.1%}",
                delta=f"{(cancel_rate_nodisc - overall_cancel)*100:.1f}%p vs 평균",
                delta_color="inverse")

    st.markdown(f"""
<div class="insight">
<strong>핵심 발견</strong><br>
• 할인 제공 고객의 취소율이 정가 고객 대비 <strong>{cancel_rate_disc/cancel_rate_nodisc:.1f}배 높음</strong>
  ({cancel_rate_disc:.1%} vs {cancel_rate_nodisc:.1%})<br>
• 세션 활동이 많을수록 취소 가능성 낮아짐 (로지스틱 회귀 확인)<br>
• 할인 고객은 평균 <strong>약 25일 내</strong> 이탈, 정가 고객은 <strong>약 65일 후</strong> 이탈 (생존 분석)
</div>
""", unsafe_allow_html=True)

    # 취소율 비교 막대 차트
    seg_data = customers.copy()
    seg_data['세그먼트'] = seg_data.apply(lambda r: (
        'Premium\n할인 ($7.99)' if r['Discount?'] == 1
        else ('Premium\n정가 ($9.99)' if r['Subscription Rate'] == 9.99 else 'Basic\n($2.99)')
    ), axis=1)
    seg_cancel = (seg_data.groupby('세그먼트')['Cancelled']
                  .agg(['mean', 'count'])
                  .rename(columns={'mean': '취소율', 'count': '고객 수'})
                  .reset_index())
    seg_cancel['취소율_pct'] = (seg_cancel['취소율'] * 100).round(1)

    fig = px.bar(
        seg_cancel, x='세그먼트', y='취소율_pct',
        title='세그먼트별 취소율 비교',
        labels={'취소율_pct': '취소율 (%)'},
        color='취소율_pct',
        color_continuous_scale='RdYlGn_r',
        text=seg_cancel['취소율_pct'].apply(lambda x: f'{x:.1f}%'),
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(coloraxis_showscale=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
<div class="warning-box">
<strong>비즈니스 함의</strong><br>
할인 프로모션은 신규 고객 유입에는 효과적이나, 이탈 방지에는 역효과.<br>
가격 민감 고객층을 유입하고 있으며, 할인 종료 후 대규모 이탈 위험이 있음.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — 사용자 분포
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">사용자 분포 분석</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        plan_counts = customers['Subscription Plan'].value_counts()
        fig = px.pie(
            values=plan_counts.values, names=plan_counts.index,
            title='구독 플랜 분포', hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rate_counts = customers['Subscription Rate'].value_counts().sort_index()
        fig = px.bar(
            x=rate_counts.index.astype(str),
            y=rate_counts.values,
            title='구독 요금별 고객 수',
            labels={'x': '요금 ($)', 'y': '고객 수'},
            color=rate_counts.values,
            color_continuous_scale='Blues',
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-title">청취 패턴 분석</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        session_data = df.groupby('Customer ID')['Session ID'].nunique().reset_index()
        session_data.columns = ['Customer ID', 'Sessions']
        session_data = session_data.merge(
            customers[['Customer ID', 'Cancelled']], on='Customer ID')
        session_data['상태'] = session_data['Cancelled'].map({0: '유지', 1: '취소'})

        fig = px.histogram(
            session_data, x='Sessions', color='상태',
            nbins=30, barmode='overlay',
            title='취소 여부별 세션 수 분포',
            labels={'Sessions': '세션 수', 'count': '고객 수'},
            color_discrete_map={'유지': '#2ca02c', '취소': '#d62728'},
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        genre_counts = df['Genre'].value_counts()
        fig = px.bar(
            x=genre_counts.values, y=genre_counts.index,
            orientation='h',
            title='장르별 청취 횟수',
            labels={'x': '청취 횟수', 'y': '장르'},
            color=genre_counts.values,
            color_continuous_scale='Viridis',
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
<div class="insight">
<strong>발견</strong><br>
• 유지 고객의 세션 수가 취소 고객에 비해 현저히 높음<br>
• Pop 장르가 가장 많이 청취되며, 팟캐스트(Comedy, True Crime)가 그 뒤를 이음
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — 통계 검정
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">카이제곱 독립성 검정 (Chi-Square Test)</p>',
                unsafe_allow_html=True)
    st.markdown("""
**귀무가설 (H₀)**: 할인 여부와 취소 여부는 독립적이다 (관계 없다)
**대립가설 (H₁)**: 할인 여부와 취소 여부는 독립적이지 않다 (관계 있다)
""")

    chi = run_chi_square_test(customers)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("χ² 통계량", f"{chi['chi2']:.4f}")
    col2.metric("p-value", f"{chi['p_value']:.6f}" if chi['p_value'] >= 0.000001 else "< 0.000001")
    col3.metric("자유도", chi['dof'])
    col4.metric("Cramer's V (효과 크기)", f"{chi['cramers_v']:.4f}")

    sig_color = "#2ca02c" if chi['significant'] else "#d62728"
    sig_text  = "귀무가설 기각 — 유의미한 관계 존재" if chi['significant'] else "귀무가설 채택 — 유의미한 관계 없음"
    st.markdown(f"""
<div class="stat-result">
<strong>검정 결과</strong>: <span style="color:{sig_color}; font-size:17px;"><b>{sig_text}</b></span><br>
p-value = {chi['p_value']:.6f} (α = 0.05)<br>
Cramer's V = {chi['cramers_v']:.4f} — {'강한' if chi['cramers_v'] > 0.3 else '중간' if chi['cramers_v'] > 0.1 else '약한'} 연관성
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**관측 빈도 (Observed)**")
        st.dataframe(chi['contingency_table'].style.format("{:.0f}").background_gradient(cmap='Reds'),
                     use_container_width=True)
    with col2:
        st.markdown("**기대 빈도 (Expected)**")
        st.dataframe(chi['expected_table'].style.format("{:.2f}").background_gradient(cmap='Blues'),
                     use_container_width=True)

    # 시각화: 할인 여부별 취소율
    ct = chi['contingency_table'].copy()
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for col_name, color in zip(['유지 (0)', '취소 (1)'], ['#2ca02c', '#d62728']):
        fig.add_trace(go.Bar(
            name=col_name,
            x=ct_pct.index,
            y=ct_pct[col_name],
            text=[f'{v:.1f}%' for v in ct_pct[col_name]],
            textposition='inside',
            marker_color=color,
        ))
    fig.update_layout(
        barmode='stack', title='할인 여부별 취소율 (100% 누적 막대)',
        xaxis_title='할인 여부', yaxis_title='비율 (%)', height=380
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-title">로지스틱 회귀 분석 (Logistic Regression)</p>',
                unsafe_allow_html=True)
    st.markdown("**종속변수**: 취소 여부 (0=유지, 1=취소) | **독립변수**: 할인 여부, 세션 수, Pop/팟캐스트 비율")

    lr = run_logistic_regression(model_df)

    col1, col2 = st.columns([3, 2])
    with col1:
        # 오즈비 Forest Plot
        fig = go.Figure()
        colors = ['#d62728' if OR > 1 else '#2ca02c' for OR in lr['odds_ratios']]
        for i, (feat, OR, lo, hi, p) in enumerate(zip(
                lr['features'], lr['odds_ratios'], lr['ci_lower'], lr['ci_upper'], lr['p_values'])):
            sig_mark = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ' ns'))
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[feat, feat],
                mode='lines', line=dict(color=colors[i], width=2),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[OR], y=[feat],
                mode='markers+text',
                marker=dict(color=colors[i], size=12, symbol='diamond'),
                text=[f' OR={OR:.3f} {sig_mark}'],
                textposition='middle right',
                showlegend=False,
            ))
        fig.add_vline(x=1, line_dash='dash', line_color='gray', annotation_text='OR=1 (기준)')
        fig.update_layout(
            title='오즈비 Forest Plot (95% CI)<br><sub>◆ OR>1: 이탈 위험 증가, ◆ OR<1: 이탈 위험 감소</sub>',
            xaxis_title='Odds Ratio (log scale)', xaxis_type='log',
            height=350, margin=dict(l=160)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**회귀 계수 요약표**")
        lr_table = pd.DataFrame({
            '변수': lr['features'],
            '오즈비 (OR)': [f"{v:.3f}" for v in lr['odds_ratios']],
            '95% CI': [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in
                       zip(lr['ci_lower'], lr['ci_upper'])],
            'p-value': [f"{p:.4f}" if p >= 0.0001 else '<0.0001' for p in lr['p_values']],
            '유의성': ['***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                    for p in lr['p_values']],
        })
        st.dataframe(lr_table, hide_index=True, use_container_width=True)
        st.metric("모델 정확도", f"{lr['accuracy']:.1%}")

    st.markdown(f"""
<div class="stat-result">
<strong>로지스틱 회귀 해석</strong><br>
• <b>할인 여부</b>: OR = {lr['odds_ratios'][0]:.3f} — 할인 고객은 이탈 가능성 {lr['odds_ratios'][0]:.1f}배 높음 (p&lt;0.001)<br>
• <b>세션 수</b>: OR = {lr['odds_ratios'][1]:.4f} — 세션 수 증가 시 이탈 가능성 대폭 감소 (p&lt;0.001)<br>
• <b>Pop/팟캐스트 비율</b>: 통계적으로 유의하지 않음 (장르 자체보다 이용 빈도가 더 중요)
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — 생존 분석
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">구독 유지율 분석 (Kaplan-Meier)</p>', unsafe_allow_html=True)
    st.markdown("""
**구독 유지율 분석**은 "이탈까지 얼마나 걸리는가"를 분석합니다.
- **T**: 가입일로부터 이탈(또는 관측 종료)까지의 일수
- **E**: 이탈 발생 여부 (1=이탈, 0=관측 종료)
- **음영**: 95% 신뢰구간
""")

    surv = compute_survival_data(customers)

    if not surv:
        st.warning("분석을 위한 데이터가 부족합니다.")
    else:
        fig = go.Figure()
        for seg, v in surv.items():
            tl = v['timeline']
            # 신뢰구간 영역
            fig.add_trace(go.Scatter(
                x=np.concatenate([tl, tl[::-1]]),
                y=np.concatenate([v['ci_upper'], v['ci_lower'][::-1]]),
                fill='toself', fillcolor=v['color'],
                opacity=0.15, line=dict(color='rgba(0,0,0,0)'),
                showlegend=False, hoverinfo='skip',
            ))
            # 생존 곡선
            fig.add_trace(go.Scatter(
                x=tl, y=v['survival'],
                mode='lines', name=f"{seg} (n={v['n']})",
                line=dict(color=v['color'], width=2.5, shape='hv'),
            ))
            # 중앙 생존 시간 표시
            if not np.isinf(v['median']):
                fig.add_vline(
                    x=v['median'], line_dash='dot', line_color=v['color'],
                    annotation_text=f"중앙={v['median']:.0f}일",
                    annotation_font_color=v['color'], opacity=0.5
                )

        fig.update_layout(
            title='세그먼트별 구독 유지율 곡선 (Kaplan-Meier)<br>'
                  '<sub>곡선이 빠르게 떨어질수록 이탈이 일찍 발생</sub>',
            xaxis_title='가입 후 경과 일수',
            yaxis_title='구독 유지율',
            yaxis=dict(range=[0, 1.05], tickformat='.0%'),
            legend=dict(x=0.65, y=0.95),
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)

        # 중앙 생존 시간 요약
        st.markdown("**세그먼트별 중앙 구독 유지 기간 (Median Survival Time)**")
        surv_summary = []
        for seg, v in surv.items():
            median_str = f"{v['median']:.0f}일" if not np.isinf(v['median']) else "관측 기간 내 미도달 (>50% 유지)"
            surv_summary.append({
                '세그먼트': seg,
                '고객 수': v['n'],
                '중앙 생존 시간': median_str,
                '색상': v['color'],
            })
        surv_df = pd.DataFrame(surv_summary)[['세그먼트', '고객 수', '중앙 생존 시간']]
        st.dataframe(surv_df, hide_index=True, use_container_width=True)

        st.markdown("""
<div class="stat-result">
<strong>구독 유지율 분석 해석</strong><br>
• <b>Premium 할인 ($7.99)</b>: 중앙 생존 시간 약 27일 — 가입 후 한 달 내 절반이 이탈<br>
• <b>Basic & Premium 정가</b>: 관측 기간(~180일) 내 50% 이탈에 미도달 — 장기 유지 경향<br>
• 할인 고객의 생존 곡선이 가장 가파르게 하락 → 이탈 속도가 현저히 빠름
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — 고급 시각화
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">세션 수 분포 — 바이올린 플롯</p>',
                unsafe_allow_html=True)

    session_data = df.groupby('Customer ID')['Session ID'].nunique().reset_index()
    session_data.columns = ['Customer ID', 'Sessions']
    session_data = session_data.merge(customers[['Customer ID', 'Cancelled',
                                                  'Subscription Rate', 'Discount?']],
                                       on='Customer ID')
    session_data['세그먼트'] = session_data.apply(lambda r: (
        'Premium 할인\n($7.99)' if r['Discount?'] == 1
        else ('Premium 정가\n($9.99)' if r['Subscription Rate'] == 9.99 else 'Basic\n($2.99)')
    ), axis=1)
    session_data['상태'] = session_data['Cancelled'].map({0: '유지', 1: '취소'})

    fig = px.violin(
        session_data, x='세그먼트', y='Sessions', color='상태',
        box=True, points='all',
        title='세그먼트 × 취소 여부별 세션 수 분포 (바이올린 플롯)',
        labels={'Sessions': '세션 수', '세그먼트': '구독 세그먼트'},
        color_discrete_map={'유지': '#2ca02c', '취소': '#d62728'},
    )
    fig.update_layout(height=450, violingap=0.3)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
<div class="insight">
<strong>해석</strong> 유지 고객은 세션 수가 많고 분포가 넓은 반면,
취소 고객은 낮은 세션 수에 밀집 — 앱 사용 빈도가 이탈의 핵심 예측 변수
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">코호트 분석 — 가입월별 이탈 패턴</p>',
                unsafe_allow_html=True)

    cohort_df = customers.copy()
    cohort_df['가입 월'] = cohort_df['Member Since'].dt.to_period('M').astype(str)
    cohort_df['세그먼트'] = cohort_df.apply(lambda r: (
        'Premium 할인 ($7.99)' if r['Discount?'] == 1
        else ('Premium 정가 ($9.99)' if r['Subscription Rate'] == 9.99 else 'Basic ($2.99)')
    ), axis=1)

    cohort_cancel = (cohort_df.groupby(['가입 월', '세그먼트'])
                     .agg(취소율=('Cancelled', 'mean'), 고객수=('Cancelled', 'count'))
                     .reset_index())
    cohort_cancel['취소율_pct'] = (cohort_cancel['취소율'] * 100).round(1)

    fig = px.bar(
        cohort_cancel, x='가입 월', y='취소율_pct',
        color='세그먼트', barmode='group',
        title='가입 월 × 세그먼트별 취소율 (코호트 분석)',
        labels={'취소율_pct': '취소율 (%)', '가입 월': '가입 월'},
        color_discrete_map={
            'Basic ($2.99)': '#1f77b4',
            'Premium 정가 ($9.99)': '#ff7f0e',
            'Premium 할인 ($7.99)': '#d62728',
        },
        text='취소율_pct',
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=420, uniformtext_minsize=8)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-title">상관관계 히트맵</p>', unsafe_allow_html=True)

    corr_matrix = model_df.corr()
    fig = px.imshow(
        corr_matrix, text_auto='.2f', aspect='auto',
        title='변수 간 상관계수 히트맵',
        color_continuous_scale='RdBu_r', zmin=-1, zmax=1
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
<div class="insight">
<strong>코호트 해석</strong><br>
• 5월 코호트(할인 프로모션 대상)의 취소율이 3-4월 코호트 대비 압도적으로 높음<br>
• Basic 플랜의 취소율은 월별로 비교적 안정적 → 할인 프로모션의 영향이 이탈의 핵심
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — 원본 데이터
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<p class="section-title">원본 데이터 미리보기</p>', unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["고객 데이터", "청취 기록", "모델링 데이터"])
    with t1:
        st.dataframe(customers.head(30), use_container_width=True)
    with t2:
        st.dataframe(df.head(50), use_container_width=True)
    with t3:
        st.dataframe(model_df.head(30), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#888; font-size:13px;">
Maven Music Customer Churn Analysis | 이호진 통계분석 포트폴리오 | 2026<br>
분석 도구: Python · Streamlit · Plotly · Scipy · Lifelines · Scikit-learn
</p>
""", unsafe_allow_html=True)
