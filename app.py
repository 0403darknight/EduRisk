import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="EduRisk — Student Dropout Intelligence",
    page_icon="🎓", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Sora',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1.8rem 2.2rem 3rem;max-width:1400px;}
[data-testid="stSidebar"]{background:#0c0f1a!important;border-right:1px solid #1c2035;}
[data-testid="stSidebar"] *{color:#8892b0!important;}
div[data-testid="metric-container"]{background:linear-gradient(135deg,#131929,#1a2035);border:1px solid #1e2a42;border-radius:16px;padding:1.2rem 1.4rem 1rem;transition:all .25s;position:relative;overflow:hidden;}
div[data-testid="metric-container"]::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#5c6bc0,#7c3aed);}
div[data-testid="metric-container"]:hover{border-color:#3d4fd6;transform:translateY(-2px);box-shadow:0 8px 24px rgba(92,107,192,.2);}
div[data-testid="metric-container"] label{color:#6878a0!important;font-size:.7rem!important;text-transform:uppercase;letter-spacing:.1em;font-weight:600;}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#e2e8f8!important;font-size:2rem!important;font-weight:700;}
div[data-testid="metric-container"] [data-testid="stMetricDelta"]{font-size:.75rem!important;}
.sh{font-size:.95rem;font-weight:600;color:#cdd5f0;border-left:3px solid #5c6bc0;padding-left:10px;margin:1.6rem 0 .9rem;}
.card{background:linear-gradient(135deg,#131929,#1a2035);border:1px solid #1e2a42;border-radius:16px;padding:1.3rem 1.5rem;margin-bottom:.9rem;}
.badge{display:inline-flex;align-items:center;gap:5px;border-radius:20px;padding:4px 14px;font-size:.75rem;font-weight:600;}
.b-red{background:#2d0f0f;color:#f87171;border:1px solid #5c1a1a;}
.b-yellow{background:#2d200a;color:#fbbf24;border:1px solid #7a4f0d;}
.b-green{background:#0a2d14;color:#4ade80;border:1px solid #14532d;}
.b-blue{background:#0d1e3d;color:#60a5fa;border:1px solid #1e3a6e;}
.pred-wrap{border-radius:20px;padding:2.2rem 1.8rem;text-align:center;margin:1rem 0;}
.pd{background:radial-gradient(ellipse at 50% 0%,#2d0a0a,#130606);border:1px solid #5c1a1a;}
.pe{background:radial-gradient(ellipse at 50% 0%,#0a1a3d,#060b1a);border:1px solid #1e3a8a;}
.pg{background:radial-gradient(ellipse at 50% 0%,#0a2d14,#061309);border:1px solid #14532d;}
.pred-wrap h2{font-size:2.2rem;font-weight:800;margin:.4rem 0 0;}
.pd h2{color:#f87171;}.pe h2{color:#60a5fa;}.pg h2{color:#4ade80;}
.iv{background:#131929;border-left:3px solid #5c6bc0;border-radius:0 10px 10px 0;padding:.85rem 1.1rem;margin:.45rem 0;font-size:.88rem;color:#a8b4d0;line-height:1.5;}
.div{border:none;border-top:1px solid #1c2035;margin:1.4rem 0;}
.ptitle{font-size:1.8rem;font-weight:800;color:#e2e8f8;margin:0 0 4px;letter-spacing:-.02em;}
.psub{font-size:.88rem;color:#6878a0;margin:0 0 1.5rem;}
div[data-baseweb="select"]>div{background:#131929!important;border-color:#1e2a42!important;}
div[data-baseweb="select"] *{color:#cdd5f0!important;}
.stNumberInput input{background:#131929!important;color:#cdd5f0!important;border-color:#1e2a42!important;}
.stButton>button{background:linear-gradient(135deg,#4f5fd6,#7c3aed);color:white;border:none;border-radius:12px;padding:.75rem 2.5rem;font-weight:700;font-family:'Sora',sans-serif;font-size:.95rem;transition:all .25s;width:100%;}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 6px 24px rgba(92,107,192,.5);}
[data-testid="stForm"]{background:#0f1420;border:1px solid #1c2035;border-radius:18px;padding:1.5rem 1.8rem;}
</style>
""", unsafe_allow_html=True)

# ── LOAD ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all():
    m   = joblib.load('models/best_model.pkl')
    sc  = joblib.load('models/scaler.pkl')
    km  = joblib.load('models/kmeans.pkl')
    pc  = joblib.load('models/pca.pkl')
    fc  = joblib.load('models/feature_cols.pkl')
    with open('models/results.json') as f:
        res = json.load(f)
    df    = pd.read_csv('data/processed_dataset.csv')
    pcadf = pd.read_csv('models/pca_data.csv')
    clu   = pd.read_csv('models/cluster_profiles.csv')
    # Normalize column names — handles both old and new formats
    clu.columns = [c.strip().replace(' ','_') for c in clu.columns]
    clu.rename(columns={
        'Avg_Sem1_Grade_x':'Avg_Sem1_Grade','Avg Sem1 Grade':'Avg_Sem1_Grade',
        'Avg_Sem2_Grade_x':'Avg_Sem2_Grade','Avg Sem2 Grade':'Avg_Sem2_Grade',
        'Avg Attendance':'Avg_Attendance',
        'Dropout Rate (%)':'Dropout_Rate','Dropout_Rate_(%)':'Dropout_Rate',
    }, inplace=True)
    return m, sc, km, pc, fc, res, df, pcadf, clu

with st.spinner("Loading EduRisk..."):
    model, scaler, kmeans, pca_obj, feat_cols, results, df, pca_df, clusters = load_all()

stats = results['dataset_stats']
rmap  = results['risk_map']
BG, PBG, GRD, TXT = '#0f1420','#131929','#1c2035','#a8b4d0'
COLS = ['#f87171','#60a5fa','#4ade80','#fbbf24','#c084fc','#34d399','#f97316']

def dark(fig, title='', h=340):
    fig.update_layout(
        plot_bgcolor=PBG, paper_bgcolor=BG,
        font=dict(family='Sora', color=TXT, size=12),
        title=dict(text=title, font=dict(size=13, color='#cdd5f0'), x=0.01, xanchor='left', y=0.97),
        height=h, margin=dict(l=8, r=8, t=38 if title else 8, b=8),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=GRD, font=dict(size=11), orientation='h', y=-0.12),
        xaxis=dict(gridcolor=GRD, zerolinecolor=GRD, color=TXT, linecolor=GRD, showline=True),
        yaxis=dict(gridcolor=GRD, zerolinecolor=GRD, color=TXT, linecolor=GRD, showline=True),
    )
    return fig

def safe_predict(model, X_scaled):
    pred = model.predict(X_scaled)[0]
    try:
        proba = model.predict_proba(X_scaled)[0]
    except Exception:
        proba = np.array([0.8,0.1,0.1]) if pred==0 else (np.array([0.1,0.8,0.1]) if pred==1 else np.array([0.1,0.1,0.8]))
    return pred, proba

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 1.8rem;">
        <div style="font-size:2.8rem;margin-bottom:.4rem;">🎓</div>
        <div style="font-size:1.25rem;font-weight:800;color:#e2e8f8;letter-spacing:-.02em;">EduRisk</div>
        <div style="font-size:.68rem;color:#5c6bc0;text-transform:uppercase;letter-spacing:.12em;margin-top:3px;">Student Dropout Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "📊  Dashboard",
        "🔍  Predict Student",
        "🤖  Model Analytics",
        "👥  Cluster Analysis",
        "📋  About",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1c2035;margin:1rem 0;'>", unsafe_allow_html=True)
    best_acc = results['model_results'][results['best_model_name']]['accuracy']
    st.markdown(f"""
    <div style="font-size:.75rem;color:#4a5a7a;line-height:2.2;">
      🎯 Best Accuracy &nbsp;<span style="color:#4ade80;font-weight:600;">{best_acc}%</span><br>
      🏆 Best Model &nbsp;<span style="color:#8892b0;">{results['best_model_name']}</span><br>
      📁 Records &nbsp;<span style="color:#8892b0;">{stats['total_students']:,}</span><br>
      🧠 Models &nbsp;<span style="color:#8892b0;">5 ML + 1 ANN</span>
    </div>
    """, unsafe_allow_html=True)
    

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊  Dashboard":
    st.markdown('<div class="ptitle">Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">Real-time academic insights across 2,000 student records</div>', unsafe_allow_html=True)

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Total Students", f"{stats['total_students']:,}")
    k2.metric("Dropouts",       f"{stats['dropout_count']:,}",  f"▼ {stats['dropout_rate']}%",  delta_color="inverse")
    k3.metric("Enrolled",       f"{stats['enrolled_count']:,}")
    k4.metric("Graduates",      f"{stats['graduate_count']:,}", f"▲ {stats['graduate_rate']}%")
    k5.metric("Avg Sem1 Grade", f"{stats['avg_sem1_grade']}")
    k6.metric("Avg Attendance", f"{stats['avg_attendance']}%")
    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.1, 1.9])
    with c1:
        st.markdown('<div class="sh">Outcome Distribution</div>', unsafe_allow_html=True)
        cnts = df['Target'].value_counts().sort_index()
        fig_d = go.Figure(go.Pie(
            labels=['Dropout','Enrolled','Graduate'],
            values=[cnts.get(i,0) for i in range(3)],
            hole=0.65,
            marker=dict(colors=['#f87171','#60a5fa','#4ade80'], line=dict(color=BG, width=3)),
            textinfo='percent+label', textfont=dict(size=12),
            hovertemplate='<b>%{label}</b><br>%{value} students<br>%{percent}<extra></extra>'
        ))
        fig_d.add_annotation(text=f"<b>{stats['dropout_rate']}%</b><br>Dropout",
            x=0.5, y=0.5, showarrow=False, font=dict(size=16, color='#f87171'), align='center')
        dark(fig_d, h=310)
        fig_d.update_layout(showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_d, use_container_width=True)

    with c2:
        st.markdown('<div class="sh">Grade Distribution by Outcome</div>', unsafe_allow_html=True)
        lmap = {0:'Dropout',1:'Enrolled',2:'Graduate'}
        df['Label'] = df['Target'].map(lmap)
        clr = {'Dropout':'#f87171','Enrolled':'#60a5fa','Graduate':'#4ade80'}
        fig_v = go.Figure()
        for lbl in ['Dropout','Enrolled','Graduate']:
            sub = df[df['Label']==lbl]
            fig_v.add_trace(go.Violin(y=sub['Curricular units 2nd sem (grade)'],
                name=lbl, line_color=clr[lbl], fillcolor=clr[lbl], opacity=0.25,
                box_visible=True, meanline_visible=True, points='outliers'))
        dark(fig_v, "2nd Semester Grade by Outcome", h=310)
        st.plotly_chart(fig_v, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="sh">Dropout Rate by Attendance Band</div>', unsafe_allow_html=True)
        bins=[0,40,50,60,70,80,100]; labs=['0-40','40-50','50-60','60-70','70-80','80-100']
        df['AttBin'] = pd.cut(df['Attendance (%)'], bins=bins, labels=labs, right=True)
        att = df.groupby('AttBin', observed=True).agg(
            rate=('Target', lambda x:(x==0).mean()*100), n=('Target','count')).reset_index()
        fig_att = go.Figure(go.Bar(
            x=att['AttBin'].astype(str), y=att['rate'].round(1),
            marker=dict(color=att['rate'],
                        colorscale=[[0,'#4ade80'],[0.45,'#fbbf24'],[1,'#f87171']],showscale=False),
            text=att['rate'].round(1).astype(str)+'%', textposition='outside',
            textfont=dict(size=11, color='#cdd5f0')))
        dark(fig_att, h=310)
        fig_att.update_xaxes(title_text='Attendance Band (%)'); fig_att.update_yaxes(title_text='Dropout Rate (%)')
        st.plotly_chart(fig_att, use_container_width=True)

    with c4:
        st.markdown('<div class="sh">Grade Trajectory (Sem1 → Sem2)</div>', unsafe_allow_html=True)
        trd = df.groupby('Label')[['Curricular units 1st sem (grade)','Curricular units 2nd sem (grade)']].mean().round(2)
        fig_tr = go.Figure()
        for lbl in ['Dropout','Enrolled','Graduate']:
            v = [trd.loc[lbl,'Curricular units 1st sem (grade)'], trd.loc[lbl,'Curricular units 2nd sem (grade)']]
            fig_tr.add_trace(go.Scatter(
                x=['Semester 1','Semester 2'], y=v, mode='lines+markers+text', name=lbl,
                line=dict(color=clr[lbl], width=2.5),
                marker=dict(size=10, color=clr[lbl], line=dict(width=2, color=BG)),
                text=[f'{v[0]:.1f}',f'{v[1]:.1f}'], textposition=['bottom center','bottom center'],
                textfont=dict(color=clr[lbl], size=12)))
        dark(fig_tr, "Average Grade Trend by Outcome", h=310)
        fig_tr.update_yaxes(title_text='Average Grade', range=[11,15])
        st.plotly_chart(fig_tr, use_container_width=True)

    st.markdown('<div class="sh">Key Risk Factors</div>', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    def rfbar(col, gcol, lmap2, title, colors):
        gb = df.groupby(gcol)['Target'].apply(lambda x:(x==0).mean()*100).round(1).reset_index()
        gb['G'] = gb[gcol].map(lmap2)
        fig = go.Figure(go.Bar(x=gb['G'], y=gb['Target'], marker_color=colors,
            text=gb['Target'].astype(str)+'%', textposition='outside', textfont=dict(size=12)))
        dark(fig, title, h=260); col.plotly_chart(fig, use_container_width=True)

    rfbar(r1,'Scholarship holder',{0:'No Scholarship',1:'Scholarship'},'Scholarship Effect',['#f87171','#4ade80'])
    rfbar(r2,'Debtor',{0:'No Debt',1:'Has Debt'},'Debt Effect',['#4ade80','#f87171'])
    rfbar(r3,'Tuition fees up to date',{0:'Fees Overdue',1:'Fees Paid'},'Tuition Status',['#f87171','#4ade80'])
    rfbar(r4,'Gender',{0:'Female',1:'Male'},'Gender Breakdown',['#c084fc','#60a5fa'])

    st.markdown('<div class="sh">Grade vs Attendance — Dropout Landscape</div>', unsafe_allow_html=True)
    samp = df.sample(min(600,len(df)), random_state=1)
    fig_sc = px.scatter(samp, x='Attendance (%)', y='Curricular units 2nd sem (grade)',
        color='Label', color_discrete_map=clr, opacity=0.65)
    fig_sc.update_traces(marker=dict(size=7))
    dark(fig_sc, h=340)
    fig_sc.update_xaxes(title_text='Attendance (%)')
    fig_sc.update_yaxes(title_text='2nd Semester Grade')
    fig_sc.update_layout(legend=dict(orientation='h', y=1.08, x=0))
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown('<div class="sh">Correlation Heatmap — Academic Features</div>', unsafe_allow_html=True)
    top_f = ['Attendance (%)','Curricular units 1st sem (grade)','Curricular units 2nd sem (grade)',
             'Curricular units 1st sem (approved)','Curricular units 2nd sem (approved)',
             'Admission grade','Age at enrollment','Debtor','Scholarship holder',
             'Tuition fees up to date','Target']
    corr = df[top_f].corr().round(2)
    fig_hm = px.imshow(corr, text_auto=True, aspect='auto',
        color_continuous_scale=[[0,'#f87171'],[0.5,PBG],[1,'#4ade80']], zmin=-1, zmax=1)
    fig_hm.update_layout(plot_bgcolor=PBG, paper_bgcolor=BG,
        font=dict(family='Sora', color=TXT, size=10),
        height=380, margin=dict(l=4,r=4,t=8,b=4), coloraxis_showscale=True)
    fig_hm.update_traces(textfont=dict(size=9, color='white'))
    st.plotly_chart(fig_hm, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT STUDENT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Predict Student":
    st.markdown('<div class="ptitle">Student Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">Enter student details for an instant AI-powered dropout risk assessment</div>', unsafe_allow_html=True)

    with st.form("pf", clear_on_submit=False):
        st.markdown('<div class="sh" style="margin-top:0">📚 Academic Performance</div>', unsafe_allow_html=True)
        a1,a2,a3,a4 = st.columns(4)
        s1g = a1.slider("1st Sem Grade (0–20)", 0.0, 20.0, 12.5, 0.5)
        s2g = a2.slider("2nd Sem Grade (0–20)", 0.0, 20.0, 11.0, 0.5)
        s1a = a3.slider("Units Approved Sem1",  0, 7, 4)
        s2a = a4.slider("Units Approved Sem2",  0, 7, 3)

        b1,b2,b3 = st.columns(3)
        s1e = b1.slider("Units Enrolled Sem1", 0, 7, 5)
        s2e = b2.slider("Units Enrolled Sem2", 0, 7, 5)
        adm = b3.slider("Admission Grade",     95.0, 190.0, 138.0, 1.0)

        st.markdown('<div class="sh">👤 Personal & Financial</div>', unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        age = c1.number_input("Age", 17, 60, 20)
        att = c2.slider("Attendance (%)", 0, 100, 72)
        sch = c3.selectbox("Scholarship",  [0,1], format_func=lambda x:"Yes" if x else "No")
        dbt = c4.selectbox("Has Debt",     [0,1], format_func=lambda x:"Yes" if x else "No")
        fee = c5.selectbox("Fees Paid",    [1,0], format_func=lambda x:"Yes" if x else "No")

        d1,d2,d3,d4 = st.columns(4)
        gen = d1.selectbox("Gender",        [0,1], format_func=lambda x:"Female" if x==0 else "Male")
        mar = d2.selectbox("Marital Status",[0,1,2], format_func=lambda x:{0:"Single",1:"Married",2:"Other"}[x])
        dis = d3.selectbox("Displaced",     [0,1], format_func=lambda x:"Yes" if x else "No")
        pvg = d4.slider("Prev Qual Grade",  90.0, 190.0, 133.0, 1.0)

        st.markdown('<div class="sh">🌐 Economic Context</div>', unsafe_allow_html=True)
        e1,e2,e3 = st.columns(3)
        ump = e1.slider("Unemployment Rate (%)", 5.0, 20.0, 10.0, 0.5)
        inf = e2.slider("Inflation Rate (%)",   -1.0,  4.0,  1.5, 0.1)
        gdp = e3.slider("GDP Growth (%)",        -5.0,  4.0,  1.0, 0.1)

        submitted = st.form_submit_button("🔮  Run AI Risk Assessment")

    if submitted:
        row = {c: 0.0 for c in feat_cols}
        row.update({
            'Marital status': mar, 'Application mode': 1, 'Application order': 1, 'Course': 9,
            'Daytime/evening attendance': 1, 'Previous qualification': 1,
            'Previous qualification (grade)': pvg, 'Nationality': 1,
            "Mother's qualification": 3, "Father's qualification": 3,
            "Mother's occupation": 5,  "Father's occupation": 5,
            'Admission grade': adm, 'Displaced': dis, 'Educational special needs': 0,
            'Debtor': dbt, 'Tuition fees up to date': fee, 'Gender': gen,
            'Scholarship holder': sch, 'Age at enrollment': age, 'International': 0,
            'Curricular units 1st sem (credited)': 0,
            'Curricular units 1st sem (enrolled)': s1e,
            'Curricular units 1st sem (evaluations)': s1e,
            'Curricular units 1st sem (approved)': s1a,
            'Curricular units 1st sem (grade)': s1g,
            'Curricular units 1st sem (without evaluations)': 0,
            'Curricular units 2nd sem (credited)': 0,
            'Curricular units 2nd sem (enrolled)': s2e,
            'Curricular units 2nd sem (evaluations)': s2e,
            'Curricular units 2nd sem (approved)': s2a,
            'Curricular units 2nd sem (grade)': s2g,
            'Curricular units 2nd sem (without evaluations)': 0,
            'Unemployment rate': ump, 'Inflation rate': inf, 'GDP': gdp,
            'Attendance (%)': att,
        })
        Xi   = np.array([[row[c] for c in feat_cols]])
        Xs   = scaler.transform(Xi)
        pred, proba = safe_predict(model, Xs)
        conf = round(max(proba)*100, 1)

        lbls  = {0:'Dropout Risk', 1:'Enrolled', 2:'Graduate'}
        csscl = {0:'pd', 1:'pe', 2:'pg'}
        icons = {0:'⚠️', 1:'📘', 2:'🎓'}

        rc, pc2 = st.columns([1, 1.4])
        with rc:
            st.markdown(f"""
            <div class="pred-wrap {csscl[pred]}">
                <div style="font-size:3rem;line-height:1">{icons[pred]}</div>
                <h2>{lbls[pred]}</h2>
                <div style="font-size:.88rem;color:#6878a0;margin-top:.4rem">
                    Confidence: <span style="color:#e2e8f8;font-weight:600">{conf}%</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with pc2:
            st.markdown('<div class="sh" style="margin-top:0">Probability Breakdown</div>', unsafe_allow_html=True)
            pn = ['Dropout','Enrolled','Graduate']
            pclr = ['#f87171','#60a5fa','#4ade80']
            fig_p = go.Figure()
            for i, (nm, pv, pc3) in enumerate(zip(pn, proba, pclr)):
                fig_p.add_trace(go.Bar(
                    x=[pv*100], y=[nm], orientation='h',
                    marker=dict(color=pc3, opacity=0.9 if i==pred else 0.35),
                    text=f'{pv*100:.1f}%', textposition='inside',
                    textfont=dict(color='white', size=14), name=nm))
            dark(fig_p, h=190)
            fig_p.update_xaxes(range=[0,105], showticklabels=False, showgrid=False)
            fig_p.update_layout(barmode='stack', showlegend=False, margin=dict(l=5,r=5,t=5,b=5))
            st.plotly_chart(fig_p, use_container_width=True)

        st.markdown("<div class='div'></div>", unsafe_allow_html=True)

        gd = s2g - s1g; f1 = s1e - s1a; f2 = s2e - s2a
        flags = []
        if att < 55:      flags.append(("🔴","Critical", f"Attendance {att}% — below 55% threshold, highest risk predictor"))
        elif att < 65:    flags.append(("🟠","High",     f"Attendance {att}% — moderate risk zone"))
        if gd < -2:       flags.append(("🔴","Critical", f"Grade dropped {abs(gd):.1f} pts Sem1→Sem2 — declining trajectory"))
        elif gd < 0:      flags.append(("🟡","Low",      f"Grade slightly declined {abs(gd):.1f} pts — monitor closely"))
        if dbt == 1:      flags.append(("🟠","Medium",   "Outstanding financial debt present"))
        if fee == 0:      flags.append(("🟠","Medium",   "Tuition fees not up to date"))
        if f2 >= 2:       flags.append(("🟠","Medium",   f"Failed {f2} unit(s) in Semester 2"))
        if sch == 1:      flags.append(("🟢","Positive", "Scholarship holder — financial stability present"))
        if gd >= 0.5:     flags.append(("🟢","Positive", f"Grades improving +{gd:.1f} pts Sem1→Sem2"))
        if att >= 80:     flags.append(("🟢","Positive", f"Strong attendance at {att}%"))

        if flags:
            st.markdown('<div class="sh">📊 Risk Factor Analysis</div>', unsafe_allow_html=True)
            fc1, fc2 = st.columns(2)
            sev_c = {'Critical':'#f87171','High':'#fb923c','Medium':'#fbbf24','Low':'#a3e635','Positive':'#4ade80'}
            for i,(icon,sev,msg) in enumerate(flags):
                sc2 = sev_c.get(sev,'#8892b0')
                [fc1,fc2][i%2].markdown(
                    f"<div class='iv'>{icon} &nbsp;<strong style='color:{sc2}'>{sev}:</strong> {msg}</div>",
                    unsafe_allow_html=True)

        st.markdown('<div class="sh">💡 Recommended Actions</div>', unsafe_allow_html=True)
        if pred == 0:
            acts = [
                "📞 Schedule urgent academic counseling session within 48 hours",
                "💰 Review emergency financial aid or fee waiver eligibility",
                "📋 Assign a dedicated peer mentor from the same course",
                "📊 Conduct immediate mid-semester academic performance review",
                "🏫 Refer to student welfare committee for holistic support",
                "📱 Enable weekly academic advisor check-in notifications",
            ]
        elif pred == 1:
            acts = [
                "📈 Monitor academic progress closely next semester",
                "🎯 Encourage enrollment in skill-building workshops",
                "📚 Recommend supplementary study resources",
                "🤝 Connect with peer study groups and learning circles",
            ]
        else:
            acts = [
                "🎓 Strong graduation trajectory — continue current support",
                "🏆 Consider nomination for merit recognition or scholarship",
                "🚀 Encourage research projects or industry internship applications",
                "🌟 Invite to mentor at-risk junior students",
            ]
        for a in acts:
            st.markdown(f"<div class='iv'>{a}</div>", unsafe_allow_html=True)

        st.markdown('<div class="sh">📋 Snapshot Metrics</div>', unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Sem1 Grade",    f"{s1g:.1f}/20")
        m2.metric("Sem2 Grade",    f"{s2g:.1f}/20",
                  f"{'▼' if gd<0 else '▲'} {abs(gd):.1f}",
                  delta_color="normal" if gd>=0 else "inverse")
        m3.metric("Attendance",    f"{att}%")
        m4.metric("Sem2 Pass Rate",f"{int(s2a/max(s2e,1)*100)}%")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Analytics":
    st.markdown('<div class="ptitle">Model Analytics & Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">5 trained ML models compared — accuracy, F1, precision, recall, confusion matrices</div>', unsafe_allow_html=True)

    mr   = results['model_results']
    best = results['best_model_name']
    bc2  = {'Logistic Regression':'#60a5fa','KNN':'#4ade80','Decision Tree':'#fbbf24',
            'Random Forest':'#f97316','Gradient Boosting':'#c084fc'}

    st.markdown('<div class="sh">Model Scorecards</div>', unsafe_allow_html=True)
    mcols = st.columns(5)
    for i, name in enumerate(mr.keys()):
        r  = mr[name]; ib = (name == best)
        bd = 'border:2px solid #5c6bc0;' if ib else ''
        col = bc2.get(name,'#8892b0')
        mcols[i].markdown(f"""
        <div class="card" style="{bd}">
            <div style="font-size:.72rem;font-weight:700;color:{col};text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">
                {name}{'  🏆' if ib else ''}
            </div>
            <div style="font-size:2rem;font-weight:800;color:#e2e8f8;">{r['accuracy']}%</div>
            <div style="font-size:.7rem;color:#4a5a7a;margin-bottom:10px;">Accuracy</div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;">
                <span class="badge b-blue" style="font-size:.68rem;padding:2px 8px;">F1 {r['f1']}%</span>
                <span class="badge b-green" style="font-size:.68rem;padding:2px 8px;">P {r['precision']}%</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="sh">All Metrics Comparison</div>', unsafe_allow_html=True)
        names = list(mr.keys()); mets = ['accuracy','f1','precision','recall']
        mlabs = ['Accuracy','F1','Precision','Recall']
        fig_gb = go.Figure()
        for j,(mk,ml) in enumerate(zip(mets,mlabs)):
            fig_gb.add_trace(go.Bar(name=ml, x=names, y=[mr[n][mk] for n in names],
                marker_color=COLS[j],
                text=[f"{mr[n][mk]}%" for n in names], textposition='outside',
                textfont=dict(size=9)))
        dark(fig_gb, h=380)
        fig_gb.update_layout(barmode='group', yaxis_range=[40,108])
        fig_gb.update_xaxes(tickangle=-15)
        st.plotly_chart(fig_gb, use_container_width=True)

    with cb:
        st.markdown('<div class="sh">Radar Performance Map</div>', unsafe_allow_html=True)
        cats = ['Accuracy','F1 Score','Precision','Recall']
        fig_r = go.Figure()
        for i, name in enumerate(mr.keys()):
            r = mr[name]; vals = [r['accuracy'],r['f1'],r['precision'],r['recall']]
            fig_r.add_trace(go.Scatterpolar(
                r=vals+[vals[0]], theta=cats+[cats[0]], fill='toself', name=name,
                line=dict(color=COLS[i], width=2), fillcolor=COLS[i], opacity=0.12))
        dark(fig_r, h=380)
        fig_r.update_layout(polar=dict(bgcolor=PBG,
            radialaxis=dict(visible=True, range=[40,100], gridcolor=GRD, color=TXT, tickfont=dict(size=9)),
            angularaxis=dict(gridcolor=GRD, color=TXT)))
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown('<div class="sh">Top 15 Feature Importances (Random Forest)</div>', unsafe_allow_html=True)
    short = {
        'Curricular units 2nd sem (grade)':'Sem2 Grade',
        'Curricular units 1st sem (grade)':'Sem1 Grade',
        'Curricular units 2nd sem (approved)':'Sem2 Approved',
        'Curricular units 1st sem (approved)':'Sem1 Approved',
        'Attendance (%)':'Attendance','Admission grade':'Admission Grade',
        'Age at enrollment':'Age at Enrollment','Tuition fees up to date':'Fees Up-to-Date',
        'Scholarship holder':'Scholarship','Debtor':'Debtor Status',
        'Previous qualification (grade)':'Prev Qual Grade','Unemployment rate':'Unemployment Rate',
        'Curricular units 2nd sem (enrolled)':'Sem2 Enrolled',
        'Curricular units 1st sem (enrolled)':'Sem1 Enrolled','GDP':'GDP Growth',
    }
    fi    = results['feature_importance']
    fi_df = pd.DataFrame(list(fi.items()), columns=['Feature','Score'])
    fi_df = fi_df.sort_values('Score', ascending=True).tail(15)
    fi_df['Name'] = fi_df['Feature'].map(lambda x: short.get(x, x[:26]))
    fig_fi = go.Figure(go.Bar(x=fi_df['Score'], y=fi_df['Name'], orientation='h',
        marker=dict(color=fi_df['Score'],
                    colorscale=[[0,'#1e2a60'],[0.5,'#3d50c3'],[1,'#60a5fa']],showscale=False),
        text=fi_df['Score'].round(4), textposition='outside', textfont=dict(size=11, color='#a8b4d0')))
    dark(fig_fi, h=440)
    fig_fi.update_layout(margin=dict(l=8,r=50,t=10,b=8))
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="sh">Confusion Matrices — All Models</div>', unsafe_allow_html=True)
    cmcols = st.columns(len(mr))
    for i, name in enumerate(mr.keys()):
        cm = np.array(mr[name]['confusion_matrix'])
        fig_cm = px.imshow(cm, text_auto=True,
            x=['D','E','G'], y=['D','E','G'], aspect='auto',
            color_continuous_scale=[[0,PBG],[0.4,'#2d3a8c'],[1,'#60a5fa']])
        fig_cm.update_layout(plot_bgcolor=PBG, paper_bgcolor=BG,
            font=dict(family='Sora',color=TXT,size=10), height=220,
            margin=dict(l=4,r=4,t=28,b=4),
            title=dict(text=name[:16],font=dict(size=11,color='#cdd5f0'),x=0),
            coloraxis_showscale=False,
            xaxis=dict(color=TXT,title='Pred'),yaxis=dict(color=TXT,title='Actual'))
        fig_cm.update_traces(textfont=dict(size=12, color='white'))
        cmcols[i].plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<div class="sh">ANN Deep Learning Architecture</div>', unsafe_allow_html=True)
    st.code("""
# ANN Multi-class Classifier — 3 output classes: Dropout / Enrolled / Graduate
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model = Sequential([
    Dense(256, activation='relu', input_shape=(37,)),
    BatchNormalization(),
    Dropout(0.35),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.30),
    Dense(64, activation='relu'),
    Dropout(0.25),
    Dense(3, activation='softmax')   # 3-class output
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(X_train, y_train, validation_split=0.2,
                    epochs=150, batch_size=32, callbacks=[
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
])
    """, language='python')

    st.markdown('<div class="sh">Final Model Performance Summary</div>', unsafe_allow_html=True)
    mr_data = [{'Model':f"{'🏆 ' if n==best else ''}{n}",
                'Accuracy (%)':r['accuracy'],'F1 Score (%)':r['f1'],
                'Precision (%)':r['precision'],'Recall (%)':r['recall']}
               for n,r in mr.items()]
    mdf = pd.DataFrame(mr_data).set_index('Model')
    st.dataframe(mdf.style.highlight_max(axis=0, color='#1e3a6e').format("{:.2f}"),
                 use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CLUSTER ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "👥  Cluster Analysis":
    st.markdown('<div class="ptitle">Student Cluster Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">K-Means (k=3) segments students into Low / Medium / High risk profiles for targeted intervention</div>', unsafe_allow_html=True)

    bcl = {'High Risk':'b-red','Medium Risk':'b-yellow','Low Risk':'b-green'}
    icl = {'High Risk':'🔴','Medium Risk':'🟡','Low Risk':'🟢'}
    bdr = {'High Risk':'#5c1a1a','Medium Risk':'#7a4f0d','Low Risk':'#14532d'}

    st.markdown('<div class="sh">Risk Cluster Profiles</div>', unsafe_allow_html=True)
    cc = st.columns(3)
    for i, row in clusters.iterrows():
        rl  = rmap.get(str(i), f'Cluster {i}')
        bd  = bdr.get(rl,'#1e2a42'); ic = icl.get(rl,'⚪'); bc3 = bcl.get(rl,'b-blue')
        cnt = int(row.get('Count', 0))
        s1  = float(row.get('Avg_Sem1_Grade', 0))
        s2  = float(row.get('Avg_Sem2_Grade', 0))
        at  = float(row.get('Avg_Attendance', 0))
        dr  = float(row.get('Dropout_Rate', 0))
        cc[i%3].markdown(f"""
        <div class="card" style="border-color:{bd}">
            <span class="badge {bc3}">{ic} {rl}</span>
            <div style="font-size:1.1rem;font-weight:700;color:#cdd5f0;margin:10px 0 6px;">
                Cluster {i} &nbsp;·&nbsp;
                <span style="font-size:.85rem;color:#6878a0;font-weight:400">{cnt} students</span>
            </div>
            <hr style="border-color:#1c2035;margin:8px 0">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:.83rem;">
                <div><div style="color:#4a5a7a;font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;">Avg Sem1 Grade</div>
                    <div style="color:#e2e8f8;font-weight:600;font-size:1.05rem;">{s1:.1f}</div></div>
                <div><div style="color:#4a5a7a;font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;">Avg Sem2 Grade</div>
                    <div style="color:#e2e8f8;font-weight:600;font-size:1.05rem;">{s2:.1f}</div></div>
                <div><div style="color:#4a5a7a;font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;">Avg Attendance</div>
                    <div style="color:#e2e8f8;font-weight:600;font-size:1.05rem;">{at:.1f}%</div></div>
                <div><div style="color:#4a5a7a;font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;">Dropout Rate</div>
                    <div style="color:#f87171;font-weight:700;font-size:1.05rem;">{dr:.1f}%</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sh">PCA 2D Cluster Visualization</div>', unsafe_allow_html=True)
    pca_df['Risk_Label'] = [rmap.get(str(int(c)),'Unknown') for c in pca_df['Cluster']]
    cdisc = {'High Risk':'#f87171','Medium Risk':'#fbbf24','Low Risk':'#4ade80','Unknown':'#8892b0'}
    samp2 = pca_df.sample(min(900,len(pca_df)), random_state=7)
    fig_pca = px.scatter(samp2, x='PC1', y='PC2', color='Risk_Label',
        color_discrete_map=cdisc, opacity=0.6,
        hover_data={'Target_Label':True,'PC1':':.2f','PC2':':.2f','Risk_Label':False},
        labels={'Risk_Label':'Risk Group'}, symbol='Risk_Label')
    fig_pca.update_traces(marker=dict(size=5))
    dark(fig_pca, "Student Risk Groups in Principal Component Space", h=420)
    fig_pca.update_layout(legend=dict(orientation='h', y=1.08, x=0))
    st.plotly_chart(fig_pca, use_container_width=True)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="sh">Grade Comparison Across Clusters</div>', unsafe_allow_html=True)
        fig_cg = go.Figure()
        for i, row in clusters.iterrows():
            rl = rmap.get(str(i), f'Cluster {i}')
            s1 = float(row.get('Avg_Sem1_Grade',0))
            s2 = float(row.get('Avg_Sem2_Grade',0))
            fig_cg.add_trace(go.Bar(name=rl, x=['Semester 1','Semester 2'], y=[s1,s2],
                marker_color=COLS[i%3], text=[f"{s1:.1f}",f"{s2:.1f}"], textposition='outside'))
        dark(fig_cg, h=320)
        fig_cg.update_layout(barmode='group', yaxis_range=[10,16])
        st.plotly_chart(fig_cg, use_container_width=True)

    with g2:
        st.markdown('<div class="sh">Dropout Rate by Cluster</div>', unsafe_allow_html=True)
        rll = [rmap.get(str(i),f'Cluster {i}') for i in clusters.index]
        dr_vals = [float(clusters.loc[i,'Dropout_Rate']) for i in clusters.index]
        dcl = {'High Risk':'#f87171','Medium Risk':'#fbbf24','Low Risk':'#4ade80'}
        fig_dr = go.Figure(go.Bar(x=rll, y=dr_vals,
            marker_color=[dcl.get(r,'#8892b0') for r in rll],
            text=[f'{v:.1f}%' for v in dr_vals],
            textposition='outside', textfont=dict(size=13)))
        dark(fig_dr, h=320)
        fig_dr.update_yaxes(title_text='Dropout Rate (%)', range=[0,55])
        st.plotly_chart(fig_dr, use_container_width=True)

    st.markdown('<div class="sh">Outcome Breakdown Within Each Cluster</div>', unsafe_allow_html=True)
    df['Risk_Group'] = [rmap.get(str(int(c)),'Unknown') for c in df['Cluster']]
    df['Label']      = df['Target'].map({0:'Dropout',1:'Enrolled',2:'Graduate'})
    oc = df.groupby(['Risk_Group','Label']).size().reset_index(name='Count')
    fig_ob = px.bar(oc, x='Risk_Group', y='Count', color='Label',
        color_discrete_map={'Dropout':'#f87171','Enrolled':'#60a5fa','Graduate':'#4ade80'},
        barmode='group', text='Count',
        category_orders={'Risk_Group':['High Risk','Medium Risk','Low Risk']})
    dark(fig_ob, h=340)
    fig_ob.update_traces(textposition='outside', textfont_size=11)
    fig_ob.update_layout(legend=dict(orientation='h', y=1.06, x=0))
    st.plotly_chart(fig_ob, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📋  About":
    st.markdown('<div class="ptitle">About This Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">AI-Powered Student Dropout Prediction & Risk Intelligence System</div>', unsafe_allow_html=True)

    ac, sc2 = st.columns([1.4, 1])
    with ac:
        st.markdown("""
        <div class="card">
            <div class="sh" style="margin-top:0">🎯 Abstract</div>
            <p style="color:#a8b4d0;font-size:.88rem;line-height:1.8;margin:0">
            This project develops an <strong style="color:#e2e8f8">AI-powered student dropout prediction system</strong>
            that analyzes academic, financial, and demographic data to identify at-risk students before
            they leave. The system employs a complete ML pipeline — data preprocessing, exploratory analysis,
            multi-model classification, K-Means clustering, and deep learning — to generate actionable
            early-warning signals and intervention recommendations for academic institutions.
            </p>
        </div>
        <div class="card">
            <div class="sh" style="margin-top:0">❓ Problem Statement</div>
            <p style="color:#a8b4d0;font-size:.88rem;line-height:1.8;margin:0">
            Student dropout is a critical problem in higher education. Traditional methods rely on
            reactive intervention after the student has already quit. This system builds an
            <strong style="color:#e2e8f8">intelligent early-warning pipeline</strong> that predicts
            dropout risk <em>before</em> it happens, enabling proactive and targeted institutional support.
            </p>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sh">🔄 ML Pipeline Architecture</div>', unsafe_allow_html=True)
        steps = [
            ("1","Data Collection",     "2,000 records · 37 features · academic, financial & demographic"),
            ("2","Preprocessing",       "Null handling · StandardScaler · Label encoding · 80/20 stratified split"),
            ("3","EDA",                 "Correlation heatmaps · Grade trajectories · Risk factor charts · Violin plots"),
            ("4","Feature Engineering", "Grade trend delta · Fail rate · Attendance band binning"),
            ("5","Model Training",      "5 models: Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting"),
            ("6","Clustering",          "K-Means k=3 with PCA 2D — Low / Medium / High risk segments"),
            ("7","Deep Learning",       "ANN: 256→128→64→3 softmax · BatchNorm + Dropout · EarlyStopping"),
            ("8","Deployment",          "Streamlit app · Live prediction · Dashboard · Cluster explorer"),
        ]
        for num,name,desc in steps:
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin-bottom:8px;align-items:flex-start">
                <div style="min-width:28px;height:28px;background:#1e2a60;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.75rem;font-weight:700;color:#60a5fa;margin-top:1px">{num}</div>
                <div><span style="font-weight:600;color:#cdd5f0;font-size:.88rem">{name}</span>
                <span style="color:#4a5a7a;font-size:.82rem"> — {desc}</span></div>
            </div>""", unsafe_allow_html=True)

    with sc2:
        st.markdown("""<div class="card">
            <div class="sh" style="margin-top:0">🛠️ Technologies Used</div>""", unsafe_allow_html=True)
        for icon,name,role in [
            ("🐍","Python 3.11","Core language"),("🐼","Pandas & NumPy","Data manipulation"),
            ("🤖","Scikit-learn","ML models"),("🧠","TensorFlow/Keras","Deep learning"),
            ("📊","Plotly","Interactive charts"),("🚀","Streamlit","Web app deployment"),
            ("💾","Joblib","Model serialization"),("📓","Jupyter Notebook","EDA & experiments")]:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid #1c2035;font-size:.83rem">
                <span>{icon} <strong style="color:#cdd5f0">{name}</strong></span>
                <span style="color:#4a5a7a">{role}</span></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div class="card" style="margin-top:.9rem">
            <div class="sh" style="margin-top:0">🧪 Models Implemented</div>""", unsafe_allow_html=True)
        for icon,name,desc in [
            ("📈","Logistic Regression","Baseline classifier"),
            ("📍","KNN","Distance-based, k=7"),
            ("🌳","Decision Tree","depth=10"),
            ("🌲","Random Forest 🏆","200 trees — best model"),
            ("⚡","Gradient Boosting","150 estimators"),
            ("🧬","ANN (Deep Learning)","256→128→64→3 softmax"),
            ("🔵","K-Means Clustering","k=3 risk segmentation"),
            ("📉","PCA","2D cluster visualization")]:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #1c2035;font-size:.82rem">
                <span>{icon} <strong style="color:#cdd5f0">{name}</strong></span>
                <span style="color:#4a5a7a">{desc}</span></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card" style="margin-top:.9rem;border-color:#1e2a60">
            <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:#3d4fd6;font-weight:700;margin-bottom:10px">About the Developer</div>
            <div style="font-size:.85rem;color:#a8b4d0;line-height:2.1">
                👤 <strong style="color:#e2e8f8">Dhiksha C G</strong><br>
                📍 Bengaluru, Karnataka, India<br>
                💼 Aspiring Data Analyst<br>
                🔗 <a href="https://linkedin.com/in/dhiksha-c-g-43b579285" target="_blank" style="color:#60a5fa;text-decoration:none;">LinkedIn</a>
                &nbsp;·&nbsp;
                <a href="https://github.com/0403darknight" target="_blank" style="color:#60a5fa;text-decoration:none;">GitHub</a>
            </div>
            <hr style="border-color:#1c2035;margin:10px 0">
            <div style="font-size:.8rem;color:#6878a0;line-height:1.9">
                🐍 Python · Pandas · NumPy · Scikit-learn<br>
                📊 Power BI · Advanced Excel · SQL<br>
                🤖 Machine Learning · Deep Learning<br>
                📈 Dashboard Development · KPI Tracking<br>
                🔬 Published researcher — ICRCCT 2025
            </div>
            <hr style="border-color:#1c2035;margin:10px 0">
            <div style="font-size:.78rem;color:#4a5a7a;line-height:1.8;font-style:italic">
                "Data-focused analyst with hands-on experience in SQL, Excel, Power BI, and Python for analysing large datasets and generating actionable insights."
            </div>
        </div>""", unsafe_allow_html=True)
