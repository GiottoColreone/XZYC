import streamlit as st
import pandas as pd
import numpy as np
import jieba
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import urllib.request
import os
import io
import time

# ==========================================
# 0. 基础环境配置
# ==========================================
st.set_page_config(page_title="无证户智能稽查天眼 V4 - 权重平衡满血版", page_icon="👁️", layout="wide")

@st.cache_resource
def get_chinese_font():
    font_path = "SimHei.ttf"
    if not os.path.exists(font_path):
        try:
            urllib.request.urlretrieve("https://raw.githubusercontent.com/StellarCN/scp_zh/master/fonts/SimHei.ttf", font_path)
        except Exception: pass
    
    if os.path.exists(font_path):
        title_font = fm.FontProperties(fname=font_path, size=11, weight='bold')
        label_font = fm.FontProperties(fname=font_path, size=9)
    else:
        title_font = fm.FontProperties(size=11, weight='bold')
        label_font = fm.FontProperties(size=9)
    plt.rcParams['axes.unicode_minus'] = False
    return title_font, label_font

title_font, label_font = get_chinese_font()

# ==========================================
# 1. NLP 预处理模块
# ==========================================
CUSTOM_STOP_WORDS = {'有限','责任','分公司','集团','控股','股份','有限公司','徐州','地址','未知','公司', '店铺'}
TOBACCO_WORDS = {'烟草制品零售','卷烟零售','雪茄零售','烟丝零售','香烟销售','烟草销售','烟草','卷烟','雪茄','烟丝','香烟'}

def custom_tokenizer(text):
    if not isinstance(text, str) or not text: return []
    norm_map = {'百货店':'百货','百货商场':'百货','百货公司':'百货','百货超市':'百货','便利店':'便利','批发部':'批发'}
    words = jieba.lcut(text)
    processed_words = [norm_map.get(w, w) for w in words if len(w) > 1 and w not in CUSTOM_STOP_WORDS and not any(tob_w in w for tob_w in TOBACCO_WORDS)]
    return processed_words

# ==========================================
# 2. 可视化模块 (满血恢复 12 张图表)
# ==========================================
def draw_analysis_charts(df, t_font, l_font):
    st.markdown("### 📊 AI 模型全盘数据可视化分析")
    color_map = {'低风险': '#32CD32', '中风险': '#FFD700', '高风险': '#FF6B00', '极高风险': '#FF0000'}
    level_order = ['低风险', '中风险', '高风险', '极高风险']
    
    # 第一板块：概率分布分析
    st.markdown("#### 一、 无证户概率综合分析")
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8))
    
    # 1.1 直方图
    for level in level_order:
        subset = df[df['风险等级'] == level]
        if not subset.empty:
            axes1[0, 0].hist(subset['无证户综合概率(%)'], bins=15, color=color_map[level], alpha=0.7, label=level, edgecolor='black')
    axes1[0, 0].set_title('所有商户无证户概率分布', fontproperties=t_font)
    axes1[0, 0].legend(prop=l_font)

    # 1.2 风险等级饼图
    risk_counts = df['风险等级'].value_counts().reindex(level_order).fillna(0)
    axes1[0, 1].pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=[color_map[l] for l in risk_counts.index], startangle=90, textprops={'fontproperties': l_font})
    axes1[0, 1].set_title('所有商户风险等级分布', fontproperties=t_font)

    # 1.3 信用值密度
    sns.kdeplot(data=df, x='信用值', hue='风险等级', palette=color_map, ax=axes1[0, 2], fill=True, common_norm=False)
    axes1[0, 2].set_title('信用值密度分布', fontproperties=t_font)
    legend = axes1[0, 2].get_legend()
    if legend: plt.setp(legend.texts, fontproperties=l_font)

    # 1.4 平均概率
    avg_prob = df.groupby('风险等级')['无证户综合概率(%)'].mean().reindex(level_order)
    axes1[1, 0].bar(avg_prob.index, avg_prob.values, color=[color_map[l] for l in avg_prob.index])
    axes1[1, 0].set_title('各等级平均概率', fontproperties=t_font)
    axes1[1, 0].set_xticklabels(avg_prob.index, fontproperties=l_font)

    # 1.5 散点图
    for level in level_order:
        subset = df[df['风险等级'] == level]
        if not subset.empty:
            axes1[1, 1].scatter(subset['信用值'], subset['无证户综合概率(%)'], color=color_map[level], label=level, alpha=0.6, s=15)
    axes1[1, 1].set_title('信用值 vs 概率散点', fontproperties=t_font)
    axes1[1, 1].legend(prop=l_font)

    # 1.6 法人比例
    high_risk_reps = df[df['该商户负责人是否在无证户名录（可能重名）'] == '是（可能重名）'].shape[0]
    axes1[1, 2].pie([high_risk_reps, df.shape[0] - high_risk_reps], labels=['历史无证重名', '普通法人'], autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'], startangle=140, textprops={'fontproperties': l_font})
    axes1[1, 2].set_title('法人身份重名比例', fontproperties=t_font)
    
    fig1.tight_layout(pad=3.0) 
    st.pyplot(fig1)

    # 第二板块：详细统计分析
    st.markdown("#### 二、 风险等级详细分析")
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    
    # 2.1 数量统计
    bars = axes2[0, 0].bar(risk_counts.index, risk_counts.values, color=[color_map[l] for l in risk_counts.index])
    axes2[0, 0].set_title('商户数量分布', fontproperties=t_font)
    axes2[0, 0].set_xticklabels(risk_counts.index, fontproperties=l_font)

    # 2.2 概率箱线图
    box_data = [df[df['风险等级'] == level]['无证户综合概率(%)'].dropna() for level in level_order]
    axes2[0, 1].boxplot(box_data, labels=level_order, patch_artist=True)
    axes2[0, 1].set_title('各等级概率箱线', fontproperties=t_font)
    axes2[0, 1].set_xticklabels(level_order, fontproperties=l_font)

    # 2.3 信用箱线图
    box_data_score = [df[df['风险等级'] == level]['信用值'].dropna() for level in level_order]
    axes2[0, 2].boxplot(box_data_score, labels=level_order, patch_artist=True)
    axes2[0, 2].set_title('信用值分布箱线', fontproperties=t_font)
    axes2[0, 2].set_xticklabels(level_order, fontproperties=l_font)

    # 2.4 累积分布
    sorted_probs = np.sort(df['无证户综合概率(%)'])
    cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs) * 100
    axes2[1, 0].plot(sorted_probs, cumulative, color='#1E90FF', linewidth=2)
    axes2[1, 0].set_title('概率累积分布曲线', fontproperties=t_font)

    # 2.5 平均信用分
    avg_score = df.groupby('风险等级')['信用值'].mean().reindex(level_order)
    axes2[1, 1].bar(avg_score.index, avg_score.values, color=[color_map[l] for l in avg_score.index])
    axes2[1, 1].set_title('平均信用分', fontproperties=t_font)
    axes2[1, 1].set_xticklabels(avg_score.index, fontproperties=l_font)

    # 2.6 极高风险快照
    axes2[1, 2].axis('off')
    axes2[1, 2].set_title('极高风险目标快照', fontproperties=t_font)
    y_pos = 0.9
    for idx, row in df.head(8).reset_index().iterrows():
        name = str(row['公司名称'])[:10] + ".." if len(str(row['公司名称'])) > 10 else row['公司名称']
        axes2[1, 2].text(0.0, y_pos, f"{idx+1}. {name} ({row['无证户综合概率(%)']}%)", fontproperties=l_font, color='red' if idx < 3 else 'black')
        y_pos -= 0.12

    fig2.tight_layout(pad=3.0)
    st.pyplot(fig2)

# ==========================================
# 3. 主程序逻辑
# ==========================================
st.title("👁️ 卷烟无证经营户动态筛查 AI 模型 (三权平衡版)")

with st.sidebar:
    st.header("📂 1. 数据接入库")
    file_biz = st.file_uploader("上传【营业执照】全量名单", type=["xlsx", "csv"])
    file_unl = st.file_uploader("上传【历史无证户】名单", type=["xlsx", "csv"])
    start_btn = st.button("🚀 2. 启动 AI 深度筛查演算", type="primary", use_container_width=True)

if start_btn:
    if not file_biz or not file_unl:
        st.warning("⚠️ 权限阻断：请先在左侧上传两个必须的数据文件！")
    else:
        st.markdown("### 💻 系统核心演算终端")
        log_container = st.container(height=250)
        terminal = log_container.empty()
        log_lines = []
        
        def log_to_terminal(message, delay=0.1):
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]
            log_lines.insert(0, f"[{timestamp}] {message}")
            display_text = "▼ 实时终端日志 [倒序输出]\n" + "="*65 + "\n" + "\n".join(log_lines)
            terminal.code(display_text, language="bash")
            time.sleep(delay)

        start_time = time.time()

        # --- 数据加载 ---
        log_to_terminal("[SYSTEM] 引擎初始化，正在加载数据源...")
        biz = pd.read_excel(file_biz) if file_biz.name.endswith('.xlsx') else pd.read_csv(file_biz)
        unl = pd.read_excel(file_unl) if file_unl.name.endswith('.xlsx') else pd.read_csv(file_unl)
        
        fill_dict = {'公司名称':'未知', '法定代表人':'未知', '经营范围':'未知', '天眼评分':0, '统一社会信用代码':'未知'}
        biz = biz.rename(columns={'天眼评分': '信用值'}).fillna(fill_dict)
        unl = unl.rename(columns={'天眼评分': '信用值'}).fillna(fill_dict)
        biz['信用值'] = pd.to_numeric(biz['信用值'], errors='coerce').fillna(0)
        unl['信用值'] = pd.to_numeric(unl['信用值'], errors='coerce').fillna(0)

        # 法人对比 (不入模型)
        bad_reps = set(unl[~unl['法定代表人'].isin(['未知', '', '无'])]['法定代表人'].unique())
        biz['该商户负责人是否在无证户名录（可能重名）'] = biz['法定代表人'].apply(lambda x: '是（可能重名）' if x in bad_reps else '否')
        
        biz['label'], unl['label'] = 0, 1
        df_all = pd.concat([unl, biz], ignore_index=True)

        # --- 特征工程 (三权平衡逻辑) ---
        log_to_terminal("[NLP] 启动三维平衡计算模型 (权重分配：名称 33.3%, 范围 33.3%, 信用 33.3%)...")
        
        # 1. 公司名称模型
        vec_name = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=500)
        X_name = vec_name.fit_transform(df_all['公司名称'])
        model_name = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42).fit(X_name, df_all['label'])
        prob_name = model_name.predict_proba(X_name)[:, 1]

        # 2. 经营范围模型
        vec_scope = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=500)
        X_scope = vec_scope.fit_transform(df_all['经营范围'])
        model_scope = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42).fit(X_scope, df_all['label'])
        prob_scope = model_scope.predict_proba(X_scope)[:, 1]

        # 3. 信用偏离模型 (进行归一化并反转：低分高风险)
        scaler = MinMaxScaler()
        score_norm = scaler.fit_transform(df_all[['信用值']])
        prob_credit = 1 - score_norm.flatten()

        # 核心：等权融合
        combined_prob = (prob_name * 0.333) + (prob_scope * 0.334) + (prob_credit * 0.333)
        
        df_all['无证户综合概率(%)'] = np.round(combined_prob * 100, 2)
        target_pool = df_all[df_all['label'] == 0].copy()

        # 判定依据生成
        log_to_terminal("[EXPLAINER] 执行加权归因分析，锁定每个因素的贡献度...")
        explanations = []
        for i in range(len(target_pool)):
            p_n = prob_name[target_pool.index[i]] * 33.3
            p_s = prob_scope[target_pool.index[i]] * 33.4
            p_c = prob_credit[target_pool.index[i]] * 33.3
            explanations.append(f"名称特征({p_n:.1f}%) + 范围特征({p_s:.1f}%) + 信用偏离({p_c:.1f}%)")
        target_pool['AI 判定依据'] = explanations

        def assign_risk(p):
            if p >= 80: return '极高风险', '🚨 立即排查'
            elif p >= 60: return '高风险', '⚠️ 重点监控'
            elif p >= 35: return '中风险', '👀 定期关注'
            return '低风险', '✅ 常规监管'
            
        target_pool[['风险等级', '监管建议']] = target_pool.apply(lambda r: pd.Series(assign_risk(r['无证户综合概率(%)'])), axis=1)
        target_pool = target_pool.sort_values('无证户综合概率(%)', ascending=False)
        
        log_to_terminal(f"[SYSTEM] 演算完成。核查规模: {len(target_pool)} 条。")

        # --- 结果展示 ---
        st.success("🎯 稽查演算收官！各维度权重已配平。")
        m1, m2, m3, m4 = st.columns(4)
        total = len(target_pool)
        m1.metric("极高风险 (80%-100%)", f"{len(target_pool[target_pool['风险等级']=='极高风险'])} 家", f"占 {len(target_pool[target_pool['风险等级']=='极高风险'])/total*100:.1f}%")
        m2.metric("高风险 (60%-79%)", f"{len(target_pool[target_pool['风险等级']=='高风险'])} 家", f"占 {len(target_pool[target_pool['风险等级']=='高风险'])/total*100:.1f}%")
        m3.metric("中风险 (35%-59%)", f"{len(target_pool[target_pool['风险等级']=='中风险'])} 家", f"占 {len(target_pool[target_pool['风险等级']=='中风险'])/total*100:.1f}%")
        m4.metric("核查总规模", f"{total} 条", "权重配平: 1:1:1")

        st.divider()

        # --- 风险解释示例 (公式化) ---
        with st.expander("💡 了解 AI 白盒解释器如何计算风险？(权重各为1/3)", expanded=True):
            col_ex1, col_ex2 = st.columns([1, 2])
            with col_ex1:
                st.markdown("""
                **示例商户：** `沛县龙城街道某百货超市`  
                **统一社会信用代码：** `92320322MA******11`  
                **信用分：** `42分`  
                **最终概率：** <span style='color:red; font-weight:bold; font-size:20px;'>76.4%</span>
                """, unsafe_allow_html=True)
            with col_ex2:
                st.info("""
                **各因素量化贡献拆解 (权重均为 33.3%)：**
                * **1. 公司名称贡献度 (28.2/33.3)**：AI 提取关键词 TF-IDF 权重，通过公式计算出名称维度的原始风险为 84.7%，折算后贡献度为 28.2%。
                * **2. 经营范围贡献度 (22.5/33.3)**：经营范围分词在历史无证节点落入高风险概率为 67.5%，折算后贡献度为 22.5%。
                * **3. 信用偏离度 (25.7/33.3)**：商户信用 42 分，通过公式计算得出归一化偏离度为 0.77，折算后贡献度为 25.7%。
                * **综合判定公式**：$28.2 + 22.5 + 25.7 = 76.4$。
                """)
            
            st.markdown("---")
            st.markdown(r"""
            #### 📚 核心专业名词解释与底层计算公式
            * **TF-IDF 词频权重**：评估一个词对一段文本的重要程度。词语在当前店铺信息中出现频率越高（TF），但在全部数据库所有店铺中出现的频率越低（IDF），它的权重就越高。
              * **公式核心**：$$TF\text{-}IDF = \frac{f_{t,d}}{\sum f} \times \log\left(\frac{N}{|\{d \in D : t \in d\}|}\right)$$
            * **基尼不纯度（分类区分度）**：衡量特征分类“干净程度”的指标。数值越低说明这个词能越明确地区分“无证户”和“合规户”。
              * **公式核心**：$$Gini = 1 - \sum_{i=1}^{C} p_i^2$$
            * **决策树与高风险叶子节点**：算法模拟的网状判断路径。当一条数据经过层层条件筛选落入末端（叶子节点）时，系统统计该群体的高危比例。
              * **公式核心**：$$P(Risk) = \frac{N_{high\_risk}}{N_{total\_in\_leaf}}$$
            * **MinMaxScaler 归一化**：将跨度极大的原始数值（如 0-100分的信用分）等比例压缩至统一的 0 到 1 的范围内。这避免了绝对数值过大的指标在计算时“霸凌”掩盖了文字特征的权重。
              * **公式核心**：$$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
            * **特征分布**：指某项数据在整个商户群体中的聚集规律区间。一旦某商户的数值偏离了正常聚集的中心点，就会被算法迅速捕捉为异常离群点。
              * **偏离度参考（Z-score）**：$$Z = \frac{X - \mu}{\sigma}$$
            """)

        # --- 打击名单 ---
        st.subheader("🚨 打击名单 TOP 20（风险度从高到低排序）")
        display_cols = ['公司名称', '统一社会信用代码', '无证户综合概率(%)', 'AI 判定依据', '风险等级', '监管建议', '法定代表人', '注册地址', '该商户负责人是否在无证户名录（可能重名）']
        st.dataframe(
            target_pool[display_cols].head(20).style.format({"无证户综合概率(%)": "{:.2f}%"})
            .map(lambda x: 'color: red; font-weight: bold' if x == '极高风险' else '', subset=['风险等级']),
            use_container_width=True
        )

        # 导出
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            target_pool[display_cols].to_excel(writer, index=False)
        st.download_button("📥 导出全量名单", buffer, "权重平衡版风险名单.xlsx", "application/vnd.ms-excel")

        st.divider()
        draw_analysis_charts(target_pool, title_font, label_font)
