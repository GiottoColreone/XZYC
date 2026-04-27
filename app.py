import streamlit as st
import pandas as pd
import numpy as np
import jieba
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import urllib.request
import os
import io
import time

# ==========================================
# 0. 网页基本设置与【字体乱码强力修复】
# ==========================================
st.set_page_config(page_title="无证户智能稽查天天", page_icon="👁️", layout="wide")

@st.cache_resource
def setup_chinese_fonts():
    # 强制在云端下载并挂载黑体
    font_path = "SimHei.ttf"
    if not os.path.exists(font_path):
        try:
            urllib.request.urlretrieve("https://raw.githubusercontent.com/StellarCN/scp_zh/master/fonts/SimHei.ttf", font_path)
        except Exception:
            pass
    if os.path.exists(font_path):
        # 提取真实字体属性，用于逐个元素注入
        f_prop = fm.FontProperties(fname=font_path)
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = [f_prop.get_name(), 'SimHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return f_prop
    return None

font_prop = setup_chinese_fonts()

CUSTOM_STOP_WORDS = {'有限','责任','分公司','集团','控股','股份','有限公司','徐州','地址','未知','公司', '店铺'}
TOBACCO_WORDS = {'烟草制品零售','卷烟零售','雪茄零售','烟丝零售','香烟销售','烟草销售','烟草','卷烟','雪茄','烟丝','香烟'}

def custom_tokenizer(text):
    if not isinstance(text, str) or not text: return []
    norm_map = {'百货店':'百货','百货商场':'百货','百货公司':'百货','百货超市':'百货','便利店':'便利','批发部':'批发'}
    words = jieba.lcut(text)
    processed_words = [norm_map.get(w, w) for w in words if len(w) > 1 and w not in CUSTOM_STOP_WORDS and not any(tob_w in w for tob_w in TOBACCO_WORDS)]
    return processed_words

# ==========================================
# 1. 图表生成函数 (解决重叠与乱码)
# ==========================================
def draw_analysis_charts(df):
    st.markdown("### 📊 AI 模型全盘数据可视化分析")
    color_map = {'低风险': '#32CD32', '中风险': '#FFD700', '高风险': '#FF6B00', '极高风险': '#FF0000'}
    level_order = ['低风险', '中风险', '高风险', '极高风险']
    
    # 字体配置
    t_font = {'fontproperties': font_prop, 'fontsize': 11, 'weight': 'bold'} if font_prop else {'fontsize': 11}
    l_font = {'fontproperties': font_prop, 'fontsize': 9} if font_prop else {'fontsize': 9}

    # --- 图表一 ---
    st.markdown("#### 一、 无证户概率综合分析")
    fig1, axes1 = plt.subplots(2, 3, figsize=(13, 7.5))
    # 【核心修改】：显著增大 wspace (0.6) 彻底解决图2与图3重叠问题
    fig1.subplots_adjust(hspace=0.4, wspace=0.6)
    
    # 1.1 分布直方图
    for level in level_order:
        subset = df[df['风险等级'] == level]
        if not subset.empty:
            axes1[0, 0].hist(subset['无证户综合概率(%)'], bins=15, color=color_map[level], alpha=0.7, label=level, edgecolor='black')
    axes1[0, 0].set_title('所有商户无证户概率分布', **t_font)
    axes1[0, 0].legend(prop=font_prop if font_prop else None, fontsize=8)

    # 1.2 风险饼图 (移除饼内文字，改用侧边图例防止重叠)
    risk_counts = df['风险等级'].value_counts().reindex(level_order).fillna(0)
    axes1[0, 1].pie(risk_counts, labels=None, autopct=lambda p: f'{p:.1f}%' if p > 3 else '', 
                    colors=[color_map[l] for l in risk_counts.index], startangle=90)
    axes1[0, 1].set_title('所有商户风险等级分布', **t_font)
    axes1[0, 1].legend(risk_counts.index, loc="center left", bbox_to_anchor=(1.0, 0.5), prop=font_prop)

    # 1.3 密度图
    import seaborn as sns
    sns.kdeplot(data=df, x='信用值', hue='风险等级', palette=color_map, ax=axes1[0, 2], fill=True, common_norm=False)
    axes1[0, 2].set_title('信用值密度分布', **t_font)
    axes1[0, 2].set_xlabel('信用值', **l_font)
    axes1[0, 2].set_ylabel('密度', **l_font)
    leg = axes1[0, 2].get_legend()
    if leg: 
        plt.setp(leg.get_texts(), fontproperties=font_prop)
        leg.set_title("风险等级", prop=font_prop)

    # 下排柱状图、散点图、法人饼图同理应用字体注入...
    avg_prob = df.groupby('风险等级')['无证户综合概率(%)'].mean().reindex(level_order)
    axes1[1, 0].bar(avg_prob.index, avg_prob.values, color=[color_map[l] for l in avg_prob.index])
    axes1[1, 0].set_title('各等级平均概率', **t_font)
    axes1[1, 0].set_xticklabels(avg_prob.index, **l_font)

    for level in level_order:
        subset = df[df['风险等级'] == level]
        if not subset.empty: axes1[1, 1].scatter(subset['信用值'], subset['无证户综合概率(%)'], color=color_map[level], s=15, alpha=0.6)
    axes1[1, 1].set_title('信用值 vs 概率散点', **t_font)

    high_risk_reps = df[df['高危法人关联'] == 1].shape[0]
    axes1[1, 2].pie([high_risk_reps, len(df)-high_risk_reps], colors=['#FF6B6B', '#4ECDC4'], autopct='%1.1f%%')
    axes1[1, 2].set_title('法人身份识别比例', **t_font)
    st.pyplot(fig1)

    # --- 图表二 (同样应用 wspace=0.6) ---
    st.markdown("#### 二、 风险等级详细分析")
    fig2, axes2 = plt.subplots(2, 3, figsize=(13, 7.5))
    fig2.subplots_adjust(hspace=0.4, wspace=0.6)
    
    # 抽取部分核心图表展示
    axes2[0, 0].bar(risk_counts.index, risk_counts.values, color=[color_map[l] for l in risk_counts.index])
    axes2[0, 0].set_title('各等级商户数量', **t_font)
    axes2[0, 0].set_xticklabels(risk_counts.index, **l_font)

    box_data = [df[df['风险等级'] == level]['无证户综合概率(%)'].dropna() for level in level_order]
    axes2[0, 1].boxplot(box_data, labels=level_order, patch_artist=True)
    axes2[0, 1].set_title('概率分布箱线图', **t_font)
    axes2[0, 1].set_xticklabels(level_order, **l_font)

    avg_score = df.groupby('风险等级')['信用值'].mean().reindex(level_order)
    axes2[1, 1].bar(avg_score.index, avg_score.values, color=[color_map[l] for l in avg_score.index])
    axes2[1, 1].set_title('各等级平均信用值', **t_font)
    axes2[1, 1].set_xticklabels(avg_score.index, **l_font)

    axes2[1, 2].axis('off')
    axes2[1, 2].set_title('极高风险名单快照', **t_font)
    st.pyplot(fig2)

# ==========================================
# 2. 核心系统界面与演算控制
# ==========================================
st.title("👁️ 卷烟无证经营户动态筛查 AI 模型")

with st.sidebar:
    st.header("📂 1. 数据接入库")
    file_biz = st.file_uploader("上传【营业执照】全量名单", type=["xlsx", "csv"])
    file_unl = st.file_uploader("上传【历史无证户】名单", type=["xlsx", "csv"])
    start_btn = st.button("🚀 2. 启动 AI 深度筛查演算", type="primary", use_container_width=True)

if start_btn:
    if not file_biz or not file_unl:
        st.warning("⚠️ 权限阻断：请先在左侧上传必须的两个数据文件！")
    else:
        st.markdown("### 💻 系统核心演算终端")
        # 恢复经典的 terminal.code 格式
        terminal = st.empty()
        log_lines = []
        def log_to_terminal(message):
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]
            log_lines.append(f"[{timestamp}] {message}")
            if len(log_lines) > 20:
                log_lines.pop(0)
            terminal.code("\n".join(log_lines), language="bash")

        log_to_terminal("[SYSTEM] 正在初始化天眼稽查引擎...")
        biz = pd.read_excel(file_biz) if file_biz.name.endswith('.xlsx') else pd.read_csv(file_biz)
        unl = pd.read_excel(file_unl) if file_unl.name.endswith('.xlsx') else pd.read_csv(file_unl)
        
        # 数据清洗...
        if '天眼评分' in biz.columns: biz.rename(columns={'天眼评分': '信用值'}, inplace=True)
        if '天眼评分' in unl.columns: unl.rename(columns={'天眼评分': '信用值'}, inplace=True)
        
        fill_dict = {'公司名称':'未知', '法定代表人':'未知', '统一社会信用代码':'未知', '信用值':0}
        biz = biz.fillna(fill_dict).drop_duplicates(subset=['统一社会信用代码'])
        unl = unl.fillna(fill_dict).drop_duplicates(subset=['统一社会信用代码'])
        
        log_to_terminal("[GRAPH] 正在构建高危法人关系网络...")
        bad_reps = set(unl[~unl['法定代表人'].isin(['未知', '', '无'])]['法定代表人'].unique())
        biz['高危法人关联'] = biz['法定代表人'].apply(lambda x: 1 if x in bad_reps else 0)
        unl['高危法人关联'] = 1
        
        biz['label'], unl['label'] = 0, 1
        df_all = pd.concat([unl, biz], ignore_index=True)
        
        # NLP 向量化...
        log_to_terminal("[NLP] 启动 TF-IDF 引擎执行文本特征抽取...")
        vec_name = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=1000)
        X_name = vec_name.fit_transform(df_all['公司名称'])
        scaler = MinMaxScaler()
        X_numeric = scaler.fit_transform(df_all[['信用值', '高危法人关联']])
        X_combined = sp.hstack((X_name, X_numeric))
        y_combined = df_all['label'].values

        # 训练模型...
        log_to_terminal("[ML] 激活随机森林 200 个并发决策引擎...")
        ml_model = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', n_jobs=-1)
        ml_model.fit(X_combined, y_combined)
        all_probs = ml_model.predict_proba(X_combined)[:, 1]
        
        # 结果处理
        df_all['无证户综合概率(%)'] = np.round(all_probs * 100, 2)
        target_pool = df_all[df_all['label'] == 0].copy()
        
        # 【核心修改】：精准定罪逻辑，显示具体关联的法人姓名
        log_to_terminal("[EXPLAINER] 正在进行概率溯源及证据链解析...")
        importances = ml_model.feature_importances_
        feature_names = np.array(vec_name.get_feature_names_out().tolist() + ['信用异常惩罚', '历史无证前科'])
        
        explanations = []
        X_target = X_combined.tocsr()[target_pool.index.tolist()]
        weighted_X = X_target.multiply(importances).tocsr()
        
        for i in range(weighted_X.shape[0]):
            row = weighted_X.getrow(i)
            top_idx = row.indices[row.data.argsort()[-3:][::-1]]
            final_prob = target_pool.iloc[i]['无证户综合概率(%)']
            total_w = row.data.sum()
            
            parts = []
            for idx in top_idx:
                feat = feature_names[idx]
                contrib = (row.data[row.indices == idx][0] / total_w) * final_prob if total_w > 0 else 0
                if feat == '历史无证前科':
                    # 穿透显示具体姓名
                    rep_name = target_pool.iloc[i]['法定代表人']
                    feat = f"关联高危前科法人[{rep_name}]"
                parts.append(f"{feat}({contrib:.1f}%)")
            explanations.append(" + ".join(parts) if parts else "背景综合评估")
            
        target_pool['AI 判定依据'] = explanations

        # 风险定级
        def assign_risk(prob):
            if prob >= 85: return '极高风险', '🚨 立即排查'
            elif prob >= 65: return '高风险', '⚠️ 重点监控'
            elif prob >= 40: return '中风险', '👀 定期关注'
            else: return '低风险', '✅ 常规监管'
            
        target_pool[['风险等级', '监管建议']] = target_pool.apply(lambda r: pd.Series(assign_risk(r['无证户综合概率(%)'])), axis=1)
        target_pool = target_pool.sort_values(by='无证户综合概率(%)', ascending=False)
        
        log_to_terminal("[SYSTEM] ✅ 演算任务全量闭环。生成分析简报。")

        # ==========================================
        # 布局重构：名单置顶，图表置后
        # ==========================================
        st.success("🎯 智能筛查任务收官！")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("极高风险锁定", f"{len(target_pool[target_pool['风险等级'] == '极高风险'])} 家", "需行动")
        col2.metric("高危法人揪出", f"{target_pool['高危法人关联'].sum()} 人", "存在记录")
        col3.metric("总计排查目标", f"{len(target_pool)} 家", "实时")

        st.divider()
        st.subheader("🚨 极高风险打击名单 (名单已置顶)")
        display_cols = ['公司名称', '无证户综合概率(%)', 'AI 判定依据', '风险等级', '监管建议', '法定代表人', '信用值']
        st.dataframe(target_pool[display_cols].head(15), use_container_width=True)
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            target_pool[display_cols].to_excel(writer, index=False)
        st.download_button("📥 导出作战名单", buffer, "风险排查名单.xlsx")

        st.divider()
        draw_analysis_charts(target_pool)
