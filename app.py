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
from sklearn.mixture import GaussianMixture
import urllib.request
import os
import io
import time
import re

# ==========================================
# 0. 基础环境配置 (中文字体支持)
# ==========================================
st.set_page_config(page_title="无证户智能稽查天眼", page_icon="👁️", layout="wide")

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
# 【防线1：将“超市”、“商行”等无业务实质的格式词加入黑名单】
CUSTOM_STOP_WORDS = {
    '徐州','徐州市','江苏','江苏省','地址','未知','公司','店铺','个体','工商户',
    '商贸','企业','中心','工作室','经营部','销售部','市','省','区','县',
    '项目','活动','服务','管理','咨询','开发','贸易','代理','批发','零售','销售',
    '批零','兼营','制造','加工','用品','制品','器材','物资','产品','设备','科技',
    '发展','实业','经营','相关','业务','一般','许可','包含','商行','厂','店',
    '提供','预包装','散装','其他','一切','合法','许可项目','一般项目','沛县',
    '睢宁','泉山','云龙','丰县','邳州','经开','铜山','新沂','贾汪','睢宁县','山区',
    '超市','商行','商店','专卖店','专营店','门市','卖场' # <-- 新增业态过滤
}
TOBACCO_WORDS = {'烟草','卷烟','雪茄','烟丝','香烟','电子烟','烟具','电子烟雾化物'}

def custom_tokenizer(text):
    if not isinstance(text, str) or not text: return []
    norm_map = {'百货店':'百货','百货商场':'百货','百货公司':'百货','百货超市':'百货','便利店':'便利'}
    
    # 1. 基础分词
    raw_words = jieba.lcut(text)
    
    # 2. 过滤掉标点符号等无意义字符，只保留纯文字
    valid_words = [norm_map.get(w, w) for w in raw_words if re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9]+$', w)]
    
    # 3. 扩充分词规模 (N-gram)：提取单词的同时，将相邻的两个有效词组合
    tokens = []
    for i in range(len(valid_words)):
        tokens.append(valid_words[i])
        if i > 0:
            tokens.append(valid_words[i-1] + valid_words[i])
            
    processed_words = []
    for w in tokens:
        if len(w) > 1 and w not in CUSTOM_STOP_WORDS and not any(tob in w for tob in TOBACCO_WORDS):
            processed_words.append(w)
    return processed_words

# ==========================================
# 2. 可视化模块
# ==========================================
def draw_analysis_charts(df, t_font, l_font):
    st.markdown("### 📊 智能筛查模型全盘数据可视化分析")
    color_map = {'低风险': '#32CD32', '中风险': '#FFD700', '高风险': '#FF6B00', '极高风险': '#FF0000'}
    level_order = ['低风险', '中风险', '高风险', '极高风险']
    
    st.markdown("#### 一、 无证户概率综合分析")
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8))
    
    for level in level_order:
        subset = df[df['风险等级'] == level]
        if not subset.empty:
            axes1[0, 0].hist(subset['无证户综合概率(%)'], bins=15, color=color_map[level], alpha=0.7, label=level, edgecolor='black')
    axes1[0, 0].set_title('所有商户无证户概率分布', fontproperties=t_font)
    axes1[0, 0].legend(prop=l_font)

    risk_counts = df['风险等级'].value_counts().reindex(level_order).fillna(0)
    axes1[0, 1].pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=[color_map[l] for l in risk_counts.index], startangle=90, textprops={'fontproperties': l_font})
    axes1[0, 1].set_title('所有商户风险等级分布', fontproperties=t_font)

    sns.kdeplot(data=df, x='信用值', hue='风险等级', palette=color_map, ax=axes1[0, 2], fill=True, common_norm=False)
    axes1[0, 2].set_title('信用值密度分布', fontproperties=t_font)
    legend = axes1[0, 2].get_legend()
    if legend: plt.setp(legend.texts, fontproperties=l_font)

    avg_prob = df.groupby('风险等级')['无证户综合概率(%)'].mean().reindex(level_order)
    axes1[1, 0].bar(avg_prob.index, avg_prob.values, color=[color_map[l] for l in avg_prob.index])
    axes1[1, 0].set_title('各等级平均概率', fontproperties=t_font)
    axes1[1, 0].set_xticklabels(avg_prob.index, fontproperties=l_font)

    for level in level_order:
        subset = df[df['风险等级'] == level]
        if not subset.empty:
            axes1[1, 1].scatter(subset['信用值'], subset['无证户综合概率(%)'], color=color_map[level], label=level, alpha=0.6, s=15)
    axes1[1, 1].set_title('信用值 vs 概率散点', fontproperties=t_font)
    axes1[1, 1].legend(prop=l_font)

    top_risk_df = df[df['风险等级'].isin(['极高风险', '高风险'])]
    if not top_risk_df.empty:
        name_lengths = top_risk_df['公司名称'].astype(str).apply(len)
        sizes = [len(name_lengths[name_lengths < 6]), len(name_lengths[(name_lengths >= 6) & (name_lengths <= 12)]), len(name_lengths[name_lengths > 12])]
        axes1[1, 2].pie(sizes, labels=['<6字', '6-12字', '>12字'], autopct='%1.1f%%', colors=['#FF9999', '#66B2FF', '#99FF99'], startangle=140, textprops={'fontproperties': l_font})
        axes1[1, 2].set_title('高危目标名称字数特征', fontproperties=t_font)
    else:
        axes1[1, 2].axis('off')
    
    fig1.tight_layout(pad=3.0) 
    st.pyplot(fig1)

    st.markdown("#### 二、 风险等级详细分析")
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    
    bars = axes2[0, 0].bar(risk_counts.index, risk_counts.values, color=[color_map[l] for l in risk_counts.index])
    axes2[0, 0].set_title('商户数量分布', fontproperties=t_font)
    axes2[0, 0].set_xticklabels(risk_counts.index, fontproperties=l_font)

    box_data = [df[df['风险等级'] == level]['无证户综合概率(%)'].dropna() for level in level_order]
    axes2[0, 1].boxplot(box_data, labels=level_order, patch_artist=True)
    axes2[0, 1].set_title('各等级概率箱线', fontproperties=t_font)
    axes2[0, 1].set_xticklabels(level_order, fontproperties=l_font)

    box_data_score = [df[df['风险等级'] == level]['信用值'].dropna() for level in level_order]
    axes2[0, 2].boxplot(box_data_score, labels=level_order, patch_artist=True)
    axes2[0, 2].set_title('信用值分布箱线', fontproperties=t_font)
    axes2[0, 2].set_xticklabels(level_order, fontproperties=l_font)

    sorted_probs = np.sort(df['无证户综合概率(%)'])
    cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs) * 100
    axes2[1, 0].plot(sorted_probs, cumulative, color='#1E90FF', linewidth=2)
    axes2[1, 0].set_title('概率累积分布曲线', fontproperties=t_font)

    avg_score = df.groupby('风险等级')['信用值'].mean().reindex(level_order)
    axes2[1, 1].bar(avg_score.index, avg_score.values, color=[color_map[l] for l in avg_score.index])
    axes2[1, 1].set_title('平均信用分', fontproperties=t_font)
    axes2[1, 1].set_xticklabels(avg_score.index, fontproperties=l_font)

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
# 3. 核心加载逻辑与主程序
# ==========================================
st.title("👁️ 卷烟无证经营户智能筛查模型 ")

with st.sidebar:
    st.header("📂 1. 数据接入库")
    file_lic_list = st.file_uploader("1️⃣ 上传【持证户名录】", type=["xlsx", "csv"], accept_multiple_files=True)
    file_unl_list = st.file_uploader("2️⃣ 上传【无证户名录】 ", type=["xlsx", "csv"], accept_multiple_files=True)
    file_biz_list = st.file_uploader("3️⃣ 上传【营业执照名录】", type=["xlsx", "csv"], accept_multiple_files=True)
    st.info("💡 核心逻辑：基于统一社会信用代码进行匹配，从大盘名单中剥离持证户，留下【经营范围有烟但无烟草证】的商户，分析现有无证户特征进行建模，筛选商户。")
    start_btn = st.button("🚀 2. 启动深度筛查演算", type="primary", use_container_width=True)

def load_uploaded_files(file_list):
    df_list = []
    for f in file_list:
        is_excel = f.name.endswith('.xlsx') or f.name.endswith('.xls')
        df = pd.read_excel(f) if is_excel else pd.read_csv(f)
        
        if len(df.columns) > 0 and '声明' in str(df.columns[0]):
            f.seek(0)
            df = pd.read_excel(f, header=1) if is_excel else pd.read_csv(f, header=1)
            if len(df.columns) > 0:
                last_col_name = df.columns[-1]
                df = df.rename(columns={last_col_name: '信用值'})
                
        df_list.append(df)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()


if start_btn:
    if not file_biz_list or not file_lic_list:
        st.warning("⚠️ 权限阻断：请至少上传【持证户名录】和【营业执照名录】！")
    else:
        st.markdown("---")
        st.markdown("### 💻 系统核心演算终端")
        log_container = st.container(height=400)
        terminal = log_container.empty()
        log_lines = []
        
        def log_to_terminal(message, delay=0.1):
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]
            log_lines.insert(0, f"[{timestamp}] {message}")
            display_text = "▼ 实时终端日志 [最新指令始终在最上方显示]\n" + "="*70 + "\n" + "\n".join(log_lines)
            terminal.code(display_text, language="bash")
            time.sleep(delay)

        start_time = time.time()

        # --- 步骤 1: 数据加载 ---
        log_to_terminal("[SYSTEM] 正在初始化天眼稽查引擎，读取异构数据源...")
        log_to_terminal("[SYSTEM] 分配核心内存空间，执行多文件数据流合并...")
        lic = load_uploaded_files(file_lic_list)
        unl = load_uploaded_files(file_unl_list) if file_unl_list else pd.DataFrame()
        biz = load_uploaded_files(file_biz_list)
        
        log_to_terminal(f"[DATA] 数据加载完毕。大盘执照 {len(biz)} 条，持证库 {len(lic)} 条，历史无证 {len(unl)} 条。")

        # --- 步骤 2: 底层数据清洗与表头对齐 ---
        log_to_terminal("[CLEAN] 启动底层数据清洗管线：对齐异构表头并强制规范化...")
        
        for df_temp in [biz, unl, lic]:
            if not df_temp.empty:
                df_temp.columns = df_temp.columns.str.strip()

        rename_rules = {
            '企业(字号)名称': '公司名称',
            '企业（字号）名称': '公司名称',
            '企业名称': '公司名称',
            '经营人': '法定代表人',
            '持证人': '法定代表人',
            '负责人': '法定代表人'
        }
        
        biz = biz.rename(columns=rename_rules)
        if not unl.empty: unl = unl.rename(columns=rename_rules)
        lic = lic.rename(columns=rename_rules)

        required_cols = {'公司名称': '未知', '法定代表人': '未知', '经营范围': '未知', '信用值': 0, '统一社会信用代码': '未知'}
        
        for df_temp in [biz, unl, lic]:
            if not df_temp.empty:
                for col, default_val in required_cols.items():
                    if col not in df_temp.columns: 
                        df_temp[col] = default_val
                    df_temp[col] = df_temp[col].fillna(default_val)
                
                df_temp['统一社会信用代码'] = df_temp['统一社会信用代码'].astype(str).str.strip().str.upper()

        biz['信用值'] = pd.to_numeric(biz['信用值'], errors='coerce').fillna(0)
        if not unl.empty: unl['信用值'] = pd.to_numeric(unl['信用值'], errors='coerce').fillna(0)
        
        log_to_terminal("[CLEAN] 缺失值探测完毕，已安全将异常信用分转换为纯数字类型。")

        # ==============================================================
        # --- 步骤 2.5: 【核心逻辑】基于统一社会信用
