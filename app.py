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
CUSTOM_STOP_WORDS = {'有限','责任','分公司','集团','控股','股份','有限公司','徐州','地址','未知','公司', '店铺'}
TOBACCO_WORDS = {'烟草制品零售','卷烟零售','雪茄零售','烟丝零售','香烟销售','烟草销售','烟草','卷烟','雪茄','烟丝','香烟'}

def custom_tokenizer(text):
    if not isinstance(text, str) or not text: return []
    norm_map = {'百货店':'百货','百货商场':'百货','百货公司':'百货','百货超市':'百货','便利店':'便利','批发部':'批发'}
    words = jieba.lcut(text)
    processed_words = [norm_map.get(w, w) for w in words if len(w) > 1 and w not in CUSTOM_STOP_WORDS and not any(tob_w in w for tob_w in TOBACCO_WORDS)]
    return processed_words

# ==========================================
# 2. 可视化模块
# ==========================================
def draw_analysis_charts(df, t_font, l_font):
    st.markdown("### 📊 AI 模型全盘数据可视化分析")
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

    high_risk_reps = df[df['该商户负责人是否在无证户名录（可能重名）'] == '是（可能重名）'].shape[0]
    axes1[1, 2].pie([high_risk_reps, df.shape[0] - high_risk_reps], labels=['历史无证重名', '普通法人'], autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'], startangle=140, textprops={'fontproperties': l_font})
    axes1[1, 2].set_title('法人身份重名比例', fontproperties=t_font)
    
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
st.title("👁️ 卷烟无证经营户动态筛查 AI 模型 (精准洗涤版)")

with st.sidebar:
    st.header("📂 1. 数据接入库")
    file_biz_list = st.file_uploader("上传【营业执照】名单 (支持多选)", type=["xlsx", "csv"], accept_multiple_files=True)
    file_unl_list = st.file_uploader("上传【历史无证户】名单 (支持多选)", type=["xlsx", "csv"], accept_multiple_files=True)
    file_lic_list = st.file_uploader("上传【现有持证户】名单 (可选 / 自动剔除)", type=["xlsx", "csv"], accept_multiple_files=True)
    st.info("💡 提示：系统将通过【统一社会信用代码】自动从营业执照中剔除“现有持证户”和“历史无证户”。")
    start_btn = st.button("🚀 2. 启动 AI 深度筛查演算", type="primary", use_container_width=True)

def load_uploaded_files(file_list):
    df_list = []
    for f in file_list:
        df = pd.read_excel(f) if f.name.endswith('.xlsx') else pd.read_csv(f)
        if '声明' in str(df.columns[0]) or '公司名称' not in df.columns:
            f.seek(0)
            df = pd.read_excel(f, header=1) if f.name.endswith('.xlsx') else pd.read_csv(f, header=1)
        if len(df.columns) > 0:
            last_col_name = df.columns[-1]
            df = df.rename(columns={last_col_name: '信用值'})
        df_list.append(df)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()


if start_btn:
    if not file_biz_list or not file_unl_list:
        st.warning("⚠️ 权限阻断：请先在左侧分别上传至少一个营业执照和历史无证户数据文件！")
    else:
        st.markdown("### 💻 系统核心演算终端")
        log_container = st.container(height=350)
        terminal = log_container.empty()
        log_lines = []
        
        def log_to_terminal(message, delay=0.1):
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]
            log_lines.insert(0, f"[{timestamp}] {message}")
            display_text = "▼ 实时终端日志 [最新指令始终在最上方显示]\n" + "="*65 + "\n" + "\n".join(log_lines)
            terminal.code(display_text, language="bash")
            time.sleep(delay)

        start_time = time.time()

        # --- 步骤 1: 初始化与数据加载 ---
        log_to_terminal("[SYSTEM] 正在初始化天眼稽查引擎...")
        biz = load_uploaded_files(file_biz_list)
        unl = load_uploaded_files(file_unl_list)
        lic = load_uploaded_files(file_lic_list) if file_lic_list else pd.DataFrame()
        
        log_msg = f"[DATA] 合并后营业执照 {len(biz)} 条，无证卷宗 {len(unl)} 条"
        if not lic.empty: log_msg += f"，持证白名单 {len(lic)} 条。"
        else: log_msg += "。"
        log_to_terminal(log_msg)

        # --- 步骤 2: 数据清洗与强制格式化 (解决匹配为0的核心) ---
        log_to_terminal("[CLEAN] 启动底层数据清洗：清除所有隐藏空格与异常格式...")
        required_cols = {'公司名称': '未知', '法定代表人': '未知', '经营范围': '未知', '信用值': 0, '统一社会信用代码': '未知'}
        
        for df_temp in [biz, unl, lic]:
            if not df_temp.empty:
                for col, default_val in required_cols.items():
                    if col not in df_temp.columns: df_temp[col] = default_val
                df_temp.fillna(required_cols, inplace=True)
                
                # 【防污染核心代码】：强制转为字符串并去除两端空格（防Excel导出带隐形空格）
                for text_col in ['公司名称', '法定代表人', '统一社会信用代码']:
                    df_temp[text_col] = df_temp[text_col].astype(str).str.strip()

        biz['信用值'] = pd.to_numeric(biz['信用值'], errors='coerce').fillna(0)
        unl['信用值'] = pd.to_numeric(unl['信用值'], errors='coerce').fillna(0)

        # --- 步骤 2.5: 使用信用代码剔除持证户与无证户 ---
        log_to_terminal("[FILTER] 正在使用【统一社会信用代码】进行全库交叉比对与自动剥离...")
        exclude_codes = set()
        
        # 将无证和持证的信用代码都放进排除名单
        if not lic.empty:
            lic_codes = set(lic[~lic['统一社会信用代码'].isin(['未知', '', 'nan'])]['统一社会信用代码'].unique())
            exclude_codes.update(lic_codes)
        
        if not unl.empty:
            unl_codes = set(unl[~unl['统一社会信用代码'].isin(['未知', '', 'nan'])]['统一社会信用代码'].unique())
            exclude_codes.update(unl_codes)

        orig_biz_len = len(biz)
        if exclude_codes:
            # 只保留不在排除名单里的营业执照
            biz = biz[~biz['统一社会信用代码'].isin(exclude_codes)]
            filtered_count = orig_biz_len - len(biz)
            log_to_terminal(f"[FILTER] 🟢 清洗完毕！已成功从营业执照中剔除 {filtered_count} 家(持证或已知无证)的商户。")
        else:
            log_to_terminal("[FILTER] ⚠️ 未提取到有效的信用代码用于过滤。")
            
        log_to_terminal(f"[FILTER] 剩余 {len(biz)} 家“纯待查”商户，即将开展核心演算。")

        # --- 步骤 3: 实体特征提取与法人比对 ---
        log_to_terminal("[GRAPH] 正在从历史档案提取核心实体，执行跨表网络穿透比对...")
        bad_reps = set(unl[~unl['法定代表人'].isin(['未知', '', 'nan', '无'])]['法定代表人'].unique())
        biz['该商户负责人是否在无证户名录（可能重名）'] = biz['法定代表人'].apply(lambda x: '是（可能重名）' if x in bad_reps else '否')
        
        biz['label'], unl['label'] = 0, 1
        df_all = pd.concat([unl, biz], ignore_index=True)
        log_to_terminal(f"[GRAPH] 成功跨表追踪到 {len(bad_reps)} 个高危法人特征，污染链条标记完毕。")

        # --- 步骤 4: NLP 语义映射 ---
        log_to_terminal("[NLP] 启动独立特征工程：公司名称与经营范围文本解析...")
        
        vec_name = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=500)
        X_name = vec_name.fit_transform(df_all['公司名称'])
        model_name = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42).fit(X_name, df_all['label'])
        prob_name = model_name.predict_proba(X_name)[:, 1]

        vec_scope = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=500)
        X_scope = vec_scope.fit_transform(df_all['经营范围'])
        model_scope = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42).fit(X_scope, df_all['label'])
        prob_scope = model_scope.predict_proba(X_scope)[:, 1]

        log_to_terminal("[MATH] 启动高斯混合模型 (GMM) 进行信用值概率分布映射...")
        credit_values = df_all[['信用值']].values
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(credit_values)
        risk_component_idx = np.argmin(gmm.means_.flatten())
        prob_credit = gmm.predict_proba(credit_values)[:, risk_component_idx]

        # --- 步骤 5: AI 权重融合 (30-50-20 配比) ---
        log_to_terminal("[ML-CORE] 正在执行三权融合决策 (名称30% | 范围50% | 信用20%)...")
        combined_prob = (prob_name * 0.30) + (prob_scope * 0.50) + (prob_credit * 0.20)
        df_all['无证户综合概率(%)'] = np.round(combined_prob * 100, 2)
        # 目标池仅保留刚才筛出的 biz 商户 (label == 0)
        target_pool = df_all[df_all['label'] == 0].copy()
        
        # --- 步骤 6: 白盒归因与具体内容提取 ---
        log_to_terminal("[EXPLAINER] 激活白盒解释器，生成可追溯证据链...")
        explanations = []
        name_features = vec_name.get_feature_names_out()
        scope_features = vec_scope.get_feature_names_out()
        
        for idx in range(len(target_pool)):
            row_n = X_name.getrow(target_pool.index[idx])
            top_word_n = "常规名"
            if row_n.nnz > 0:
                top_idx_n = row_n.toarray()[0].argsort()[-1]
                top_word_n = name_features[top_idx_n]
            
            row_s = X_scope.getrow(target_pool.index[idx])
            top_word_s = "常规业务"
            if row_s.nnz > 0:
                top_idx_s = row_s.toarray()[0].argsort()[-1]
                top_word_s = scope_features[top_idx_s]
                
            orig_credit = target_pool.iloc[idx]['信用值']
            p_n = prob_name[target_pool.index[idx]] * 30.0
            p_s = prob_scope[target_pool.index[idx]] * 50.0
            p_c = prob_credit[target_pool.index[idx]] * 20.0
            
            explanations.append(f"{top_word_n}({p_n:.1f}%) + {top_word_s}({p_s:.1f}%) + 信用值{int(orig_credit)}({p_c:.1f}%)")
        
        target_pool['AI 判定依据'] = explanations

        # --- 风险定级 ---
        def assign_risk(p):
            if p >= 80: return '极高风险', '🚨 立即排查'
            elif p >= 60: return '高风险', '⚠️ 重点监控'
            elif p >= 35: return '中风险', '👀 定期关注'
            return '低风险', '✅ 常规监管'
        target_pool[['风险等级', '监管建议']] = target_pool.apply(lambda r: pd.Series(assign_risk(r['无证户综合概率(%)'])), axis=1)
        target_pool = target_pool.sort_values('无证户综合概率(%)', ascending=False)
        
        elapsed_time = time.time() - start_time
        calc_speed = int(len(target_pool) / max(elapsed_time, 0.001))
        log_to_terminal(f"[SYSTEM] ✅ 演算结束！用时 {elapsed_time:.2f} 秒。系统正在生成大屏...")

        # --- 结果展示区 ---
        st.success("🎯 稽查演算收官！已成功从底册中精准剔除重复项，完成净网测算。")
        m1, m2, m3, m4 = st.columns(4)
        total = len(target_pool)
        
        m1.metric("极高风险数量 (80%-100%)", f"{len(target_pool[target_pool['风险等级']=='极高风险'])} 家", f"占纯净底册 {len(target_pool[target_pool['风险等级']=='极高风险'])/total*100:.2f}%" if total >0 else "0%")
        m2.metric("高风险数量 (60%-79%)", f"{len(target_pool[target_pool['风险等级']=='高风险'])} 家", f"占纯净底册 {len(target_pool[target_pool['风险等级']=='高风险'])/total*100:.2f}%" if total >0 else "0%")
        m3.metric("中风险数量 (35%-59%)", f"{len(target_pool[target_pool['风险等级']=='中风险'])} 家", f"占纯净底册 {len(target_pool[target_pool['风险等级']=='中风险'])/total*100:.2f}%" if total >0 else "0%")
        m4.metric("锁定待查总规模", f"{total} 条", f"AI筛查时效: 极速 ({calc_speed} 条/秒)")

        st.divider()

        with st.expander("💡 了解本模型如何实现“净网排查”？", expanded=True):
            st.info("""
            **1. 数据防污染清洗**：导出数据往往带有肉眼看不见的空格字符，导致 AI 认为“张三”和“ 张三 ”是两个人。本模型增加了强制清洗层，100% 提取有效比对字符串。
            
            **2. 统一社会信用代码过滤**：系统将【持证户名单】和【历史无证户】提取统一社会信用代码合并为“已知对象池”。只要营业执照中的企业信用代码落入这个池子，立刻剥离不作计算，确保算力集中在未知风险商户上。
            """)

        # --- 打击名单 ---
        st.subheader("🚨 稽查排查名单 TOP 20（风险度从高到低排序，已去除已知名单）")
        display_cols = ['公司名称', '统一社会信用代码', '无证户综合概率(%)', 'AI 判定依据', '风险等级', '监管建议', '法定代表人', '该商户负责人是否在无证户名录（可能重名）']
        
        if '注册地址' in target_pool.columns:
            display_cols.insert(-1, '注册地址')

        st.dataframe(
            target_pool[display_cols].head(20).style.format({"无证户综合概率(%)": "{:.2f}%"})
            .map(lambda x: 'color: red; font-weight: bold' if x == '极高风险' else '', subset=['风险等级']),
            use_container_width=True
        )

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            target_pool[display_cols].to_excel(writer, index=False)
        st.download_button("📥 导出最终排查名单 (净网版)", buffer, "天眼查风险排查名单_净网纯净版.xlsx", "application/vnd.ms-excel")

        st.divider()
        draw_analysis_charts(target_pool, title_font, label_font)
