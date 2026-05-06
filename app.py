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
st.title("👁️ 卷烟无证经营户动态筛查 AI 模型 (信用代码漏斗版)")

with st.sidebar:
    st.header("📂 1. 数据接入库")
    file_lic_list = st.file_uploader("1️⃣ 上传【现有持证户】名单 (支持多选)", type=["xlsx", "csv"], accept_multiple_files=True)
    file_unl_list = st.file_uploader("2️⃣ 上传【历史无证户】名单 (用于 AI 训练)", type=["xlsx", "csv"], accept_multiple_files=True)
    file_biz_list = st.file_uploader("3️⃣ 上传【天眼查营业执照】大盘名单", type=["xlsx", "csv"], accept_multiple_files=True)
    st.info("💡 核心逻辑：基于统一社会信用代码进行匹配，从大盘名单中剥离持证户，留下【范围有烟但无证】的高危盲区。")
    start_btn = st.button("🚀 2. 启动 AI 深度筛查演算", type="primary", use_container_width=True)

def load_uploaded_files(file_list):
    df_list = []
    for f in file_list:
        is_excel = f.name.endswith('.xlsx') or f.name.endswith('.xls')
        df = pd.read_excel(f) if is_excel else pd.read_csv(f)
        
        # 【致命错误已修复】：极其严格地判断，只有第一列包含“声明”字样，才断定是天眼查文件并跳过表头
        if len(df.columns) > 0 and '声明' in str(df.columns[0]):
            f.seek(0)
            df = pd.read_excel(f, header=1) if is_excel else pd.read_csv(f, header=1)
            # 只有天眼查的数据，才强制将最后一列作为信用值
            if len(df.columns) > 0:
                last_col_name = df.columns[-1]
                df = df.rename(columns={last_col_name: '信用值'})
                
        df_list.append(df)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()


if start_btn:
    if not file_biz_list or not file_lic_list:
        st.warning("⚠️ 权限阻断：请至少上传【持证户名单】和【天眼查大盘名单】！")
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
        lic = load_uploaded_files(file_lic_list)
        unl = load_uploaded_files(file_unl_list) if file_unl_list else pd.DataFrame()
        biz = load_uploaded_files(file_biz_list)
        
        log_to_terminal(f"[DATA] 数据加载完毕。大盘执照 {len(biz)} 条，持证库 {len(lic)} 条。")

        # --- 步骤 2: 底层数据清洗与表头对齐 ---
        log_to_terminal("[CLEAN] 启动底层数据清洗：对齐表头并强制规范化信用代码...")
        
        for df_temp in [biz, unl, lic]:
            if not df_temp.empty:
                df_temp.columns = df_temp.columns.str.strip()

        # 映射字典：确保都能找到信用代码和法人
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
                
                # 【极其关键】清洗统一社会信用代码：去空格、转大写，确保 100% 能匹配上
                df_temp['统一社会信用代码'] = df_temp['统一社会信用代码'].astype(str).str.strip().str.upper()

        biz['信用值'] = pd.to_numeric(biz['信用值'], errors='coerce').fillna(0)
        if not unl.empty: unl['信用值'] = pd.to_numeric(unl['信用值'], errors='coerce').fillna(0)

        # ==============================================================
        # --- 步骤 2.5: 【核心逻辑】基于统一社会信用代码剔除已持证户 ---
        # ==============================================================
        log_to_terminal("[FILTER] 启动核心漏斗逻辑：【天眼查大盘】 - 【持证库】 = 疑似无证盲区")
        
        invalid_strs = {'未知', '', 'NAN', 'NAT', 'NONE', '无'}

        # 提取真实有效的持证户【统一社会信用代码】集合
        lic_codes = set(lic[~lic['统一社会信用代码'].isin(invalid_strs)]['统一社会信用代码'])
        log_to_terminal(f"[DEBUG] 提取成功！已从【持证库】精准抓取 {len(lic_codes)} 个有效社会信用代码/注册号。")

        # 提取历史无证户的信用代码
        if not unl.empty:
            unl_codes = set(unl[~unl['统一社会信用代码'].isin(invalid_strs)]['统一社会信用代码'])
            lic_codes.update(unl_codes)

        orig_biz_len = len(biz)
        
        # 🔥 核心剔除动作：只保留大盘里【不在】排除名单里的商户
        biz = biz[~biz['统一社会信用代码'].isin(lic_codes)]
        
        filtered_count = orig_biz_len - len(biz)
        log_to_terminal(f"[FILTER] 🟢 过滤成功！基于唯一信用代码，已从大盘中精准剔除 {filtered_count} 家商户。")
        log_to_terminal(f"[FILTER] 最终锁定 {len(biz)} 家【范围涉烟，但未持证】的高危盲区，准备开展 AI 综合研判。")

        # --- 步骤 3 & 4 & 5: AI 概率预测演算 ---
        log_to_terminal("[ML-CORE] 正在启动自然语言处理模型，解析企业图谱与文本风险...")
        
        if not unl.empty:
            bad_reps = set(unl[~unl['法定代表人'].isin(invalid_strs)]['法定代表人'].unique())
            biz['该商户负责人是否在无证户名录（可能重名）'] = biz['法定代表人'].apply(lambda x: '是（可能重名）' if x in bad_reps else '否')
            biz['label'], unl['label'] = 0, 1
            df_all = pd.concat([unl, biz], ignore_index=True)
        else:
            biz['该商户负责人是否在无证户名录（可能重名）'] = '否'
            biz['label'] = 0
            df_all = biz.copy()

        # 文本特征工程
        vec_name = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=500)
        X_name = vec_name.fit_transform(df_all['公司名称'])
        
        vec_scope = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=500)
        X_scope = vec_scope.fit_transform(df_all['经营范围'])
        
        # 信用值 GMM 映射
        log_to_terminal("[MATH] 激活高斯混合模型 (GMM)，执行企业信用异动偏离度测算...")
        credit_values = df_all[['信用值']].values
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(credit_values)
        risk_component_idx = np.argmin(gmm.means_.flatten())
        prob_credit = gmm.predict_proba(credit_values)[:, risk_component_idx]

        if not unl.empty:
            model_name = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42).fit(X_name, df_all['label'])
            prob_name = model_name.predict_proba(X_name)[:, 1]
            model_scope = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42).fit(X_scope, df_all['label'])
            prob_scope = model_scope.predict_proba(X_scope)[:, 1]
            combined_prob = (prob_name * 0.30) + (prob_scope * 0.50) + (prob_credit * 0.20)
        else:
            combined_prob = prob_credit * 1.0  

        df_all['无证户综合概率(%)'] = np.round(combined_prob * 100, 2)
        target_pool = df_all[df_all['label'] == 0].copy()
        
        # --- 步骤 6: 白盒归因 ---
        log_to_terminal("[EXPLAINER] AI 引擎演算完毕，正在封装最终可追溯证据链...")
        explanations = []
        name_features = vec_name.get_feature_names_out()
        scope_features = vec_scope.get_feature_names_out()
        
        for idx in range(len(target_pool)):
            row_n = X_name.getrow(target_pool.index[idx])
            top_word_n = name_features[row_n.toarray()[0].argsort()[-1]] if row_n.nnz > 0 else "常规名"
            row_s = X_scope.getrow(target_pool.index[idx])
            top_word_s = scope_features[row_s.toarray()[0].argsort()[-1]] if row_s.nnz > 0 else "常规业务"
                
            orig_credit = target_pool.iloc[idx]['信用值']
            p_c = prob_credit[target_pool.index[idx]] * 20.0
            
            if not unl.empty:
                p_n = prob_name[target_pool.index[idx]] * 30.0
                p_s = prob_scope[target_pool.index[idx]] * 50.0
                explanations.append(f"{top_word_n}({p_n:.1f}%) + {top_word_s}({p_s:.1f}%) + 信用风险({p_c:.1f}%)")
            else:
                explanations.append(f"经营范围涉卷烟 + 企业信用评分偏离")
        
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
        log_to_terminal(f"[SYSTEM] ✅ 任务圆满收官！总计用时 {elapsed_time:.2f} 秒，系统正在生成动态大屏...")

        # --- 结果展示区 ---
        st.success("🎯 过滤完成！已基于【统一社会信用代码】彻底从大盘中清除了持证商户，锁定了经营盲区。")
        m1, m2, m3, m4 = st.columns(4)
        total = len(target_pool)
        
        m1.metric("极高风险数量 (80%-100%)", f"{len(target_pool[target_pool['风险等级']=='极高风险'])} 家", f"占盲区底册 {len(target_pool[target_pool['风险等级']=='极高风险'])/total*100:.2f}%" if total >0 else "0%")
        m2.metric("高风险数量 (60%-79%)", f"{len(target_pool[target_pool['风险等级']=='高风险'])} 家", f"占盲区底册 {len(target_pool[target_pool['风险等级']=='高风险'])/total*100:.2f}%" if total >0 else "0%")
        m3.metric("中风险数量 (35%-59%)", f"{len(target_pool[target_pool['风险等级']=='中风险'])} 家", f"占盲区底册 {len(target_pool[target_pool['风险等级']=='中风险'])/total*100:.2f}%" if total >0 else "0%")
        m4.metric("精准锁定总规模", f"{total} 条", f"AI筛查时效: 极速 ({calc_speed} 条/秒)")

        st.divider()

        with st.expander("💡 关于本次漏斗净网过滤的逻辑说明", expanded=True):
            st.info("""
            **1. 第一重判定 (范围界定)**：您导入的天眼查大盘（文件3），本身就是带有“卷烟”相关经营范围的企业库。系统将其视为所有潜在的售烟商户。
            **2. 第二重判定 (合法剥离)**：您导入的现有持证库（文件1），是系统认可的合法经营白名单。
            **3. 机器级精准对撞**：模型强制提取两份名单中的【统一社会信用代码】进行一对一碰撞（无视名字重名或错别字）。只要信用代码存在于持证库中，就会从大盘里剔除。
            **4. 最终产出**：剩下呈现在下方表格中的商户，其经营范围中包含了涉烟关键字，但**社会信用代码未在您的发证系统中登记**，具有极高的无证经营嫌疑。
            """)

        # --- 打击名单 ---
        st.subheader("🚨 疑似无证经营盲区名单 TOP 20（按风险度排序）")
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
        st.download_button("📥 导出最终【有烟无证】稽查名单", buffer, "有烟无证盲区_精准排查名单.xlsx", "application/vnd.ms-excel")

        st.divider()
        draw_analysis_charts(target_pool, title_font, label_font)
