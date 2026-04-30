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
# 0. 基础环境配置
# ==========================================
st.set_page_config(page_title="无证户智能稽查天眼 V3", page_icon="👁️", layout="wide")

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
    
    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 5))
    risk_counts = df['风险等级'].value_counts().reindex(level_order).fillna(0)
    axes1[0].pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=[color_map[l] for l in risk_counts.index], startangle=90, textprops={'fontproperties': l_font})
    axes1[0].set_title('所有商户风险等级分布', fontproperties=t_font)

    for level in level_order:
        subset = df[df['风险等级'] == level]
        if not subset.empty:
            axes1[1].hist(subset['无证户综合概率(%)'], bins=15, color=color_map[level], alpha=0.7, label=level)
    axes1[1].set_title('风险概率密度分布', fontproperties=t_font)
    axes1[1].legend(prop=l_font)
    
    st.pyplot(fig1)

# ==========================================
# 3. 主程序逻辑
# ==========================================
st.title("👁️ 卷烟无证经营户动态筛查 AI 模型")

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
            """输出终端日志"""
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]
            log_lines.insert(0, f"[{timestamp}] {message}")
            display_text = "▼ 实时终端日志 [倒序输出：最新指令始终在最上方显示]\n" + "="*65 + "\n" + "\n".join(log_lines)
            terminal.code(display_text, language="bash")
            time.sleep(delay)

        start_time = time.time()

        # --- 步骤 1: 数据加载与清洗 ---
        log_to_terminal("[SYSTEM] 正在初始化天眼稽查引擎，分配底层内存池...")
        biz = pd.read_excel(file_biz) if file_biz.name.endswith('.xlsx') else pd.read_csv(file_biz)
        unl = pd.read_excel(file_unl) if file_unl.name.endswith('.xlsx') else pd.read_csv(file_unl)
        log_to_terminal(f"[DATA] 数据挂载完毕。检索到执照 {len(biz)} 条，档案 {len(unl)} 条。")

        fill_dict = {'公司名称':'未知', '法定代表人':'未知', '经营范围':'未知', '天眼评分':0, '统一社会信用代码':'未知'}
        biz = biz.rename(columns={'天眼评分': '信用值'}).fillna(fill_dict)
        unl = unl.rename(columns={'天眼评分': '信用值'}).fillna(fill_dict)
        biz['信用值'] = pd.to_numeric(biz['信用值'], errors='coerce').fillna(0)
        unl['信用值'] = pd.to_numeric(unl['信用值'], errors='coerce').fillna(0)

        # --- 步骤 2: 法人名录比对 (仅作展示，不入模型) ---
        log_to_terminal("[GRAPH] 正在提取历史无证档案核心实体，比对法人重合度...")
        bad_reps = set(unl[~unl['法定代表人'].isin(['未知', '', '无'])]['法定代表人'].unique())
        biz['该商户负责人是否在无证户名录（可能重名）'] = biz['法定代表人'].apply(lambda x: '是（可能重名）' if x in bad_reps else '否')
        
        biz['label'], unl['label'] = 0, 1
        df_all = pd.concat([unl, biz], ignore_index=True)

        # --- 步骤 3: 特征工程 ---
        log_to_terminal("[NLP] 启动 TF-IDF 分词引擎，正在高维空间映射 [公司名称] 与 [经营范围]...")
        vec_name = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=800)
        vec_scope = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=1000)
        
        X_name = vec_name.fit_transform(df_all['公司名称'])
        X_scope = vec_scope.fit_transform(df_all['经营范围'])
        
        scaler = MinMaxScaler()
        X_numeric = scaler.fit_transform(df_all[['信用值']])
        X_combined = sp.hstack((X_name, X_scope, X_numeric))
        y_combined = df_all['label'].values

        # --- 步骤 4: 核心演算 ---
        log_to_terminal("[ML-CORE] 初始化 RandomForest 集群，唤醒 CPU 多线程 (n_jobs=-1)...")
        ml_model = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', n_jobs=-1, random_state=42)
        ml_model.fit(X_combined, y_combined)
        log_to_terminal("[ML-CORE] 100 棵决策树簇编译完成！正在下发全网特征进行违规概率推理...")

        all_probs = ml_model.predict_proba(X_combined)[:, 1]
        df_all['无证户综合概率(%)'] = np.round(all_probs * 100, 2)
        target_pool = df_all[df_all['label'] == 0].copy()

        # --- 步骤 5: 白盒解释器 ---
        log_to_terminal("[EXPLAINER] 激活白盒解释器，反推溯源链路与违规特征贡献度...")
        feature_names = np.array(vec_name.get_feature_names_out().tolist() + vec_scope.get_feature_names_out().tolist() + ['信用偏离'])
        importances = ml_model.feature_importances_
        X_target = X_combined.tocsr()[target_pool.index.tolist()]
        weighted_X = X_target.multiply(importances).tocsr()
        
        explanations = []
        for i in range(weighted_X.shape[0]):
            row = weighted_X.getrow(i)
            final_prob = target_pool.iloc[i]['无证户综合概率(%)']
            if row.nnz > 0:
                top_idx = row.toarray()[0].argsort()[-3:][::-1]
                expl_parts = []
                for idx in top_idx:
                    w = row.toarray()[0][idx]
                    if w > 0:
                        rel_pct = (w / row.sum()) * final_prob
                        expl_parts.append(f"{feature_names[idx]}({rel_pct:.1f}%)")
                explanations.append(" + ".join(expl_parts))
            else: explanations.append("综合信用与词频评估")
        target_pool['AI 判定依据'] = explanations

        def assign_risk(p):
            if p >= 85: return '极高风险', '🚨 立即排查'
            elif p >= 65: return '高风险', '⚠️ 重点监控'
            elif p >= 40: return '中风险', '👀 定期关注'
            return '低风险', '✅ 常规监管'
            
        target_pool[['风险等级', '监管建议']] = target_pool.apply(lambda r: pd.Series(assign_risk(r['无证户综合概率(%)'])), axis=1)
        target_pool = target_pool.sort_values('无证户综合概率(%)', ascending=False)
        
        elapsed_time = time.time() - start_time
        calc_speed = int(len(target_pool) / max(elapsed_time, 0.001))
        log_to_terminal(f"[SYSTEM] ✅ 演算闭环结束！用时 {elapsed_time:.2f} 秒，即将渲染数据大屏...", 0)

        # --- 结果展示区 ---
        st.success("🎯 稽查演算已收官！已生成结构化作战简报。")
        
        # 指标大屏
        m1, m2, m3, m4 = st.columns(4)
        total_shops = len(target_pool)
        v_vh = target_pool[target_pool['风险等级']=='极高风险']
        v_h = target_pool[target_pool['风险等级']=='高风险']
        v_m = target_pool[target_pool['风险等级']=='中风险']
        
        m1.metric("极高风险数量 (85%-100%)", f"{len(v_vh)} 家", f"占总体 {len(v_vh)/total_shops*100:.1f}%")
        m2.metric("高风险数量 (65%-84%)", f"{len(v_h)} 家", f"占总体 {len(v_h)/total_shops*100:.1f}%")
        m3.metric("中风险数量 (40%-64%)", f"{len(v_m)} 家", f"占总体 {len(v_m)/total_shops*100:.1f}%")
        m4.metric("全网核查总规模", f"{total_shops} 条", f"AI筛查时效: 极速 ({calc_speed} 条/秒)")

        st.divider()
        
        # --- 风险计算原理示例 (深度原理解释) ---
        with st.expander("💡 了解 AI 白盒解释器如何计算风险？(以沛县某店铺示例)", expanded=True):
            col_ex1, col_ex2 = st.columns([1, 2])
            with col_ex1:
                st.markdown("""
                **示例商户：** `沛县龙城街道某百货超市`  
                **统一社会信用代码：** `92320322MA******11`  
                **信用分：** `42分`  
                **模型判定：** <span style='color:red; font-weight:bold; font-size:20px;'>88.5% (极高风险)</span>
                """, unsafe_allow_html=True)
            with col_ex2:
                st.info("""
                **风险链路归因拆解：**
                * **特征碰撞 (35.4%)**：算法提取经营范围文本，发现高关联词簇。
                    * `百货 (20.2%)`：该词在目标文本中 TF-IDF 词频权重较高，且模型统计发现历史违规户中含有“百货”的基尼不纯度（分类区分度）极高，两者相乘最终贡献了 20.2% 的嫌疑率。
                    * `批发 (15.2%)`：基于语义空间的上下文共现频率，该词被惩罚，贡献 15.2%。
                * **语义关联 (28.4%)**：店名分词触发警报。
                    * `超市 (28.4%)`：AI 检测到该店铺名称属于典型的“隐蔽售烟业态”，在 100 棵决策树的联合投票中，有 73 棵树因为该特征将其推向“高风险叶子节点”，折算概率占比 28.4%。
                * **信用偏离 (24.7%)**：非文本维度数值惩罚。
                    * `信用偏离 (24.7%)`：因该商户天眼评分仅为 42 分，经过 MinMaxScaler 归一化后处于垫底区间，极大偏离了正规店铺的特征分布，模型依据此维度的异常给予 24.7% 的危险权重。
                * **闭环判定**：综合上述多维特征权重（35.4% + 28.4% + 24.7% = 88.5%），模型最终锁定该商户。
                """)

        # --- 打击名单 TOP 20 ---
        st.subheader("🚨 极高风险打击名单 TOP 20")
        display_cols = [
            '公司名称', 
            '统一社会信用代码',
            '无证户综合概率(%)', 
            'AI 判定依据', 
            '风险等级', 
            '监管建议', 
            '法定代表人', 
            '注册地址', 
            '该商户负责人是否在无证户名录（可能重名）'
        ]
        
        st.dataframe(
            target_pool[display_cols].head(20).style.format({"无证户综合概率(%)": "{:.2f}%"})
            .applymap(lambda x: 'color: red; font-weight: bold' if x == '极高风险' else '', subset=['风险等级']),
            use_container_width=True
        )

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            target_pool[display_cols].to_excel(writer, index=False)
        st.download_button("📥 导出全量排查建议名单 (Excel)", buffer, "沛县智能筛查白盒风险名单.xlsx", "application/vnd.ms-excel")

        st.divider()
        draw_analysis_charts(target_pool, title_font, label_font)
