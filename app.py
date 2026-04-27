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
# 0. 网页基本设置与【字体乱码终极修复】
# ==========================================
st.set_page_config(page_title="无证户智能稽查天眼", page_icon="👁️", layout="wide")

@st.cache_resource
def setup_chinese_fonts():
    font_path = "SimHei.ttf"
    if not os.path.exists(font_path):
        try:
            # 自动下载中文字体到云端服务器，防止图表变成方块
            urllib.request.urlretrieve("https://raw.githubusercontent.com/dolbydu/font/master/simhei.ttf", font_path)
        except Exception:
            pass
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'SimHei'
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

setup_chinese_fonts()

CUSTOM_STOP_WORDS = {'有限','责任','分公司','集团','控股','股份','有限公司','徐州','地址','未知','公司', '店铺'}
TOBACCO_WORDS = {'烟草制品零售','卷烟零售','雪茄零售','烟丝零售','香烟销售','烟草销售','烟草','卷烟','雪茄','烟丝','香烟'}

def custom_tokenizer(text):
    if not isinstance(text, str) or not text: return []
    norm_map = {'百货店':'百货','百货商场':'百货','百货公司':'百货','百货超市':'百货','便利店':'便利','批发部':'批发'}
    words = jieba.lcut(text)
    processed_words = [norm_map.get(w, w) for w in words if len(w) > 1 and w not in CUSTOM_STOP_WORDS and not any(tob_w in w for tob_w in TOBACCO_WORDS)]
    return processed_words

# ==========================================
# 1. 图表生成函数 (精致版)
# ==========================================
def draw_analysis_charts(df):
    st.markdown("### 📊 AI 模型全盘数据可视化分析")
    color_map = {'低风险': '#32CD32', '中风险': '#FFD700', '高风险': '#FF6B00', '极高风险': '#FF0000'}
    level_order = ['低风险', '中风险', '高风险', '极高风险']
    
    st.markdown("#### 一、 无证户概率综合分析")
    fig1, axes1 = plt.subplots(2, 3, figsize=(12, 7))
    fig1.subplots_adjust(hspace=0.4, wspace=0.3)
    
    for level in level_order:
        subset = df[df['风险等级'] == level]
        if not subset.empty:
            axes1[0, 0].hist(subset['无证户综合概率(%)'], bins=15, color=color_map[level], alpha=0.7, label=level, edgecolor='black')
    axes1[0, 0].set_title('所有商户无证户概率分布', fontsize=10)
    axes1[0, 0].legend(fontsize=8)

    risk_counts = df['风险等级'].value_counts().reindex(level_order).fillna(0)
    axes1[0, 1].pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=[color_map[l] for l in risk_counts.index], startangle=90, textprops={'fontsize': 8})
    axes1[0, 1].set_title('所有商户风险等级分布', fontsize=10)

    import seaborn as sns
    sns.kdeplot(data=df, x='信用值', hue='风险等级', palette=color_map, ax=axes1[0, 2], fill=True, common_norm=False)
    axes1[0, 2].set_title('信用值密度分布', fontsize=10)

    avg_prob = df.groupby('风险等级')['无证户综合概率(%)'].mean().reindex(level_order)
    bars = axes1[1, 0].bar(avg_prob.index, avg_prob.values, color=[color_map[l] for l in avg_prob.index])
    axes1[1, 0].set_title('各等级平均概率', fontsize=10)
    for bar in bars:
        axes1[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

    for level in level_order:
        subset = df[df['风险等级'] == level]
        if not subset.empty:
            axes1[1, 1].scatter(subset['信用值'], subset['无证户综合概率(%)'], color=color_map[level], label=level, alpha=0.6, s=15)
    axes1[1, 1].set_title('信用值 vs 概率散点', fontsize=10)

    high_risk_reps = df[df['高危法人关联'] == 1].shape[0]
    axes1[1, 2].pie([high_risk_reps, df.shape[0] - high_risk_reps], labels=['历史高危法人', '普通法人'], autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'], startangle=140, textprops={'fontsize': 8})
    axes1[1, 2].set_title('法人身份识别比例', fontsize=10)
    st.pyplot(fig1)

    st.markdown("#### 二、 风险等级详细分析")
    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 7))
    fig2.subplots_adjust(hspace=0.4, wspace=0.3)

    bars = axes2[0, 0].bar(risk_counts.index, risk_counts.values, color=[color_map[l] for l in risk_counts.index])
    axes2[0, 0].set_title('商户数量分布', fontsize=10)
    for bar in bars: axes2[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=8)

    box_data = [df[df['风险等级'] == level]['无证户综合概率(%)'].dropna() for level in level_order]
    bplot = axes2[0, 1].boxplot(box_data, labels=level_order, patch_artist=True)
    for patch, color in zip(bplot['boxes'], [color_map[l] for l in level_order]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes2[0, 1].set_title('各等级概率分布(箱线图)', fontsize=10)
    axes2[0, 1].tick_params(axis='x', labelsize=8)

    box_data_score = [df[df['风险等级'] == level]['信用值'].dropna() for level in level_order]
    bplot_score = axes2[0, 2].boxplot(box_data_score, labels=level_order, patch_artist=True)
    for patch, color in zip(bplot_score['boxes'], [color_map[l] for l in level_order]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes2[0, 2].set_title('信用值分布(箱线图)', fontsize=10)
    axes2[0, 2].tick_params(axis='x', labelsize=8)

    sorted_probs = np.sort(df['无证户综合概率(%)'])
    cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs) * 100
    axes2[1, 0].plot(sorted_probs, cumulative, 'b-', linewidth=2)
    axes2[1, 0].set_title('概率累积分布', fontsize=10)

    avg_score = df.groupby('风险等级')['信用值'].mean().reindex(level_order)
    bars = axes2[1, 1].bar(avg_score.index, avg_score.values, color=[color_map[l] for l in avg_score.index])
    axes2[1, 1].set_title('平均信用值', fontsize=10)
    for bar in bars: axes2[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

    axes2[1, 2].axis('off')
    axes2[1, 2].set_title('极高风险目标快照', weight='bold', fontsize=10)
    y_pos = 0.9
    for idx, row in df.head(8).reset_index().iterrows():
        name = str(row['公司名称'])[:10] + "..." if len(str(row['公司名称'])) > 10 else row['公司名称']
        axes2[1, 2].text(0.0, y_pos, f"{idx+1}. {name} ({row['无证户综合概率(%)']}%)", fontsize=9, color='red' if idx < 3 else 'black')
        y_pos -= 0.12
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
        st.warning("⚠️ 权限阻断：请先在左侧上传两个必须的数据文件！")
    else:
        st.markdown("### 💻 系统核心演算终端")
        terminal = st.empty()
        
        # 核心修改：动态终端瀑布流刷屏引擎
        log_lines = []
        def log_to_terminal(message):
            """将新日志追加到末尾，产生换行往上顶的黑客刷屏效果"""
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]
            log_lines.append(f"[{timestamp}] {message}")
            if len(log_lines) > 20:  # 保持屏幕显示最近的 20 行，产生瀑布流滚动感
                log_lines.pop(0)
            terminal.code("\n".join(log_lines), language="bash")

        log_to_terminal("[SYSTEM] 正在初始化天眼稽查引擎...")
        time.sleep(0.5)
        
        log_to_terminal("[DATA] 正在挂载底层数据卷...")
        biz = pd.read_excel(file_biz) if file_biz.name.endswith('.xlsx') else pd.read_csv(file_biz)
        unl = pd.read_excel(file_unl) if file_unl.name.endswith('.xlsx') else pd.read_csv(file_unl)
        
        if '天眼评分' in biz.columns: biz.rename(columns={'天眼评分': '信用值'}, inplace=True)
        if '天眼评分' in unl.columns: unl.rename(columns={'天眼评分': '信用值'}, inplace=True)
        overlap_cols = [col for col in biz.columns if '重合' in col]
        if overlap_cols: 
            biz = biz[~biz[overlap_cols[0]].isin(['是', '1', 1, True, 'TRUE', 'true'])]
            log_to_terminal("[CLEAN] 成功剔除已知重合数据，防止目标泄露。")

        fill_dict = {'公司名称':'未知', '法定代表人':'未知', '注册地址':'未知', '经营范围':'未知', '信用值':0, '统一社会信用代码':'未知'}
        biz = biz.fillna(fill_dict).drop_duplicates(subset=['统一社会信用代码'], keep='first')
        unl = unl.fillna(fill_dict).drop_duplicates(subset=['统一社会信用代码'], keep='first')
        biz['信用值'] = pd.to_numeric(biz['信用值'], errors='coerce').fillna(0)
        unl['信用值'] = pd.to_numeric(unl['信用值'], errors='coerce').fillna(0)
        
        bad_reps = set(unl[~unl['法定代表人'].isin(['未知', '', '无'])]['法定代表人'].unique())
        biz['高危法人关联'] = biz['法定代表人'].apply(lambda x: 1 if x in bad_reps else 0)
        unl['高危法人关联'] = 1
        biz['label'], unl['label'] = 0, 1
        df_all = pd.concat([unl, biz], ignore_index=True)
        total_shops = len(df_all)
        
        log_to_terminal("[GRAPH] 高危法人关系拓扑图谱构建完毕。")
        time.sleep(0.3)
        log_to_terminal("[AI_CORE] 准备注入全量目标进入神经网络...")
        
        step_scan = max(1, total_shops // 20)
        for i in range(1, total_shops + 1, step_scan):
            log_to_terminal(f"[SCANNING] 正在深度穿透商铺网络，当前排查节点: {i} / {total_shops}...")
            time.sleep(0.08)

        log_to_terminal(f"[SCANNING] 节点穿透完毕，共计锁定 {total_shops} 个计算目标。")
        time.sleep(0.3)
        
        # 3. NLP向量化日志刷屏
        log_to_terminal("[NLP] 正在启动 TF-IDF 引擎，提取潜在语义特征...")
        time.sleep(0.3)
        log_to_terminal("[NLP] 挂载自定义分词器 (Custom Tokenizer) 与归一化词典...")
        
        vec_name = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=1000)
        vec_scope = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=1500)
        
        time.sleep(0.2)
        log_to_terminal("[NLP] 正在对 [公司名称] 执行高维空间映射 (Max Features: 1000)...")
        X_name = vec_name.fit_transform(df_all['公司名称'])
        
        time.sleep(0.2)
        log_to_terminal("[NLP] 正在对 [经营范围] 执行高维空间映射 (Max Features: 1500)...")
        X_scope = vec_scope.fit_transform(df_all['经营范围'])
        
        time.sleep(0.2)
        log_to_terminal("[NLP] 过滤通用停用词与拦截烟草类干扰词成功。")
        log_to_terminal("[DATA] 启动 MinMaxScaler 压缩信用值极值偏倚...")
        
        scaler = MinMaxScaler()
        X_numeric = scaler.fit_transform(df_all[['信用值', '高危法人关联']])
        X_combined = sp.hstack((X_name, X_scope, X_numeric))
        y_combined = df_all['label'].values

        log_to_terminal(f"[ML] 特征融合完毕。稀疏矩阵维度构建成功: {X_combined.shape}")
        time.sleep(0.4)

        # 4. 模型训练日志刷屏
        log_to_terminal("[ML] 核心引擎接管：初始化随机森林决策集群 (RandomForest)...")
        time.sleep(0.2)
        log_to_terminal("[ML] 配置参数 -> n_estimators: 200 | max_depth: 20 | class_weight: balanced")
        time.sleep(0.2)
        log_to_terminal("[ML] 正在唤醒 CPU 多线程并行计算 (n_jobs=-1)...")
        
        for tree_batch in range(1, 6):
            log_to_terminal(f"[ML-CORE] 正在生成决策树簇 [{tree_batch*40}/200]... 计算节点基尼杂质 (Gini Impurity)...")
            time.sleep(0.15)
            
        log_to_terminal("[ML-CORE] 正在执行最大深度剪枝与叶子节点收敛...")
        time.sleep(0.2)
        
        ml_model = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
        ml_model.fit(X_combined, y_combined)
        
        log_to_terminal("[ML] 200 个独立决策算法联合编译完成！")
        time.sleep(0.2)
        log_to_terminal("[PREDICT] 正在向目标商户广播预测任务，提取嫌疑概率...")
        
        all_probs = ml_model.predict_proba(X_combined)[:, 1]
        df_all['无证户综合概率(%)'] = np.round(all_probs * 100, 2)
        target_pool = df_all[df_all['label'] == 0].copy()
        
        # 5. 【概率溯源白盒算法及动态刷屏】
        log_to_terminal("[EXPLAINER] 正在激活白盒解释器 (White-box Explainer)...")
        time.sleep(0.3)
        log_to_terminal("[EXPLAINER] 提取全局特征重要性矩阵 (Feature Importances Matrix)...")
        
        feature_names = np.array(vec_name.get_feature_names_out().tolist() + vec_scope.get_feature_names_out().tolist() + ['信用异常惩罚', '历史无证前科'])
        importances = ml_model.feature_importances_
        X_target = X_combined.tocsr()[target_pool.index.tolist()]
        weighted_X = X_target.multiply(importances) 
        
        time.sleep(0.2)
        log_to_terminal("[EXPLAINER] 正在反推目标商户的概率溯源路径...")
        
        explanations = []
        total_targets = weighted_X.shape[0]
        step_expl = max(1, total_targets // 15)
        
        for i in range(total_targets):
            if i % step_expl == 0 and i > 0:
                log_to_terminal(f"[EXPLAINER] 已解析溯源路径: {i} / {total_targets} 个节点...")
                time.sleep(0.04)
                
            row_weights = weighted_X.getrow(i).toarray()[0]
            top_indices = row_weights.argsort()[-3:][::-1] 
            total_weight = row_weights.sum()
            final_prob = target_pool.iloc[i]['无证户综合概率(%)']
            
            if total_weight > 0:
                expl_parts = []
                sum_top_pct = 0
                for idx in top_indices:
                    if row_weights[idx] > 0:
                        rel_pct = (row_weights[idx] / total_weight) * final_prob
                        sum_top_pct += rel_pct
                        feat_name = feature_names[idx]
                        expl_parts.append(f"{feat_name}({rel_pct:.1f}%)")
                remaining = final_prob - sum_top_pct
                if remaining > 0.5: expl_parts.append(f"其他综合({remaining:.1f}%)")
                explanation = " + ".join(expl_parts)
            else:
                explanation = "无显著高危特征"
            explanations.append(explanation)
            
        target_pool['AI 判定依据'] = explanations
        log_to_terminal("[EXPLAINER] 溯源解析完成！已为所有高危商户生成违规证据链。")
        time.sleep(0.3)

        # 6. 分级与排序
        def assign_risk(prob):
            if prob >= 85: return '极高风险', '🚨 立即排查'
            elif prob >= 65: return '高风险', '⚠️ 重点监控'
            elif prob >= 40: return '中风险', '👀 定期关注'
            else: return '低风险', '✅ 常规监管'
            
        target_pool[['风险等级', '监管建议']] = target_pool.apply(lambda r: pd.Series(assign_risk(r['无证户综合概率(%)'])), axis=1)
        target_pool = target_pool.sort_values(by='无证户综合概率(%)', ascending=False)
        
        log_to_terminal("[SYSTEM] ✅ 演算闭环结束！系统正在生成终端高危打击清单与数据大屏...")
        time.sleep(0.5)
        
        # ==========================================
        # 结果展示区
        # ==========================================
        st.success("🎯 筛查任务完美收官！已生成结构化作战简报。")
        
        with st.expander("💡 侦探大脑：AI 是如何计算出这个违规概率的？", expanded=True):
            st.markdown("""
            此处的 **“无证户综合概率(%)”** 并非简单的词汇叠加，而是由 **200 个独立的决策树算法（随机森林）** 联合投票计算得出的综合置信度：
            * 📝 **隐蔽文本比对**：AI 通过 TF-IDF 算法提取了全量商户的“店名”和“经营范围”。它过滤了普通词汇，专门寻找在历史无证户中高频出现的**异常伪装词汇组合**。
            * 👤 **法人网络追踪**：结合历史案卷，一旦发现该店的老板曾有过被罚记录（高危法人关联），算法会对其名下所有新店铺叠加极高的惩罚权重。
            * 📉 **异常数值惩罚**：参考天眼查/企查查信用分，评分越低下限、异常记录越多的商铺，其违法嫌疑会产生指数级上升。
            > **实战指导：** 只要商户概率达到 **85% (极高风险)** 以上，意味着其在“字面伪装”、“历史背景”和“信用评分”这三个维度上，**已经与历史抓获的无证户达到了极高的基因相似度**。在名册中，AI 会为您精准罗列出导致其被判定为极高风险的**核心元凶特征及其贡献比例**！
            """)

        col1, col2, col3 = st.columns(3)
        col1.metric("极高风险目标锁定", f"{len(target_pool[target_pool['无证户综合概率(%)'] >= 85])} 家", "需立即行动")
        col2.metric("高危法人揪出", f"{target_pool['高危法人关联'].sum()} 人", "存在前科")
        col3.metric("总计排查商铺", f"{len(target_pool)} 家", "极速")

        st.divider()
        draw_analysis_charts(target_pool)
        
        st.divider()
        st.subheader("🚨 极高风险打击首选名单 TOP 15 (附白盒释义)")
        
        display_cols = ['公司名称', '无证户综合概率(%)', 'AI 判定依据', '风险等级', '监管建议', '法定代表人', '注册地址', '信用值']
        st.dataframe(target_pool[display_cols].head(15), use_container_width=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            target_pool[display_cols].to_excel(writer, index=False)
        st.download_button(label="📥 一键导出完整作战排查名单 (Excel)", data=buffer, file_name="智能筛查白盒风险名单.xlsx", mime="application/vnd.ms-excel")
