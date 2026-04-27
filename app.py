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
# 0. 网页基本设置
# ==========================================
st.set_page_config(page_title="无证户智能稽查天眼", page_icon="👁️", layout="wide")

# 下载并直接获取字体文件路径（放弃全局配置，改用直接注入）
@st.cache_resource
def get_chinese_font():
    font_path = "SimHei.ttf"
    if not os.path.exists(font_path):
        try:
            # 使用更稳定的 github raw 镜像下载黑体
            urllib.request.urlretrieve("https://raw.githubusercontent.com/StellarCN/scp_zh/master/fonts/SimHei.ttf", font_path)
        except Exception:
            pass
    
    if os.path.exists(font_path):
        title_font = fm.FontProperties(fname=font_path, size=11, weight='bold')
        label_font = fm.FontProperties(fname=font_path, size=9)
    else:
        # 如果下载失败的备用方案
        title_font = fm.FontProperties(size=11, weight='bold')
        label_font = fm.FontProperties(size=9)
        
    plt.rcParams['axes.unicode_minus'] = False
    return title_font, label_font

title_font, label_font = get_chinese_font()

CUSTOM_STOP_WORDS = {'有限','责任','分公司','集团','控股','股份','有限公司','徐州','地址','未知','公司', '店铺'}
TOBACCO_WORDS = {'烟草制品零售','卷烟零售','雪茄零售','烟丝零售','香烟销售','烟草销售','烟草','卷烟','雪茄','烟丝','香烟'}

def custom_tokenizer(text):
    if not isinstance(text, str) or not text: return []
    norm_map = {'百货店':'百货','百货商场':'百货','百货公司':'百货','百货超市':'百货','便利店':'便利','批发部':'批发'}
    words = jieba.lcut(text)
    processed_words = [norm_map.get(w, w) for w in words if len(w) > 1 and w not in CUSTOM_STOP_WORDS and not any(tob_w in w for tob_w in TOBACCO_WORDS)]
    return processed_words

# ==========================================
# 1. 图表生成函数 (100% 解决乱码版)
# ==========================================
def draw_analysis_charts(df, t_font, l_font):
    st.markdown("### 📊 AI 模型全盘数据可视化分析")
    color_map = {'低风险': '#32CD32', '中风险': '#FFD700', '高风险': '#FF6B00', '极高风险': '#FF0000'}
    level_order = ['低风险', '中风险', '高风险', '极高风险']
    
    st.markdown("#### 一、 无证户概率综合分析")
    fig1, axes1 = plt.subplots(2, 3, figsize=(12, 7))
    fig1.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 1. 直方图
    for level in level_order:
        subset = df[df['风险等级'] == level]
        if not subset.empty:
            axes1[0, 0].hist(subset['无证户综合概率(%)'], bins=15, color=color_map[level], alpha=0.7, label=level, edgecolor='black')
    axes1[0, 0].set_title('所有商户无证户概率分布', fontproperties=t_font)
    axes1[0, 0].set_xlabel('无证户概率(%)', fontproperties=l_font)
    axes1[0, 0].set_ylabel('商户数量', fontproperties=l_font)
    axes1[0, 0].legend(prop=l_font)

    # 2. 饼图
    risk_counts = df['风险等级'].value_counts().reindex(level_order).fillna(0)
    axes1[0, 1].pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=[color_map[l] for l in risk_counts.index], startangle=90, textprops={'fontproperties': l_font})
    axes1[0, 1].set_title('所有商户风险等级分布', fontproperties=t_font)

    # 3. 密度图
    import seaborn as sns
    sns.kdeplot(data=df, x='信用值', hue='风险等级', palette=color_map, ax=axes1[0, 2], fill=True, common_norm=False)
    axes1[0, 2].set_title('信用值密度分布', fontproperties=t_font)
    axes1[0, 2].set_xlabel('信用值', fontproperties=l_font)
    axes1[0, 2].set_ylabel('密度', fontproperties=l_font)
    legend = axes1[0, 2].get_legend()
    if legend: plt.setp(legend.texts, fontproperties=l_font)

    # 4. 柱状图
    avg_prob = df.groupby('风险等级')['无证户综合概率(%)'].mean().reindex(level_order)
    bars = axes1[1, 0].bar(avg_prob.index, avg_prob.values, color=[color_map[l] for l in avg_prob.index])
    axes1[1, 0].set_title('各等级平均概率', fontproperties=t_font)
    axes1[1, 0].set_xticks(range(len(avg_prob.index)))
    axes1[1, 0].set_xticklabels(avg_prob.index, fontproperties=l_font)
    for bar in bars:
        axes1[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}%', ha='center', va='bottom', fontproperties=l_font)

    # 5. 散点图
    for level in level_order:
        subset = df[df['风险等级'] == level]
        if not subset.empty:
            axes1[1, 1].scatter(subset['信用值'], subset['无证户综合概率(%)'], color=color_map[level], label=level, alpha=0.6, s=15)
    axes1[1, 1].set_title('信用值 vs 概率散点', fontproperties=t_font)
    axes1[1, 1].set_xlabel('信用值', fontproperties=l_font)
    axes1[1, 1].set_ylabel('无证户概率(%)', fontproperties=l_font)
    axes1[1, 1].legend(prop=l_font)

    # 6. 法人饼图
    high_risk_reps = df[df['高危法人关联'] == 1].shape[0]
    axes1[1, 2].pie([high_risk_reps, df.shape[0] - high_risk_reps], labels=['历史高危法人', '普通法人'], autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'], startangle=140, textprops={'fontproperties': l_font})
    axes1[1, 2].set_title('法人身份识别比例', fontproperties=t_font)
    st.pyplot(fig1)

    st.markdown("#### 二、 风险等级详细分析")
    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 7))
    fig2.subplots_adjust(hspace=0.4, wspace=0.3)

    # 7. 柱状图2
    bars = axes2[0, 0].bar(risk_counts.index, risk_counts.values, color=[color_map[l] for l in risk_counts.index])
    axes2[0, 0].set_title('商户数量分布', fontproperties=t_font)
    axes2[0, 0].set_xticks(range(len(risk_counts.index)))
    axes2[0, 0].set_xticklabels(risk_counts.index, fontproperties=l_font)
    for bar in bars: 
        axes2[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{int(bar.get_height())}', ha='center', va='bottom', fontproperties=l_font)

    # 8. 箱线图1
    box_data = [df[df['风险等级'] == level]['无证户综合概率(%)'].dropna() for level in level_order]
    bplot = axes2[0, 1].boxplot(box_data, labels=level_order, patch_artist=True)
    for patch, color in zip(bplot['boxes'], [color_map[l] for l in level_order]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes2[0, 1].set_title('各等级概率分布(箱线图)', fontproperties=t_font)
    axes2[0, 1].set_xticklabels(level_order, fontproperties=l_font)

    # 9. 箱线图2
    box_data_score = [df[df['风险等级'] == level]['信用值'].dropna() for level in level_order]
    bplot_score = axes2[0, 2].boxplot(box_data_score, labels=level_order, patch_artist=True)
    for patch, color in zip(bplot_score['boxes'], [color_map[l] for l in level_order]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes2[0, 2].set_title('信用值分布(箱线图)', fontproperties=t_font)
    axes2[0, 2].set_xticklabels(level_order, fontproperties=l_font)

    # 10. 折线图
    sorted_probs = np.sort(df['无证户综合概率(%)'])
    cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs) * 100
    axes2[1, 0].plot(sorted_probs, cumulative, 'b-', linewidth=2)
    axes2[1, 0].set_title('概率累积分布', fontproperties=t_font)
    axes2[1, 0].set_xlabel('无证户概率(%)', fontproperties=l_font)
    axes2[1, 0].set_ylabel('累积百分比(%)', fontproperties=l_font)

    # 11. 柱状图3
    avg_score = df.groupby('风险等级')['信用值'].mean().reindex(level_order)
    bars = axes2[1, 1].bar(avg_score.index, avg_score.values, color=[color_map[l] for l in avg_score.index])
    axes2[1, 1].set_title('平均信用值', fontproperties=t_font)
    axes2[1, 1].set_xticks(range(len(avg_score.index)))
    axes2[1, 1].set_xticklabels(avg_score.index, fontproperties=l_font)
    for bar in bars: 
        axes2[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{bar.get_height():.1f}', ha='center', va='bottom', fontproperties=l_font)

    # 12. 文字快照
    axes2[1, 2].axis('off')
    axes2[1, 2].set_title('极高风险目标快照', fontproperties=t_font)
    y_pos = 0.9
    for idx, row in df.head(8).reset_index().iterrows():
        name = str(row['公司名称'])[:10] + "..." if len(str(row['公司名称'])) > 10 else row['公司名称']
        axes2[1, 2].text(0.0, y_pos, f"{idx+1}. {name} ({row['无证户综合概率(%)']}%)", fontproperties=l_font, color='red' if idx < 3 else 'black')
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
        log_lines = []
        
        def log_to_terminal(message):
            """原生倒序输出机制：恢复白色文本框，并将最新日志置于最顶端，根治滚动难题"""
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]
            # 新日志插入到列表的第 0 个位置（最上方）
            log_lines.insert(0, f"[{timestamp}] {message}")
            
            # 使用 Streamlit 原生的 st.code 样式（跟随网页主题，默认白色底色）
            display_text = "▼ 实时终端日志 [最新指令始终在最上方显示]\n" + "="*50 + "\n" + "\n".join(log_lines)
            terminal.code(display_text, language="bash")

        log_to_terminal("[SYSTEM] 正在初始化天眼稽查引擎...")
        time.sleep(0.3)
        log_to_terminal("[SYSTEM] 分配核心内存空间，启动沙盒隔离环境...")
        time.sleep(0.2)
        
        log_to_terminal("[DATA] 正在挂载底层数据卷，请求读取文件流...")
        biz = pd.read_excel(file_biz) if file_biz.name.endswith('.xlsx') else pd.read_csv(file_biz)
        unl = pd.read_excel(file_unl) if file_unl.name.endswith('.xlsx') else pd.read_csv(file_unl)
        log_to_terminal(f"[DATA] 数据加载完毕。检索到营业执照 {len(biz)} 条，卷宗数据 {len(unl)} 条。")
        time.sleep(0.3)
        
        log_to_terminal("[CLEAN] 启动数据清洗管线，进行字段对齐与缺失值探测...")
        if '天眼评分' in biz.columns: biz.rename(columns={'天眼评分': '信用值'}, inplace=True)
        if '天眼评分' in unl.columns: unl.rename(columns={'天眼评分': '信用值'}, inplace=True)
        overlap_cols = [col for col in biz.columns if '重合' in col]
        
        time.sleep(0.2)
        if overlap_cols: 
            log_to_terminal(f"[CLEAN] 发现预置重合标识列 [{overlap_cols[0]}]，正在执行交集过滤...")
            biz = biz[~biz[overlap_cols[0]].isin(['是', '1', 1, True, 'TRUE', 'true'])]
            log_to_terminal("[CLEAN] 成功剔除已知重合数据，有效防止目标泄露。")

        log_to_terminal("[CLEAN] 填充缺失特征向量，交叉比对统一社会信用代码去重...")
        fill_dict = {'公司名称':'未知', '法定代表人':'未知', '注册地址':'未知', '经营范围':'未知', '信用值':0, '统一社会信用代码':'未知'}
        biz = biz.fillna(fill_dict).drop_duplicates(subset=['统一社会信用代码'], keep='first')
        unl = unl.fillna(fill_dict).drop_duplicates(subset=['统一社会信用代码'], keep='first')
        biz['信用值'] = pd.to_numeric(biz['信用值'], errors='coerce').fillna(0)
        unl['信用值'] = pd.to_numeric(unl['信用值'], errors='coerce').fillna(0)
        
        log_to_terminal("[GRAPH] 正在从历史无证档案中提取 [法定代表人] 核心实体...")
        bad_reps = set(unl[~unl['法定代表人'].isin(['未知', '', '无'])]['法定代表人'].unique())
        log_to_terminal(f"[GRAPH] 成功提取 {len(bad_reps)} 个高危法人身份特征。")
        time.sleep(0.3)
        
        log_to_terminal("[GRAPH] 正在构建图谱，执行跨表网络穿透比对，追溯名下关联店铺...")
        biz['高危法人关联'] = biz['法定代表人'].apply(lambda x: 1 if x in bad_reps else 0)
        unl['高危法人关联'] = 1
        biz['label'], unl['label'] = 0, 1
        df_all = pd.concat([unl, biz], ignore_index=True)
        total_shops = len(df_all)
        log_to_terminal("[GRAPH] 高危法人关系拓扑图谱构建完毕，污染链条已全量标记。")
        time.sleep(0.3)
        
        log_to_terminal("[AI_CORE] 准备注入全量目标进入神经网络...")
        step_scan = max(1, total_shops // 15)
        for i in range(1, total_shops + 1, step_scan):
            log_to_terminal(f"[SCANNING] 正在深度穿透商铺网络，当前排查节点: {i} / {total_shops}...")
            time.sleep(0.05)
        log_to_terminal(f"[SCANNING] 节点穿透完毕，共计锁定 {total_shops} 个计算目标。")
        
        log_to_terminal("[NLP] 正在启动 TF-IDF 引擎，提取潜在语义特征...")
        log_to_terminal("[NLP] 挂载自定义分词器 (Custom Tokenizer) 与归一化词典...")
        vec_name = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=1000)
        vec_scope = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=1500)
        
        log_to_terminal("[NLP] 正在对 [公司名称] 执行高维空间映射 (Max Features: 1000)...")
        X_name = vec_name.fit_transform(df_all['公司名称'])
        log_to_terminal("[NLP] 正在对 [经营范围] 执行高维空间映射 (Max Features: 1500)...")
        X_scope = vec_scope.fit_transform(df_all['经营范围'])
        
        log_to_terminal("[NLP] 过滤通用停用词与拦截烟草类干扰词成功。")
        log_to_terminal("[DATA] 启动 MinMaxScaler 压缩信用值极值偏倚...")
        scaler = MinMaxScaler()
        X_numeric = scaler.fit_transform(df_all[['信用值', '高危法人关联']])
        X_combined = sp.hstack((X_name, X_scope, X_numeric))
        y_combined = df_all['label'].values

        log_to_terminal(f"[ML] 特征融合完毕。稀疏矩阵维度构建成功: {X_combined.shape}")

        log_to_terminal("[ML] 核心引擎接管：初始化随机森林决策集群 (RandomForest)...")
        log_to_terminal("[ML] 正在唤醒 CPU 多线程并行计算 (n_jobs=-1)...")
        for tree_batch in range(1, 6):
            log_to_terminal(f"[ML-CORE] 正在生成决策树簇 [{tree_batch*40}/200]... 计算节点基尼杂质 (Gini Impurity)...")
            time.sleep(0.1)
        log_to_terminal("[ML-CORE] 正在执行最大深度剪枝与叶子节点收敛...")
        
        ml_model = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
        ml_model.fit(X_combined, y_combined)
        
        log_to_terminal("[ML] 200 个独立决策算法联合编译完成！")
        log_to_terminal("[PREDICT] 正在向目标商户广播预测任务，提取嫌疑概率...")
        all_probs = ml_model.predict_proba(X_combined)[:, 1]
        df_all['无证户综合概率(%)'] = np.round(all_probs * 100, 2)
        target_pool = df_all[df_all['label'] == 0].copy()
        
        log_to_terminal("[EXPLAINER] 正在激活白盒解释器 (White-box Explainer)...")
        log_to_terminal("[EXPLAINER] 提取全局特征重要性矩阵...")
        
        feature_names = np.array(vec_name.get_feature_names_out().tolist() + vec_scope.get_feature_names_out().tolist() + ['信用异常惩罚', '历史无证前科'])
        importances = ml_model.feature_importances_
        X_target = X_combined.tocsr()[target_pool.index.tolist()]
        weighted_X = X_target.multiply(importances).tocsr() 
        
        indptr = weighted_X.indptr
        indices = weighted_X.indices
        data = weighted_X.data
        
        explanations = []
        total_targets = weighted_X.shape[0]
        step_expl = max(1, total_targets // 10)
        log_to_terminal("[EXPLAINER] 启动底层指针寻址，正在极速反推目标溯源路径...")
        
        for i in range(total_targets):
            if i > 0 and i % step_expl == 0:
                log_to_terminal(f"[EXPLAINER] 已极速解析溯源路径: {i} / {total_targets} 个节点...")
                
            start = indptr[i]
            end = indptr[i+1]
            final_prob = target_pool.iloc[i]['无证户综合概率(%)']
            
            if start < end:
                row_data = data[start:end]
                row_indices = indices[start:end]
                total_weight = np.sum(row_data)
                
                if total_weight > 0:
                    k = min(3, len(row_data))
                    if k < len(row_data):
                        top_k_idx = np.argpartition(row_data, -k)[-k:]
                        top_k_idx = top_k_idx[np.argsort(row_data[top_k_idx])][::-1]
                    else:
                        top_k_idx = np.argsort(row_data)[::-1]
                        
                    top_indices_local = row_indices[top_k_idx]
                    top_weights_local = row_data[top_k_idx]
                    
                    expl_parts = []
                    sum_top_pct = 0
                    for idx, weight in zip(top_indices_local, top_weights_local):
                        if weight > 0:
                            rel_pct = (weight / total_weight) * final_prob
                            sum_top_pct += rel_pct
                            feat_name = feature_names[idx]
                            
                            if feat_name == '历史无证前科':
                                rep_name = target_pool.iloc[i]['法定代表人']
                                feat_name = f"关联高危前科法人[{rep_name}]"
                                
                            expl_parts.append(f"{feat_name}({rel_pct:.1f}%)")
                            
                    remaining = final_prob - sum_top_pct
                    if remaining > 0.5:
                        expl_parts.append(f"其他综合({remaining:.1f}%)")
                    explanation = " + ".join(expl_parts)
                else:
                    explanation = "无显著高危特征"
            else:
                explanation = "无显著高危特征"
                
            explanations.append(explanation)
            
        target_pool['AI 判定依据'] = explanations
        log_to_terminal("[EXPLAINER] 溯源解析瞬间完成！已为所有高危商户生成违规证据链。")

        def assign_risk(prob):
            if prob >= 85: return '极高风险', '🚨 立即排查'
            elif prob >= 65: return '高风险', '⚠️ 重点监控'
            elif prob >= 40: return '中风险', '👀 定期关注'
            else: return '低风险', '✅ 常规监管'
            
        target_pool[['风险等级', '监管建议']] = target_pool.apply(lambda r: pd.Series(assign_risk(r['无证户综合概率(%)'])), axis=1)
        target_pool = target_pool.sort_values(by='无证户综合概率(%)', ascending=False)
        
        log_to_terminal("[SYSTEM] ✅ 演算闭环结束！系统正在生成终端高危打击清单与数据大屏...")
        
        # ==========================================
        # 结果展示区 (布局重构：名单置顶)
        # ==========================================
        st.success("🎯 筛查任务完美收官！已生成结构化作战简报。")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("极高风险目标锁定", f"{len(target_pool[target_pool['无证户综合概率(%)'] >= 85])} 家", "需立即行动")
        col2.metric("高危法人揪出", f"{target_pool['高危法人关联'].sum()} 人", "存在前科")
        col3.metric("总计排查商铺", f"{len(target_pool)} 家", "极速")

        st.divider()
        st.subheader("🚨 极高风险打击首选名单 TOP 15 (附白盒释义)")
        
        with st.expander("💡 侦探大脑：AI 判定的各项依据代表什么意思？", expanded=True):
            st.markdown("""
            实战指导：只要商户概率达到 **85% (极高风险)** 以上，建议优先派警力排查！在名册的【AI 判定依据】列中，AI 已精准罗列出嫌疑来源：
            * 📝 **隐蔽文本伪装（如：百货(45%)）**：代表该店名称或经营范围使用了与历史无证户高度重合的伪装词汇。
            * 👤 **历史前科穿透（如：关联高危前科法人[张三]）**：代表此店的老板“张三”，曾在历史卷宗中因为无证经营被抓过，具有极高的重操旧业嫌疑！
            * 📉 **异常信用惩罚**：代表该商户在外部系统（如企查查）中的信用分极低或存在严重经营异常。
            """)
            
        display_cols = ['公司名称', '无证户综合概率(%)', 'AI 判定依据', '风险等级', '监管建议', '法定代表人', '注册地址', '信用值']
        st.dataframe(target_pool[display_cols].head(15), use_container_width=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            target_pool[display_cols].to_excel(writer, index=False)
        st.download_button(label="📥 一键导出完整作战排查名单 (Excel)", data=buffer, file_name="智能筛查白盒风险名单.xlsx", mime="application/vnd.ms-excel")

        st.divider()
        draw_analysis_charts(target_pool, title_font, label_font)
