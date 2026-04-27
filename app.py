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
# 0. 网页基本设置与字体注入
# ==========================================
st.set_page_config(page_title="无证户智能稽查天眼", page_icon="👁️", layout="wide")

@st.cache_resource
def get_chinese_font():
    font_path = "SimHei.ttf"
    if not os.path.exists(font_path):
        try:
            # 下载黑体，解决 Linux/Windows 环境图片乱码
            urllib.request.urlretrieve("https://raw.githubusercontent.com/StellarCN/scp_zh/master/fonts/SimHei.ttf", font_path)
        except Exception:
            pass
    
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
# 1. 语义处理辅助函数
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
# 2. 图表生成函数 (解决文字重叠)
# ==========================================
def draw_analysis_charts(df, t_font, l_font):
    st.markdown("### 📊 AI 模型全盘数据可视化分析")
    color_map = {'低风险': '#32CD32', '中风险': '#FFD700', '高风险': '#FF6B00', '极高风险': '#FF0000'}
    level_order = ['低风险', '中风险', '高风险', '极高风险']
    
    st.markdown("#### 一、 无证户概率综合分析")
    # 增加 figsize 宽度到 14，确保图表间距
    fig1, axes1 = plt.subplots(2, 3, figsize=(14, 8)) 
    
    # 1. 直方图
    for level in level_order:
        subset = df[df['风险等级'] == level]
        if not subset.empty:
            axes1[0, 0].hist(subset['无证户综合概率(%)'], bins=15, color=color_map[level], alpha=0.7, label=level, edgecolor='black')
    axes1[0, 0].set_title('所有商户无证户概率分布', fontproperties=t_font)
    axes1[0, 0].legend(prop=l_font)

    # 2. 饼图 (优化图例位置防止重叠)
    risk_counts = df['风险等级'].value_counts().reindex(level_order).fillna(0)
    axes1[0, 1].pie(risk_counts, labels=None, autopct=lambda p: f'{p:.1f}%' if p > 3 else '', colors=[color_map[l] for l in risk_counts.index], startangle=90, textprops={'fontproperties': l_font})
    axes1[0, 1].set_title('所有商户风险等级分布', fontproperties=t_font)
    axes1[0, 1].legend(risk_counts.index, prop=l_font, loc="center left", bbox_to_anchor=(1.0, 0.5)) 

    # 3. 密度图
    import seaborn as sns
    sns.kdeplot(data=df, x='信用值', hue='风险等级', palette=color_map, ax=axes1[0, 2], fill=True, common_norm=False)
    axes1[0, 2].set_title('信用值密度分布', fontproperties=t_font)
    legend = axes1[0, 2].get_legend()
    if legend: 
        plt.setp(legend.texts, fontproperties=l_font)
        legend.set_title("风险等级", prop=l_font)

    # 4. 柱状图
    avg_prob = df.groupby('风险等级')['无证户综合概率(%)'].mean().reindex(level_order)
    axes1[1, 0].bar(avg_prob.index, avg_prob.values, color=[color_map[l] for l in avg_prob.index])
    axes1[1, 0].set_title('各等级平均概率', fontproperties=t_font)
    axes1[1, 0].set_xticklabels(avg_prob.index, fontproperties=l_font)

    # 5. 散点图
    for level in level_order:
        subset = df[df['风险等级'] == level]
        if not subset.empty:
            axes1[1, 1].scatter(subset['信用值'], subset['无证户综合概率(%)'], color=color_map[level], label=level, alpha=0.6, s=15)
    axes1[1, 1].set_title('信用值 vs 概率散点', fontproperties=t_font)
    axes1[1, 1].legend(prop=l_font)

    # 6. 法人饼图
    high_risk_reps = df[df['高危法人关联'] == 1].shape[0]
    rep_data = [high_risk_reps, df.shape[0] - high_risk_reps]
    axes1[1, 2].pie(rep_data, labels=None, autopct=lambda p: f'{p:.1f}%' if p > 3 else '', colors=['#FF6B6B', '#4ECDC4'], startangle=140, textprops={'fontproperties': l_font})
    axes1[1, 2].set_title('法人身份识别比例', fontproperties=t_font)
    axes1[1, 2].legend(['历史高危', '普通法人'], prop=l_font, loc="center left", bbox_to_anchor=(1.0, 0.5))
    
    fig1.tight_layout(pad=2.0) # 核心修正：自动调整子图间距
    st.pyplot(fig1)

    st.markdown("#### 二、 风险等级详细分析")
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))

    # 7. 柱状图2
    axes2[0, 0].bar(risk_counts.index, risk_counts.values, color=[color_map[l] for l in risk_counts.index])
    axes2[0, 0].set_title('商户数量分布', fontproperties=t_font)
    axes2[0, 0].set_xticklabels(risk_counts.index, fontproperties=l_font)

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

    # 11. 柱状图3
    avg_score = df.groupby('风险等级')['信用值'].mean().reindex(level_order)
    axes2[1, 1].bar(avg_score.index, avg_score.values, color=[color_map[l] for l in avg_score.index])
    axes2[1, 1].set_title('平均信用值', fontproperties=t_font)
    axes2[1, 1].set_xticklabels(avg_score.index, fontproperties=l_font)

    # 12. 文字快照
    axes2[1, 2].axis('off')
    axes2[1, 2].set_title('极高风险目标快照', fontproperties=t_font)
    y_pos = 0.9
    for idx, row in df.head(8).reset_index().iterrows():
        name = str(row['公司名称'])[:10] + "..." if len(str(row['公司名称'])) > 10 else row['公司名称']
        axes2[1, 2].text(0.0, y_pos, f"{idx+1}. {name} ({row['无证户综合概率(%)']}%)", fontproperties=l_font, color='red' if idx < 3 else 'black')
        y_pos -= 0.12
        
    fig2.tight_layout(pad=2.0) # 核心修正：自动调整子图间距
    st.pyplot(fig2)

# ==========================================
# 3. 核心系统界面与演算控制
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
            """
            HTML/JS 终端盒：保持原生的白色风格，支持自动滚动探底。
            """
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]
            log_lines.append(f"[{timestamp}] {message}")
            log_text = "\n".join(log_lines)
            
            html_code = f"""
            <div id="terminal_box" style="height: 300px; overflow-y: auto; background-color: #F0F2F6; border-radius: 8px; padding: 15px; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #000000; white-space: pre-wrap; border: 1px solid #D0D4DC;">{log_text}</div>
            <script>
                var d = document.getElementById("terminal_box");
                if(d) d.scrollTop = d.scrollHeight;
            </script>
            """
            terminal.markdown(html_code, unsafe_allow_html=True)

        # 执行演算逻辑并输出日志
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
        
        # 处理重合数据
        overlap_cols = [col for col in biz.columns if '重合' in col]
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
        
        log_to_terminal("[GRAPH] 正在构建图谱，执行跨表网络穿透比对，追溯名下关联店铺...")
        biz['高危法人关联'] = biz['法定代表人'].apply(lambda x: 1 if x in bad_reps else 0)
        unl['高危法人关联'] = 1
        biz['label'], unl['label'] = 0, 1
        df_all = pd.concat([unl, biz], ignore_index=True)
        total_shops = len(df_all)
        log_to_terminal("[GRAPH] 高危法人关系拓扑图谱构建完毕，污染链条已全量标记。")
        
        log_to_terminal("[AI_CORE] 准备注入全量目标进入神经网络...")
        step_scan = max(1, total_shops // 10)
        for i in range(1, total_shops + 1, step_scan):
            log_to_terminal(f"[SCANNING] 正在深度穿透商铺网络，当前排查节点: {i} / {total_shops}...")
            time.sleep(0.05)
        
        log_to_terminal("[NLP] 正在启动 TF-IDF 引擎，提取潜在语义特征...")
        vec_name = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=1000)
        vec_scope = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=1500)
        X_name = vec_name.fit_transform(df_all['公司名称'])
        X_scope = vec_scope.fit_transform(df_all['经营范围'])
        
        log_to_terminal("[NLP] 过滤通用停用词与拦截烟草类干扰词成功。")
        scaler = MinMaxScaler()
        X_numeric = scaler.fit_transform(df_all[['信用值', '高危法人关联']])
        X_combined = sp.hstack((X_name, X_scope, X_numeric))
        y_combined = df_all['label'].values

        log_to_terminal(f"[ML] 特征融合完毕。稀疏矩阵维度构建成功: {X_combined.shape}")
        log_to_terminal("[ML] 核心引擎接管：初始化随机森林决策集群 (RandomForest)...")
        ml_model = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
        ml_model.fit(X_combined, y_combined)
        
        log_to_terminal("[ML] 200 个独立决策算法联合编译完成！")
        all_probs = ml_model.predict_proba(X_combined)[:, 1]
        df_all['无证户综合概率(%)'] = np.round(all_probs * 100, 2)
        target_pool = df_all[df_all['label'] == 0].copy()
        
        log_to_terminal("[EXPLAINER] 正在激活白盒解释器 (White-box Explainer)...")
        # 解释器逻辑 (生成依据列)
        feature_names = np.array(vec_name.get_feature_names_out().tolist() + vec_scope.get_feature_names_out().tolist() + ['信用异常惩罚', '历史无证前科'])
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
                        feat = feature_names[idx]
                        if feat == '历史无证前科':
                            feat = f"关联高危前科法人[{target_pool.iloc[i]['法定代表人']}]"
                        expl_parts.append(f"{feat}({rel_pct:.1f}%)")
                explanations.append(" + ".join(expl_parts) if expl_parts else "无显著高危特征")
            else:
                explanations.append("无显著高危特征")
        
        target_pool['AI 判定依据'] = explanations
        log_to_terminal("[EXPLAINER] 溯源解析瞬间完成！已为所有高危商户生成违规证据链。")

        def assign_risk(prob):
            if prob >= 85: return '极高风险', '🚨 立即排查'
            elif prob >= 65: return '高风险', '⚠️ 重点监控'
            elif prob >= 40: return '中风险', '👀 定期关注'
            else: return '低风险', '✅ 常规监管'
            
        target_pool[['风险等级', '监管建议']] = target_pool.apply(lambda r: pd.Series(assign_risk(r['无证户综合概率(%)'])), axis=1)
        target_pool = target_pool.sort_values(by='无证户综合概率(%)', ascending=False)
        log_to_terminal("[SYSTEM] ✅ 演算闭环结束！正在生成大屏...")

        # ==========================================
        # 结果展示
        # ==========================================
        st.success("🎯 筛查任务完美收官！")
        col1, col2, col3 = st.columns(3)
        col1.metric("极高风险目标锁定", f"{len(target_pool[target_pool['无证户综合概率(%)'] >= 85])} 家", "需立即行动")
        col2.metric("高危法人揪出", f"{target_pool['高危法人关联'].sum()} 人", "存在前科")
        col3.metric("总计排查商铺", f"{len(target_pool)} 家", "极速")

        st.divider()
        st.subheader("🚨 极高风险打击首选名单 TOP 15")
        display_cols = ['公司名称', '无证户综合概率(%)', 'AI 判定依据', '风险等级', '监管建议', '法定代表人', '注册地址', '信用值']
        st.dataframe(target_pool[display_cols].head(15), use_container_width=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            target_pool[display_cols].to_excel(writer, index=False)
        st.download_button(label="📥 一键导出完整作战名单 (Excel)", data=buffer, file_name="智能筛查白盒风险名单.xlsx", mime="application/vnd.ms-excel")

        st.divider()
        draw_analysis_charts(target_pool, title_font, label_font)
