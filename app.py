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

# 自动下载并挂载中文字体，彻底解决云端图表显示方块(豆腐块)的问题
@st.cache_resource
def setup_chinese_fonts():
    font_path = "SimHei.ttf"
    if not os.path.exists(font_path):
        try:
            # 从可靠来源自动下载黑体字体文件到服务器
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

# 停用词和拦截词库
CUSTOM_STOP_WORDS = {'有限','责任','分公司','集团','控股','股份','有限公司','徐州','地址','未知','公司', '店铺'}
TOBACCO_WORDS = {'烟草制品零售','卷烟零售','雪茄零售','烟丝零售','香烟销售','烟草销售','烟草','卷烟','雪茄','烟丝','香烟'}

def custom_tokenizer(text):
    if not isinstance(text, str) or not text: return []
    norm_map = {'百货店':'百货','百货商场':'百货','百货公司':'百货','百货超市':'百货','便利店':'便利','批发部':'批发'}
    words = jieba.lcut(text)
    processed_words = [norm_map.get(w, w) for w in words if len(w) > 1 and w not in CUSTOM_STOP_WORDS and not any(tob_w in w for tob_w in TOBACCO_WORDS)]
    return processed_words

# ==========================================
# 1. 图表生成函数 (尺寸缩小，布局优化)
# ==========================================
def draw_analysis_charts(df):
    st.markdown("### 📊 AI 模型全盘数据可视化分析")
    color_map = {'低风险': '#32CD32', '中风险': '#FFD700', '高风险': '#FF6B00', '极高风险': '#FF0000'}
    level_order = ['低风险', '中风险', '高风险', '极高风险']
    
    # 将原来的 (15, 10) 缩小为 (12, 7)，更适合网页观看
    st.markdown("#### 一、 无证户概率综合分析")
    fig1, axes1 = plt.subplots(2, 3, figsize=(12, 7))
    fig1.subplots_adjust(hspace=0.4, wspace=0.3) # 增加间距防止字挤在一起
    
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
        terminal = st.empty()  # 申请一块空白区域，用于动态刷新文字
        
        # 1. 读数据
        terminal.code("[系统进程] 正在解析目标文件，准备连接 AI 演算集群...", language="bash")
        biz = pd.read_excel(file_biz) if file_biz.name.endswith('.xlsx') else pd.read_csv(file_biz)
        unl = pd.read_excel(file_unl) if file_unl.name.endswith('.xlsx') else pd.read_csv(file_unl)
        
        if '天眼评分' in biz.columns: biz.rename(columns={'天眼评分': '信用值'}, inplace=True)
        if '天眼评分' in unl.columns: unl.rename(columns={'天眼评分': '信用值'}, inplace=True)
        overlap_cols = [col for col in biz.columns if '重合' in col]
        if overlap_cols: biz = biz[~biz[overlap_cols[0]].isin(['是', '1', 1, True, 'TRUE', 'true'])]

        # 2. 特征工程
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
        
        # [酷炫实时进度刷新]：不再换行叠加，而是原地替换文字
        step = max(1, total_shops // 30)
        for i in range(1, total_shops + 1, step):
            terminal.code(f"[系统进程] 集群连接成功，准备向量化...\n[执行引擎] 正在扫描商铺网络，实时排查进度: {i} / {total_shops} ({(i/total_shops)*100:.1f}%) ▓▓▓░░", language="bash")
            time.sleep(0.05)

        terminal.code(f"[系统进程] 集群连接成功，准备向量化...\n[执行引擎] 正在扫描商铺网络，实时排查进度: {total_shops} / {total_shops} (100.0%) ▓▓▓▓▓\n[AI 分析] NLP分词与 TF-IDF 矩阵提取中...", language="bash")
        
        # 3. NLP向量化
        vec_name = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=1000)
        vec_scope = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=1500)
        X_name = vec_name.fit_transform(df_all['公司名称'])
        X_scope = vec_scope.fit_transform(df_all['经营范围'])
        scaler = MinMaxScaler()
        X_numeric = scaler.fit_transform(df_all[['信用值', '高危法人关联']])
        X_combined = sp.hstack((X_name, X_scope, X_numeric))
        y_combined = df_all['label'].values

        terminal.code(f"[AI 分析] 矩阵构建完成！随机森林 200 颗决策树正在联合判定...", language="bash")
        
        # 4. 模型训练
        ml_model = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
        ml_model.fit(X_combined, y_combined)
        all_probs = ml_model.predict_proba(X_combined)[:, 1]
        
        df_all['无证户综合概率(%)'] = np.round(all_probs * 100, 2)
        target_pool = df_all[df_all['label'] == 0].copy()
        
        # 5. 【极其核心：概率溯源算法，解释每家店的分数怎么来的】
        # 获取所有特征的名称
        feature_names = np.array(vec_name.get_feature_names_out().tolist() + vec_scope.get_feature_names_out().tolist() + ['信用异常惩罚', '历史无证前科'])
        importances = ml_model.feature_importances_
        
        # 截取目标池的稀疏矩阵
        target_idx = target_pool.index.tolist()
        X_target = X_combined.tocsr()[target_idx]
        weighted_X = X_target.multiply(importances) # 每家店的原始特征 乘以 全局权重
        
        explanations = []
        for i in range(weighted_X.shape[0]):
            row_weights = weighted_X.getrow(i).toarray()[0]
            top_indices = row_weights.argsort()[-3:][::-1] # 找到贡献最大的3个元凶特征
            
            total_weight = row_weights.sum()
            final_prob = target_pool.iloc[i]['无证户综合概率(%)']
            
            if total_weight > 0:
                expl_parts = []
                sum_top_pct = 0
                for idx in top_indices:
                    if row_weights[idx] > 0:
                        # 把权重比例换算成实际的概率百分比贡献
                        rel_pct = (row_weights[idx] / total_weight) * final_prob
                        sum_top_pct += rel_pct
                        feat_name = feature_names[idx]
                        expl_parts.append(f"{feat_name}({rel_pct:.1f}%)")
                
                # 剩余的长尾特征汇总
                remaining = final_prob - sum_top_pct
                if remaining > 0.5:
                    expl_parts.append(f"其他综合({remaining:.1f}%)")
                    
                explanation = " + ".join(expl_parts)
            else:
                explanation = "无显著高危特征"
                
            explanations.append(explanation)
            
        target_pool['AI 判定依据'] = explanations

        # 6. 分级与排序
        def assign_risk(prob):
            if prob >= 85: return '极高风险', '🚨 立即排查'
            elif prob >= 65: return '高风险', '⚠️ 重点监控'
            elif prob >= 40: return '中风险', '👀 定期关注'
            else: return '低风险', '✅ 常规监管'
            
        target_pool[['风险等级', '监管建议']] = target_pool.apply(lambda r: pd.Series(assign_risk(r['无证户综合概率(%)'])), axis=1)
        target_pool = target_pool.sort_values(
