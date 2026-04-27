import streamlit as st
import pandas as pd
import numpy as np
import jieba
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import io

# 网页基本设置
st.set_page_config(page_title="无证户智能稽查天眼", page_icon="👁️", layout="wide")

# 停用词和拦截词库
CUSTOM_STOP_WORDS = {'有限', '责任', '分公司', '集团', '控股', '股份', '有限公司', '徐州', '地址', '未知', '公司',
                     '店铺'}
TOBACCO_WORDS = {'烟草制品零售', '卷烟零售', '雪茄零售', '烟丝零售', '香烟销售', '烟草销售', '烟草', '卷烟', '雪茄',
                 '烟丝', '香烟'}


def custom_tokenizer(text):
    if not isinstance(text, str) or not text: return []
    norm_map = {'百货店': '百货', '百货商场': '百货', '百货公司': '百货', '百货超市': '百货', '便利店': '便利',
                '批发部': '批发'}
    words = jieba.lcut(text)
    processed_words = []
    for w in words:
        w = norm_map.get(w, w)
        if len(w) > 1 and w not in CUSTOM_STOP_WORDS:
            if not any(tob_w in w for tob_w in TOBACCO_WORDS):
                processed_words.append(w)
    return processed_words


# 网页标题
st.title("👁️ 卷烟无证经营户动态筛查 AI 模型")
st.markdown("上传数据，AI 将自动从海量营业执照中锁定高危商户。")

# 左侧上传面板
with st.sidebar:
    st.header("📂 1. 上传数据")
    file_biz = st.file_uploader("上传【营业执照】全量名单", type=["xlsx", "csv"])
    file_unl = st.file_uploader("上传【历史无证户】名单", type=["xlsx", "csv"])
    start_btn = st.button("🚀 2. 启动 AI 深度筛查", type="primary", use_container_width=True)

# 核心运行逻辑
if start_btn:
    if not file_biz or not file_unl:
        st.warning("⚠️ 请先在左侧上传【营业执照】和【无证户】两个文件！")
    else:
        # 使用进度条增强演示效果
        progress_bar = st.progress(0, text="正在读取并清洗数据...")

        # 1. 读取数据
        try:
            biz = pd.read_excel(file_biz) if file_biz.name.endswith('.xlsx') else pd.read_csv(file_biz)
            unl = pd.read_excel(file_unl) if file_unl.name.endswith('.xlsx') else pd.read_csv(file_unl)
        except Exception as e:
            st.error(f"读取文件失败，请确保格式正确: {e}")
            st.stop()

        if '天眼评分' in biz.columns: biz.rename(columns={'天眼评分': '信用值'}, inplace=True)
        if '天眼评分' in unl.columns: unl.rename(columns={'天眼评分': '信用值'}, inplace=True)

        overlap_cols = [col for col in biz.columns if '重合' in col]
        if overlap_cols:
            biz = biz[~biz[overlap_cols[0]].isin(['是', '1', 1, True, 'TRUE', 'true'])]

        progress_bar.progress(30, text="正在构建高危关联图谱与特征工程...")

        # 2. 特征工程
        fill_dict = {'公司名称': '未知', '法定代表人': '未知', '注册地址': '未知', '经营范围': '未知', '信用值': 0,
                     '统一社会信用代码': '未知'}
        biz = biz.fillna(fill_dict).drop_duplicates(subset=['统一社会信用代码'], keep='first')
        unl = unl.fillna(fill_dict).drop_duplicates(subset=['统一社会信用代码'], keep='first')
        biz['信用值'] = pd.to_numeric(biz['信用值'], errors='coerce').fillna(0)
        unl['信用值'] = pd.to_numeric(unl['信用值'], errors='coerce').fillna(0)

        bad_reps = set(unl[~unl['法定代表人'].isin(['未知', '', '无'])]['法定代表人'].unique())
        biz['高危法人关联'] = biz['法定代表人'].apply(lambda x: 1 if x in bad_reps else 0)
        unl['高危法人关联'] = 1

        biz['label'] = 0
        unl['label'] = 1
        df_all = pd.concat([unl, biz], ignore_index=True)

        progress_bar.progress(60, text="AI 正在进行百万级参数的文本向量化...")

        # 3. NLP
        vec_name = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=1000)
        vec_scope = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=1500)
        X_name = vec_name.fit_transform(df_all['公司名称'])
        X_scope = vec_scope.fit_transform(df_all['经营范围'])
        scaler = MinMaxScaler()
        X_numeric = scaler.fit_transform(df_all[['信用值', '高危法人关联']])
        X_combined = sp.hstack((X_name, X_scope, X_numeric))
        y_combined = df_all['label'].values

        progress_bar.progress(85, text="随机森林 200 个决策引擎正在进行表决计算...")

        # 4. 训练与预测
        ml_model = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42,
                                          n_jobs=-1)
        ml_model.fit(X_combined, y_combined)
        all_probs = ml_model.predict_proba(X_combined)[:, 1]

        df_all['无证户综合概率(%)'] = np.round(all_probs * 100, 2)
        target_pool = df_all[df_all['label'] == 0].copy()


        def assign_risk(prob):
            if prob >= 85:
                return '极高风险', '🚨 立即现场检查'
            elif prob >= 65:
                return '高风险', '⚠️ 重点排查监管'
            elif prob >= 40:
                return '中风险', '👀 定期关注'
            else:
                return '低风险', '✅ 常规监管'


        target_pool[['风险等级', '监管建议']] = target_pool.apply(
            lambda r: pd.Series(assign_risk(r['无证户综合概率(%)'])), axis=1)
        target_pool = target_pool.sort_values(by='无证户综合概率(%)', ascending=False)

        progress_bar.progress(100, text="✅ 演算完成！")

        # 5. 网页展示结果
        st.success(f"🎯 筛查任务完美收官！总计排查目标营业执照: {len(target_pool)} 家。")

        high_risk_count = len(target_pool[target_pool['无证户综合概率(%)'] >= 85])
        col1, col2, col3 = st.columns(3)
        col1.metric("极高风险目标锁定", f"{high_risk_count} 家", "需立即行动")
        col2.metric("高危法人揪出", f"{target_pool['高危法人关联'].sum()} 人", "存在前科")
        col3.metric("AI 演算耗时", "成功", "极速")

        st.subheader("🚨 极高风险打击名单 TOP 15")
        display_cols = ['公司名称', '无证户综合概率(%)', '风险等级', '监管建议', '法定代表人', '注册地址', '信用值']
        st.dataframe(target_pool[display_cols].head(15), use_container_width=True)

        # 提供下载按钮
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            target_pool[display_cols].to_excel(writer, index=False)
        st.download_button(label="📥 导出完整排查名单 (Excel)", data=buffer, file_name="智能筛查风险名单.xlsx",
                           mime="application/vnd.ms-excel")