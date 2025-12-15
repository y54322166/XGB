#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
二氧化碳吸附量预测网页应用
使用XGBoost模型预测二氧化碳吸附量
包含：模型加载、数据输入、预测、SHAP解释
"""

# ============== 1. 导入所需库 ==============
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ============== 2. 页面配置 ==============
st.set_page_config(
    page_title="二氧化碳吸附量预测",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== 3. 自定义CSS样式 ==============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .prediction-box {
        background-color: #F0F9FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
    .feature-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)

# ============== 4. 模型和数据处理函数 ==============
class CO2AdsorptionPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.explainer = None
        self.load_model()
    
    def load_model(self):
        """加载训练好的XGBoost模型"""
        try:
            # 尝试从不同路径加载模型
            model_paths = [
                "XGBoost.pkl",
                
            ]
            
            model_loaded = False
            for path in model_paths:
                try:
                    self.model = joblib.load(path)
                    st.sidebar.success(f"模型加载成功: {path}")
                    model_loaded = True
                    break
                except:
                    continue
            
            if not model_loaded:
                st.error("未找到模型文件，请上传模型文件")
                uploaded_model = st.sidebar.file_uploader("上传XGBoost模型文件", type=['pkl', 'joblib'])
                if uploaded_model is not None:
                    self.model = joblib.load(uploaded_model)
                    st.sidebar.success("模型上传成功")
                else:
                    st.warning("请上传模型文件以继续")
                    return None
            
            # 定义特征名称（根据您的描述）
            self.feature_names = [
                "SSA", "Vt", "Vme", "Vmi", "RT", 
                "P", "C", "N", "O", "Pre", "Mod"
            ]
            
            # 创建SHAP解释器
            if self.model is not None:
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                    st.sidebar.success("SHAP解释器初始化成功")
                except Exception as e:
                    st.sidebar.warning(f"SHAP解释器初始化失败: {e}")
            
        except Exception as e:
            st.error(f"模型加载错误: {str(e)}")
    
    def preprocess_input(self, input_df):
        """预处理输入数据"""
        try:
            # 确保列名正确
            if list(input_df.columns) != self.feature_names:
                st.warning(f"输入数据列名不匹配，期望: {self.feature_names}")
                # 尝试重命名列
                if len(input_df.columns) == len(self.feature_names):
                    input_df.columns = self.feature_names
                else:
                    return None
            
            # 处理分类变量（如果存在）
            # 这里可以根据实际的数据预处理方式进行修改
            
            return input_df
        except Exception as e:
            st.error(f"数据预处理错误: {str(e)}")
            return None
    
    def predict(self, input_data):
        """进行预测"""
        try:
            predictions = self.model.predict(input_data)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(input_data)
                return predictions, probabilities
            return predictions, None
        except Exception as e:
            st.error(f"预测错误: {str(e)}")
            return None, None
    
    def calculate_shap_values(self, input_data):
        """计算SHAP值"""
        try:
            if self.explainer is not None:
                shap_values = self.explainer.shap_values(input_data)
                return shap_values
            else:
                st.warning("SHAP解释器未初始化")
                return None
        except Exception as e:
            st.error(f"SHAP值计算错误: {str(e)}")
            return None

# ============== 5. 主应用界面 ==============
def main():
    # 标题
    st.markdown('<h1 class="main-header">🌿 二氧化碳吸附量预测系统</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 初始化预测器
    predictor = CO2AdsorptionPredictor()
    
    if predictor.model is None:
        st.warning("请上传模型文件以继续")
        return
    
    # 侧边栏
    with st.sidebar:
        st.markdown("## ⚙️ 设置")
        
        # 选择输入方式
        input_method = st.radio(
            "选择输入方式",
            ["📝 手动输入", "📁 上传CSV文件", "📊 使用测试数据"]
        )
        
        st.markdown("---")
        st.markdown("## 📊 特征描述")
        
        # 特征描述
        feature_descriptions = {
            "SSA": "比表面积 (m²/g)",
            "Vt": "总孔体积 (cm³/g)",
            "Vme": "介孔体积 (cm³/g)",
            "Vmi": "微孔体积 (cm³/g)",
            "RT": "温度 (K)",
            "P": "压强 (bar)",
            "C": "碳含量 (%)",
            "N": "氮含量 (%)",
            "O": "氧含量 (%)",
            "Pre": "前驱体类型",
            "Mod": "改性方法"
        }
        
        for feature, description in feature_descriptions.items():
            with st.expander(f"{feature}: {description}"):
                st.caption(f"特征: {feature}")
                st.caption(f"描述: {description}")
    
    # 主内容区
    if input_method == "📝 手动输入":
        st.markdown('<h2 class="sub-header">📝 手动输入特征值</h2>', unsafe_allow_html=True)
        
        # 创建两列布局
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            ssa = st.number_input("比表面积 (SSA, m²/g)", min_value=0.0, max_value=5000.0, value=1000.0, step=10.0)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            vt = st.number_input("总孔体积 (Vt, cm³/g)", min_value=0.0, max_value=10.0, value=0.5, step=0.01)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            vme = st.number_input("介孔体积 (Vme, cm³/g)", min_value=0.0, max_value=5.0, value=0.3, step=0.01)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            vmi = st.number_input("微孔体积 (Vmi, cm³/g)", min_value=0.0, max_value=5.0, value=0.2, step=0.01)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            rt = st.number_input("温度 (RT, K)", min_value=200.0, max_value=500.0, value=298.0, step=1.0)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            p = st.number_input("压强 (P, bar)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            c = st.number_input("碳含量 (C, %)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            n = st.number_input("氮含量 (N, %)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            o = st.number_input("氧含量 (O, %)", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            pre_options = [0, 1, 2, 3, 4]
            pre_labels = ["木材", "煤炭", "生物质", "聚合物", "其他"]
            pre = st.selectbox("前驱体类型 (Pre)", options=pre_options, 
                              format_func=lambda x: pre_labels[x])
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            mod_options = [0, 1, 2, 3]
            mod_labels = ["未改性", "酸改性", "碱改性", "热改性"]
            mod = st.selectbox("改性方法 (Mod)", options=mod_options, 
                              format_func=lambda x: mod_labels[x])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 创建输入数据
        input_data = pd.DataFrame([[
            ssa, vt, vme, vmi, rt, p, c, n, o, pre, mod
        ]], columns=predictor.feature_names)
        
    elif input_method == "📁 上传CSV文件":
        st.markdown('<h2 class="sub-header">📁 上传CSV文件</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("选择CSV文件", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # 读取上传的文件
                input_data = pd.read_csv(uploaded_file)
                
                # 显示数据预览
                st.success("文件上传成功!")
                st.write("数据预览:")
                st.dataframe(input_data.head())
                
                # 检查数据列
                st.write("数据信息:")
                st.write(f"行数: {input_data.shape[0]}, 列数: {input_data.shape[1]}")
                
                # 如果列名不匹配，尝试重命名
                if list(input_data.columns) != predictor.feature_names:
                    st.warning("列名不匹配，请确保CSV文件包含以下列:")
                    st.write(predictor.feature_names)
                    
                    if st.checkbox("尝试自动重命名列"):
                        if len(input_data.columns) == len(predictor.feature_names):
                            input_data.columns = predictor.feature_names
                            st.success("列名已重命名")
                        else:
                            st.error("列数不匹配，无法自动重命名")
                            return
                
            except Exception as e:
                st.error(f"文件读取错误: {str(e)}")
                return
        else:
            st.info("请上传CSV文件")
            return
    
    else:  # 使用测试数据
        st.markdown('<h2 class="sub-header">📊 使用测试数据</h2>', unsafe_allow_html=True)
        
        # 尝试加载测试数据
        test_file_paths = ["X-test.csv", "test_features.csv", "./X-test.csv", "./test_features.csv"]
        test_data_loaded = False
        
        for path in test_file_paths:
            try:
                if Path(path).exists():
                    test_data = pd.read_csv(path)
                    st.success(f"测试数据加载成功: {path}")
                    test_data_loaded = True
                    break
            except:
                continue
        
        if not test_data_loaded:
            st.warning("未找到测试数据文件，请上传测试数据")
            uploaded_test_file = st.file_uploader("上传测试数据CSV文件", type=['csv'])
            if uploaded_test_file is not None:
                test_data = pd.read_csv(uploaded_test_file)
                st.success("测试数据上传成功")
                test_data_loaded = True
            else:
                st.info("请上传测试数据文件")
                return
        
        # 显示测试数据
        st.write("测试数据预览:")
        st.dataframe(test_data.head())
        
        # 选择样本
        if len(test_data) > 1:
            sample_idx = st.slider("选择样本", 0, len(test_data)-1, 0)
        else:
            sample_idx = 0
        
        # 使用选中的样本
        input_data = test_data.iloc[[sample_idx]]
        
        # 如果数据列不匹配，尝试调整
        if len(input_data.columns) != len(predictor.feature_names):
            st.warning(f"测试数据列数 ({len(input_data.columns)}) 与模型期望 ({len(predictor.feature_names)}) 不匹配")
    
    # 预处理数据
    if 'input_data' in locals():
        processed_data = predictor.preprocess_input(input_data)
        
        if processed_data is not None:
            # 显示输入数据
            st.markdown('<h2 class="sub-header">📋 输入数据</h2>', unsafe_allow_html=True)
            
            # 创建美观的输入数据显示
            cols = st.columns(4)
            features_display = processed_data.iloc[0].to_dict()
            
            for idx, (feature, value) in enumerate(features_display.items()):
                with cols[idx % 4]:
                    st.metric(
                        label=feature_descriptions.get(feature, feature),
                        value=f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                    )
            
            # 预测按钮
            st.markdown("---")
            if st.button("🚀 开始预测", use_container_width=True):
                with st.spinner("正在计算预测结果..."):
                    # 执行预测
                    predictions, probabilities = predictor.predict(processed_data)
                    
                    if predictions is not None:
                        # 显示预测结果
                        st.markdown('<h2 class="sub-header">🎯 预测结果</h2>', unsafe_allow_html=True)
                        
                        # 创建漂亮的预测结果展示
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                            st.markdown('<p class="metric-label">预测吸附量</p>', unsafe_allow_html=True)
                            st.markdown(f'<p class="metric-value">{predictions[0]:.2f} mmol/g</p>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                            st.markdown('<p class="metric-label">置信度</p>', unsafe_allow_html=True)
                            # 如果有概率，显示置信度
                            if probabilities is not None:
                                confidence = np.max(probabilities[0]) * 100
                                st.markdown(f'<p class="metric-value">{confidence:.1f}%</p>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<p class="metric-value">N/A</p>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                            st.markdown('<p class="metric-label">预测时间</p>', unsafe_allow_html=True)
                            st.markdown(f'<p class="metric-value">实时</p>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # SHAP解释
                        st.markdown('<h2 class="sub-header">🔍 SHAP解释</h2>', unsafe_allow_html=True)
                        
                        # 计算SHAP值
                        shap_values = predictor.calculate_shap_values(processed_data)
                        
                        if shap_values is not None:
                            # 创建两个标签页显示不同的SHAP图
                            tab1, tab2, tab3 = st.tabs(["📊 特征重要性", "📈 单个样本解释", "🎨 依赖图"])
                            
                            with tab1:
                                # 特征重要性条形图
                                st.subheader("特征重要性排序")
                                
                                # 计算平均绝对SHAP值
                                if len(shap_values.shape) == 2:
                                    shap_importance = np.abs(shap_values).mean(0)
                                else:
                                    shap_importance = np.abs(shap_values[0]).mean(0)
                                
                                # 创建DataFrame
                                importance_df = pd.DataFrame({
                                    '特征': predictor.feature_names,
                                    '重要性': shap_importance
                                }).sort_values('重要性', ascending=False)
                                
                                # 使用Plotly创建交互式条形图
                                fig = px.bar(
                                    importance_df,
                                    x='重要性',
                                    y='特征',
                                    orientation='h',
                                    color='重要性',
                                    color_continuous_scale='Blues',
                                    title='特征重要性 (基于SHAP值)'
                                )
                                
                                fig.update_layout(
                                    height=500,
                                    xaxis_title="平均绝对SHAP值",
                                    yaxis_title="特征",
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 显示重要性表格
                                st.write("特征重要性详情:")
                                st.dataframe(importance_df)
                            
                            with tab2:
                                # 单个样本的SHAP解释
                                st.subheader("单个样本的SHAP解释")
                                
                                # 创建force plot
                                try:
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    
                                    # 绘制force plot
                                    shap.force_plot(
                                        predictor.explainer.expected_value,
                                        shap_values[0] if len(shap_values.shape) == 2 else shap_values,
                                        processed_data.iloc[0],
                                        matplotlib=True,
                                        show=False
                                    )
                                    
                                    plt.title("SHAP Force Plot - 单个样本解释")
                                    plt.tight_layout()
                                    
                                    st.pyplot(fig)
                                    
                                    # 解释说明
                                    st.markdown("""
                                    **图解释:**
                                    - **红色箭头**: 增加预测值的特征
                                    - **蓝色箭头**: 减少预测值的特征
                                    - **基准值**: 所有样本的平均预测值
                                    - **输出值**: 当前样本的预测值
                                    """)
                                    
                                except Exception as e:
                                    st.warning(f"无法显示force plot: {str(e)}")
                                    
                                    # 显示备用的瀑布图
                                    st.subheader("特征贡献瀑布图")
                                    
                                    # 创建瀑布图数据
                                    if len(shap_values.shape) == 2:
                                        sample_shap = shap_values[0]
                                    else:
                                        sample_shap = shap_values
                                    
                                    # 创建瀑布图
                                    fig = go.Figure(go.Waterfall(
                                        name="特征贡献",
                                        orientation="v",
                                        measure=["relative"] * len(predictor.feature_names),
                                        x=predictor.feature_names,
                                        y=sample_shap,
                                        text=[f"{val:.3f}" for val in sample_shap],
                                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                                    ))
                                    
                                    fig.update_layout(
                                        title="特征贡献瀑布图",
                                        showlegend=False,
                                        height=500
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            with tab3:
                                # SHAP依赖图
                                st.subheader("特征依赖关系")
                                
                                # 选择最重要的特征
                                if len(shap_values.shape) == 2:
                                    shap_importance = np.abs(shap_values).mean(0)
                                else:
                                    shap_importance = np.abs(shap_values[0]).mean(0)
                                
                                most_important_idx = np.argmax(shap_importance)
                                most_important_feature = predictor.feature_names[most_important_idx]
                                
                                # 创建依赖图
                                try:
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    
                                    shap.dependence_plot(
                                        most_important_feature,
                                        shap_values,
                                        processed_data.values,
                                        feature_names=predictor.feature_names,
                                        show=False
                                    )
                                    
                                    plt.title(f"SHAP依赖图 - {most_important_feature}")
                                    plt.tight_layout()
                                    
                                    st.pyplot(fig)
                                    
                                    # 解释说明
                                    st.markdown(f"""
                                    **图解释:**
                                    - **X轴**: {most_important_feature} 特征值
                                    - **Y轴**: 该特征对预测的SHAP贡献值
                                    - **颜色**: 表示与第二个最重要特征的交互作用
                                    - **趋势线**: 显示特征值变化对预测的影响趋势
                                    """)
                                    
                                except Exception as e:
                                    st.warning(f"无法显示依赖图: {str(e)}")
                        
                        else:
                            st.warning("无法计算SHAP值")
                        
                        # 下载预测结果
                        st.markdown("---")
                        st.subheader("📥 下载预测结果")
                        
                        # 创建结果DataFrame
                        result_df = processed_data.copy()
                        result_df['预测吸附量(mmol/g)'] = predictions
                        
                        if probabilities is not None:
                            result_df['预测置信度(%)'] = np.max(probabilities, axis=1) * 100
                        
                        # 转换为CSV
                        csv = result_df.to_csv(index=False, encoding='utf-8-sig')
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.download_button(
                                label="下载预测结果 (CSV)",
                                data=csv,
                                file_name="co2_adsorption_predictions.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            if st.button("🔄 重新预测"):
                                st.experimental_rerun()
        
        else:
            st.error("数据预处理失败，请检查输入数据格式")

# ============== 6. 应用说明页面 ==============
def about_page():
    st.markdown('<h1 class="main-header">📖 关于本系统</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🌿 系统简介
        
        本系统使用XGBoost机器学习模型预测二氧化碳吸附材料的吸附量。系统基于材料的多项物理化学特征进行预测，并提供可解释的AI分析。
        
        ### 主要功能
        
        1. **材料吸附量预测**: 基于11个特征预测CO₂吸附量
        2. **多种输入方式**: 支持手动输入、文件上传和测试数据
        3. **可解释性分析**: 使用SHAP值解释预测结果
        4. **可视化展示**: 提供丰富的图表展示预测结果和特征重要性
        
        ### 技术特点
        
        - 使用XGBoost机器学习算法
        - 基于SHAP的可解释AI技术
        - 响应式Web界面设计
        - 支持批量预测
        """)
    
    with col2:
        st.markdown("""
        ## 📊 特征说明
        
        本系统使用以下11个特征进行预测:
        
        **物理特征:**
        1. SSA: 比表面积
        2. Vt: 总孔体积
        3. Vme: 介孔体积
        4. Vmi: 微孔体积
        
        **实验条件:**
        5. RT: 温度
        6. P: 压强
        
        **化学组成:**
        7. C: 碳含量
        8. N: 氮含量
        9. O: 氧含量
        
        **材料特性:**
        10. Pre: 前驱体类型
        11. Mod: 改性方法
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 🎯 使用指南
    
    1. **选择输入方式**: 在侧边栏选择手动输入、文件上传或使用测试数据
    2. **输入特征值**: 根据所选方式输入或上传材料特征数据
    3. **开始预测**: 点击"开始预测"按钮获取预测结果
    4. **查看解释**: 分析SHAP解释了解各特征对预测的影响
    5. **下载结果**: 下载预测结果进行进一步分析
    
    ## ⚠️ 注意事项
    
    - 确保输入数据的格式正确
    - 特征值应在合理范围内
    - 分类变量需使用预设的数值编码
    - 文件上传支持CSV格式
    """)

# ============== 7. 应用路由 ==============
def main_app():
    # 侧边栏导航
    st.sidebar.markdown("## 🧭 导航")
    
    page = st.sidebar.radio(
        "选择页面",
        ["🏠 首页 - 预测", "📖 关于"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ℹ️ 系统信息")
    st.sidebar.info("""
    **版本**: 1.0.0  
    **更新日期**: 2024年  
    **开发者**: CO2吸附材料研究组  
    **用途**: 二氧化碳吸附量预测
    """)
    
    # 页面路由
    if page == "🏠 首页 - 预测":
        main()
    elif page == "📖 关于":
        about_page()

# ============== 8. 运行应用 ==============
if __name__ == "__main__":
    main_app()

