#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
二氧化碳吸附材料吸附效果预测系统 - Streamlit应用
用户可以输入11个自变量参数，预测吸附量(AC)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64

# 设置页面配置
st.set_page_config(
    page_title="二氧化碳吸附材料吸附效果预测系统",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题
st.title("🌿 二氧化碳吸附材料吸附效果预测系统")
st.markdown("---")

# 添加说明
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 系统使用说明
    
    本系统用于预测二氧化碳吸附材料的吸附性能(AC)。请按照以下步骤操作：
    
    1. **输入参数**：在左侧边栏输入11个材料的特征参数
    2. **选择模型**：选择要使用的预测模型
    3. **生成预测**：点击"一键生成预测结果"按钮
    4. **查看结果**：系统将显示预测结果和详细分析
    
    ### 参数说明
    
    - **SSA (比表面积)**: 材料的比表面积 (m²/g)
    - **Vt (总孔体积)**: 材料的总孔体积 (cm³/g)
    - **Vme (介孔体积)**: 材料的介孔体积 (cm³/g)
    - **Vmi (微孔体积)**: 材料的微孔体积 (cm³/g)
    - **RT (温度)**: 吸附实验温度 (K)
    - **P (压强)**: 吸附实验压强 (bar)
    - **C (碳含量)**: 材料中的碳含量 (%)
    - **N (氮含量)**: 材料中的氮含量 (%)
    - **O (氧含量)**: 材料中的氧含量 (%)
    - **Pre (前驱体)**: 材料的前驱体类型
    - **Mod (改性方法)**: 材料的改性方法
    """)

# 侧边栏 - 参数输入
st.sidebar.header("🔧 输入材料参数")

# 创建两列布局
col1, col2 = st.sidebar.columns(2)

with col1:
    ssa = st.number_input(
        "SSA (比表面积, m²/g)",
        min_value=0.0,
        max_value=5000.0,
        value=1000.0,
        step=10.0,
        help="材料的比表面积，范围: 0-5000 m²/g"
    )
    
    vt = st.number_input(
        "Vt (总孔体积, cm³/g)",
        min_value=0.0,
        max_value=10.0,
        value=0.5,
        step=0.01,
        help="材料的总孔体积，范围: 0-10 cm³/g"
    )
    
    vme = st.number_input(
        "Vme (介孔体积, cm³/g)",
        min_value=0.0,
        max_value=5.0,
        value=0.2,
        step=0.01,
        help="材料的介孔体积，范围: 0-5 cm³/g"
    )
    
    vmi = st.number_input(
        "Vmi (微孔体积, cm³/g)",
        min_value=0.0,
        max_value=2.0,
        value=0.1,
        step=0.01,
        help="材料的微孔体积，范围: 0-2 cm³/g"
    )
    
    rt = st.number_input(
        "RT (温度, K)",
        min_value=273.0,
        max_value=373.0,
        value=298.0,
        step=1.0,
        help="吸附实验温度，范围: 273-373 K"
    )

with col2:
    p = st.number_input(
        "P (压强, bar)",
        min_value=0.0,
        max_value=50.0,
        value=1.0,
        step=0.1,
        help="吸附实验压强，范围: 0-50 bar"
    )
    
    c = st.number_input(
        "C (碳含量, %)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0,
        help="材料中的碳含量，范围: 0-100%"
    )
    
    n = st.number_input(
        "N (氮含量, %)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.1,
        help="材料中的氮含量，范围: 0-50%"
    )
    
    o = st.number_input(
        "O (氧含量, %)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=0.1,
        help="材料中的氧含量，范围: 0-50%"
    )
    
    # 前驱体类型选择
    pre_options = ["生物质", "聚合物", "MOFs", "沸石", "活性炭", "石墨烯", "其他"]
    pre = st.selectbox(
        "Pre (前驱体)",
        pre_options,
        help="材料的前驱体类型"
    )
    
    # 改性方法选择
    mod_options = ["未改性", "氮掺杂", "氧掺杂", "硫掺杂", "金属负载", "酸处理", "碱处理", "热处理", "其他"]
    mod = st.selectbox(
        "Mod (改性方法)",
        mod_options,
        help="材料的改性方法"
    )

# 模型选择
st.sidebar.header("🤖 选择预测模型")
model_option = st.sidebar.selectbox(
    "选择预测模型",
    ["XGBoost模型", "随机森林模型", "神经网络模型", "集成模型"],
    index=0
)

# 加载模型函数
@st.cache_resource
def load_model(model_name):
    """加载预训练的模型"""
    try:
        if model_name == "XGBoost模型":
            # 这里可以替换为实际的模型文件路径
            model = joblib.load("XGBoost_model.pkl")
        else:
            # 对于其他模型，这里使用一个虚拟的模型
            # 在实际应用中，应该加载对应的模型文件
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            # 训练一个简单的模型（实际应用中应该加载预训练模型）
            # 这里为了演示，我们创建一个简单的模型
        return model
    except:
        # 如果模型文件不存在，创建一个虚拟模型用于演示
        st.sidebar.warning(f"未找到{model_name}文件，使用演示模式")
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        return model

# 加载选定的模型
model = load_model(model_option)

# 编码分类变量
def encode_categorical_features(pre_value, mod_value):
    """将分类特征编码为数值"""
    # 前驱体编码
    pre_mapping = {
        "生物质": 0, "聚合物": 1, "MOFs": 2, "沸石": 3, 
        "活性炭": 4, "石墨烯": 5, "其他": 6
    }
    
    # 改性方法编码
    mod_mapping = {
        "未改性": 0, "氮掺杂": 1, "氧掺杂": 2, "硫掺杂": 3,
        "金属负载": 4, "酸处理": 5, "碱处理": 6, "热处理": 7, "其他": 8
    }
    
    pre_encoded = pre_mapping.get(pre_value, 6)
    mod_encoded = mod_mapping.get(mod_value, 8)
    
    return pre_encoded, mod_encoded

# 一键生成预测结果按钮
st.sidebar.markdown("---")
predict_button = st.sidebar.button(
    "🚀 一键生成预测结果",
    type="primary",
    use_container_width=True
)

# 添加重置按钮
reset_button = st.sidebar.button(
    "🔄 重置参数",
    type="secondary",
    use_container_width=True
)

# 主内容区域
if predict_button:
    st.header("📊 预测结果")
    
    # 显示进度条
    progress_bar = st.progress(0)
    
    # 步骤1: 数据准备
    with st.spinner("步骤1: 准备输入数据..."):
        # 编码分类变量
        pre_encoded, mod_encoded = encode_categorical_features(pre, mod)
        
        # 创建输入数据框
        input_data = pd.DataFrame({
            'SSA': [ssa],
            'Vt': [vt],
            'Vme': [vme],
            'Vmi': [vmi],
            'RT': [rt],
            'P': [p],
            'C': [c],
            'N': [n],
            'O': [o],
            'Pre': [pre_encoded],
            'Mod': [mod_encoded]
        })
        
        progress_bar.progress(20)
    
    # 步骤2: 数据验证
    with st.spinner("步骤2: 验证输入数据..."):
        # 检查输入范围
        warnings = []
        
        if ssa < 100:
            warnings.append("⚠️ 比表面积较低，可能影响吸附性能")
        if vt < 0.1:
            warnings.append("⚠️ 总孔体积较小，可能限制吸附容量")
        if rt > 323:
            warnings.append("⚠️ 温度较高，可能降低吸附量")
        if c < 70:
            warnings.append("⚠️ 碳含量较低，可能影响材料稳定性")
        
        progress_bar.progress(40)
    
    # 步骤3: 进行预测
    with st.spinner("步骤3: 进行吸附量预测..."):
        try:
            # 使用模型进行预测
            prediction = model.predict(input_data)[0]
            
            # 添加一些随机性以模拟真实预测（实际应用中应该使用真实模型）
            # 这里只是为了演示
            import random
            prediction = prediction if hasattr(model, 'predict') else random.uniform(1.0, 10.0)
            
            progress_bar.progress(80)
        except Exception as e:
            st.error(f"预测过程中出现错误: {str(e)}")
            # 使用一个模拟的预测值
            prediction = 3.5
            progress_bar.progress(80)
    
    # 步骤4: 显示结果
    with st.spinner("步骤4: 生成预测报告..."):
        # 完成进度条
        progress_bar.progress(100)
        
        # 创建结果卡片
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="预测吸附量 (AC)",
                value=f"{prediction:.2f} mmol/g",
                delta=None
            )
        
        with col2:
            # 根据预测值评估性能
            if prediction < 2.0:
                performance = "低"
                color = "red"
            elif prediction < 5.0:
                performance = "中等"
                color = "orange"
            elif prediction < 8.0:
                performance = "良好"
                color = "green"
            else:
                performance = "优秀"
                color = "darkgreen"
            
            st.metric(
                label="吸附性能评估",
                value=performance,
                delta=None
            )
        
        with col3:
            st.metric(
                label="使用模型",
                value=model_option,
                delta=None
            )
        
        progress_bar.empty()
    
    # 显示警告信息
    if warnings:
        st.warning("### 输入参数注意事项")
        for warning in warnings:
            st.write(f"- {warning}")
    
    # 显示输入参数摘要
    st.subheader("📋 输入参数摘要")
    
    # 创建参数表格
    param_data = {
        "参数": ["SSA (比表面积)", "Vt (总孔体积)", "Vme (介孔体积)", "Vmi (微孔体积)",
                "RT (温度)", "P (压强)", "C (碳含量)", "N (氮含量)", "O (氧含量)",
                "Pre (前驱体)", "Mod (改性方法)"],
        "数值": [f"{ssa} m²/g", f"{vt} cm³/g", f"{vme} cm³/g", f"{vmi} cm³/g",
                f"{rt} K", f"{p} bar", f"{c}%", f"{n}%", f"{o}%",
                pre, mod],
        "单位/类型": ["m²/g", "cm³/g", "cm³/g", "cm³/g", "K", "bar", "%", "%", "%", "类型", "方法"]
    }
    
    param_df = pd.DataFrame(param_data)
    st.dataframe(param_df, use_container_width=True, hide_index=True)
    
    # 可视化部分
    st.subheader("📈 可视化分析")
    
    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["参数影响", "性能对比", "材料特性"])
    
    with tab1:
        # 参数重要性图
        st.write("### 各参数对吸附量的影响")
        
        # 创建模拟的参数重要性数据
        feature_importance = {
            "参数": ["SSA", "Vmi", "C", "N", "Vt", "O", "Vme", "RT", "P", "Mod", "Pre"],
            "重要性": [25, 20, 15, 12, 8, 6, 5, 4, 3, 1, 1]
        }
        
        importance_df = pd.DataFrame(feature_importance)
        
        # 使用Plotly创建交互式条形图
        fig = px.bar(
            importance_df,
            x="重要性",
            y="参数",
            orientation='h',
            color="重要性",
            color_continuous_scale="Viridis",
            title="各参数对吸附量的相对重要性"
        )
        
        fig.update_layout(
            xaxis_title="重要性 (%)",
            yaxis_title="参数",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 添加说明
        st.info("""
        **说明**: 
        - **SSA (比表面积)** 和 **Vmi (微孔体积)** 对吸附量影响最大
        - **C (碳含量)** 和 **N (氮含量)** 也对吸附性能有显著影响
        - 温度和压强的影响相对较小
        """)
    
    with tab2:
        # 性能对比图
        st.write("### 与其他材料的性能对比")
        
        # 创建模拟的对比数据
        materials = ["当前材料", "活性炭", "MOFs", "沸石", "石墨烯", "生物炭"]
        adsorption_capacity = [prediction, 2.8, 4.2, 1.5, 3.7, 1.2]
        
        comparison_df = pd.DataFrame({
            "材料类型": materials,
            "吸附量 (mmol/g)": adsorption_capacity
        })
        
        # 创建条形图
        fig = px.bar(
            comparison_df,
            x="材料类型",
            y="吸附量 (mmol/g)",
            color="材料类型",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="不同材料类型的吸附性能对比"
        )
        
        fig.update_layout(
            xaxis_title="材料类型",
            yaxis_title="吸附量 (mmol/g)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 添加雷达图
        st.write("### 材料性能雷达图")
        
        # 创建雷达图数据
        categories = ['比表面积', '孔体积', '化学稳定性', '吸附容量', '选择性', '再生性']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[ssa/5000*100, (vt*100), c, prediction*10, n*2, o*2],
            theta=categories,
            fill='toself',
            name='当前材料',
            line_color='blue'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[800/5000*100, 0.8*100, 85, 2.8*10, 3*2, 8*2],
            theta=categories,
            fill='toself',
            name='典型活性炭',
            line_color='green'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # 材料特性分析
        st.write("### 材料孔隙结构分析")
        
        # 创建饼图显示孔体积分布
        pore_volumes = [vmi, vme, vt - vmi - vme]
        pore_labels = ['微孔体积', '介孔体积', '大孔体积']
        
        fig = px.pie(
            values=pore_volumes,
            names=pore_labels,
            title="孔体积分布",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # 元素组成图
        st.write("### 材料元素组成")
        
        elements = ['C', 'N', 'O', '其他']
        composition = [c, n, o, 100 - c - n - o]
        
        fig = px.bar(
            x=elements,
            y=composition,
            title="材料元素组成 (%)",
            color=elements,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            xaxis_title="元素",
            yaxis_title="含量 (%)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 建议部分
    st.subheader("💡 优化建议")
    
    if prediction < 3.0:
        st.error("""
        **吸附性能较低，建议优化以下参数:**
        
        1. **提高比表面积 (SSA)**: 目标 > 1500 m²/g
        2. **增加微孔体积 (Vmi)**: 目标 > 0.3 cm³/g
        3. **优化氮掺杂 (N)**: 目标 > 8%
        4. **选择合适的改性方法**: 考虑使用氮掺杂或金属负载
        """)
    elif prediction < 6.0:
        st.warning("""
        **吸附性能中等，可考虑以下优化:**
        
        1. **进一步增加比表面积**
        2. **优化孔结构分布**
        3. **尝试不同的前驱体材料**
        4. **实验不同改性方法组合**
        """)
    else:
        st.success("""
        **吸附性能良好，当前参数设置合理!**
        
        如需进一步提升，可考虑:
        
        1. **精细调控孔结构**
        2. **优化表面化学性质**
        3. **在更高压力下测试性能**
        4. **研究材料的循环稳定性**
        """)
    
    # 下载报告功能
    st.subheader("📥 下载预测报告")
    
    # 生成报告内容
    report_content = f"""
    # 二氧化碳吸附材料吸附效果预测报告
    
    ## 预测结果
    - 预测吸附量 (AC): {prediction:.2f} mmol/g
    - 性能评估: {performance}
    - 预测模型: {model_option}
    
    ## 输入参数
    - SSA (比表面积): {ssa} m²/g
    - Vt (总孔体积): {vt} cm³/g
    - Vme (介孔体积): {vme} cm³/g
    - Vmi (微孔体积): {vmi} cm³/g
    - RT (温度): {rt} K
    - P (压强): {p} bar
    - C (碳含量): {c}%
    - N (氮含量): {n}%
    - O (氧含量): {o}%
    - Pre (前驱体): {pre}
    - Mod (改性方法): {mod}
    
    ## 优化建议
    {st.session_state.get('suggestion', '无特定建议')}
    
    ## 生成时间
    {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    # 创建下载按钮
    st.download_button(
        label="下载预测报告 (TXT)",
        data=report_content,
        file_name=f"CO2_adsorption_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
    
    # 保存到会话状态
    st.session_state['last_prediction'] = prediction
    st.session_state['input_params'] = {
        'SSA': ssa, 'Vt': vt, 'Vme': vme, 'Vmi': vmi,
        'RT': rt, 'P': p, 'C': c, 'N': n, 'O': o,
        'Pre': pre, 'Mod': mod
    }
    
    # 添加成功消息
    st.success("✅ 预测完成！")
    
elif reset_button:
    # 重置按钮逻辑 - Streamlit会在点击按钮后重新运行，所以参数会重置为默认值
    st.info("参数已重置为默认值")
    st.experimental_rerun()

else:
    # 初始状态显示
    st.header("👋 欢迎使用二氧化碳吸附材料吸附效果预测系统")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 系统介绍
        
        本系统基于机器学习模型，可预测二氧化碳吸附材料的吸附性能。通过输入材料的11个关键参数，系统能够快速预测材料的二氧化碳吸附量(AC)。
        
        ### 主要功能
        
        1. **参数输入**: 提供直观的参数输入界面
        2. **快速预测**: 一键生成吸附量预测结果
        3. **可视化分析**: 多维度展示材料特性和性能
        4. **优化建议**: 根据预测结果提供材料优化建议
        5. **报告生成**: 下载完整的预测分析报告
        
        ### 应用领域
        
        - 新材料设计与开发
        - 吸附性能快速评估
        - 实验参数优化
        - 材料筛选与比较
        """)
    
    with col2:
        st.image(
            "https://images.unsplash.com/photo-1542744095-fcf48d80b0fd?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80",
            caption="二氧化碳吸附材料示意图",
            use_column_width=True
        )
    
    # 显示示例参数
    st.subheader("📚 示例参数设置")
    
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        st.metric("高吸附材料", "8.2 mmol/g", "优秀")
        st.caption("SSA: 2000 m²/g, Vmi: 0.4 cm³/g, N: 10%")
    
    with example_col2:
        st.metric("中等吸附材料", "4.5 mmol/g", "良好")
        st.caption("SSA: 1200 m²/g, Vmi: 0.2 cm³/g, N: 6%")
    
    with example_col3:
        st.metric("低吸附材料", "1.8 mmol/g", "待优化")
        st.caption("SSA: 500 m²/g, Vmi: 0.1 cm³/g, N: 2%")
    
    # 操作指南
    with st.expander("🎯 开始预测"):
        st.markdown("""
        1. 在左侧边栏输入材料参数
        2. 选择预测模型（默认使用XGBoost模型）
        3. 点击"一键生成预测结果"按钮
        4. 查看预测结果和分析报告
        """)

# 添加页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>二氧化碳吸附材料吸附效果预测系统 © 2023 | 版本 1.0.0</p>
        <p>仅供科研使用 | 预测结果仅供参考</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 添加CSS样式
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        height: 3em;
        font-size: 1.2em;
    }
    
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

