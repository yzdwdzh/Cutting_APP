import streamlit as st
import pandas as pd
from ortools.linear_solver import pywraplp
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go


# --- 核心算法：带拼接与时限的优化 ---
def solve_cutting_stock_with_splicing(parts_dict, stock_length, kerf=0.003):
    processed_parts = []
    total_joints = 0
    for length, count in parts_dict.items():
        if length > stock_length:
            num_full_lengths = int(length // stock_length)
            remainder = length - (num_full_lengths * stock_length)
            total_joints += (num_full_lengths * count)
            for _ in range(count):
                for _ in range(num_full_lengths):
                    processed_parts.append(stock_length)
                if remainder > 0.001: processed_parts.append(remainder)
        else:
            if length > 0.001: processed_parts.extend([length] * count)

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver: return None
    solver.SetTimeLimit(120000)

    num_stock = len(processed_parts)
    x = {}
    for i in range(num_stock):
        for j in range(len(processed_parts)):
            x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')
    y = [solver.IntVar(0, 1, f'y_{i}') for i in range(num_stock)]

    for j in range(len(processed_parts)):
        solver.Add(sum(x[i, j] for i in range(num_stock)) == 1)
    for i in range(num_stock):
        solver.Add(sum(x[i, j] * (processed_parts[j] + kerf) for j in range(len(processed_parts))) <= y[i] * (
                    stock_length + kerf))

    solver.Minimize(solver.Sum(y))
    if solver.Solve() in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        schemes = []
        for i in range(num_stock):
            if y[i].solution_value() > 0.5:
                scheme = sorted(
                    [processed_parts[j] for j in range(len(processed_parts)) if x[i, j].solution_value() > 0.5])
                schemes.append(tuple(scheme))
        return {'count': sum(y[i].solution_value() for i in range(num_stock)), 'summary': Counter(schemes),
                'joints': total_joints}
    return None


# --- 界面层 ---
st.set_page_config(page_title="管材切割优化助手", layout="wide")
st.title("🏗️ 管材切割与拼接工程助手")

# 侧边栏
st.sidebar.header("参数设置")
kerf = st.sidebar.number_input("锯缝宽度 (m)", value=0.003,min_value=0.001, step=0.001,  format="%.3f")
stock_options = [float(s.strip()) for s in
                 st.sidebar.text_input("可选原材(m), 用逗号分隔", value="6.0, 9.0, 12.0").split(",")]

# 需求输入
if 'needs' not in st.session_state: st.session_state.needs = {}
c1, c2 = st.columns(2)
new_len = c1.number_input("零件长度(m)", min_value=0.001, step=0.001, format="%.3f")
new_cnt = c2.number_input("数量(个)", min_value=1, step=1)
if st.button("✅ 添加/更新需求"): st.session_state.needs[new_len] = int(new_cnt)
if st.button("🗑️ 清空所有需求"): st.session_state.needs = {}; st.rerun()

# 需求列表
for length, count in list(st.session_state.needs.items()):
    col1, col2 = st.columns([4, 1])
    col1.write(f"- 零件长度: {length:.3f}m | 数量: {count} 个")
    if col2.button("删除", key=f"del_{length}"): del st.session_state.needs[length]; st.rerun()

# 计算分析
if st.button("🚀 运行最优规划"):
    with st.spinner('正在计算中,请稍候（最长需2分钟）...'):
        report = []
        total_parts_len = sum(k * v for k, v in st.session_state.needs.items())
        for s_len in stock_options:
            res = solve_cutting_stock_with_splicing(st.session_state.needs, s_len, kerf)
            if res:
                waste_rate = ((res['count'] * s_len - total_parts_len) / (res['count'] * s_len)) * 100
                report.append({"原材规格(m)": s_len, "采购根数": res['count'], "需接头数": res['joints'],
                               "损耗率(%)": round(waste_rate, 2), "方案详情": res['summary']})

        df = pd.DataFrame(report).sort_values("损耗率(%)")
        st.table(df.drop(columns=["方案详情"]))

        # 修改损耗率柱状图的逻辑
        fig_waste = px.bar(df, x="原材规格(m)", y="损耗率(%)",
                           color="损耗率(%)", text="损耗率(%)",
                           title="不同规格损耗对比")

        # 核心：通过 update_traces 和 update_layout 调整柱子宽度
        fig_waste.update_traces(width=0.4)  # 将柱子宽度固定在 0.4，这样就不会撑满全屏了
        fig_waste.update_layout(
            xaxis=dict(tickmode='array', tickvals=df["原材规格(m)"]),  # 确保横坐标只显示你输入的规格
            bargap=0.5  # 增加柱子之间的间距
        )
        st.plotly_chart(fig_waste)

        # 2. 切割结构瀑布图
        best = df.iloc[0]
        st.success(f"🏆 推荐规格: {best['原材规格(m)']}m (接头: {best['需接头数']})")

        fig_struct = go.Figure()
        for scheme_idx, (scheme, count) in enumerate(best['方案详情'].items()):
            for part_len in scheme:
                fig_struct.add_trace(go.Bar(
                    x=[part_len], y=[f"模式 {scheme_idx + 1} (执行{count}次)"],
                    orientation='h', text=f"{part_len:.3f}m", textposition='inside'
                ))
        fig_struct.update_layout(barmode='stack', title="原材切割分布示意图")
        st.plotly_chart(fig_struct)

        # 文字清单
        for scheme, count in best['方案详情'].items():
            st.write(f"模式 [{' + '.join([f'{i:.3f}m' for i in scheme])}]: 执行 {count} 次")

