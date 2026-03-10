import streamlit as st
import pandas as pd
from ortools.linear_solver import pywraplp
from collections import Counter


# --- 核心算法：带拼接逻辑的优化 ---
def solve_cutting_stock_with_splicing(parts_dict, stock_length, kerf=0.003):
    processed_parts = []
    total_joints = 0

    # 拼接逻辑处理
    for length, count in parts_dict.items():
        if length > stock_length:
            num_full_lengths = int(length // stock_length)
            remainder = length - (num_full_lengths * stock_length)
            total_joints += (num_full_lengths * count)
            for _ in range(count):
                for _ in range(num_full_lengths):
                    processed_parts.append(stock_length)
                if remainder > 0:
                    processed_parts.append(remainder)
        else:
            processed_parts.extend([length] * count)

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver: return None

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

    if solver.Solve() == pywraplp.Solver.OPTIMAL:
        schemes = []
        for i in range(num_stock):
            if y[i].solution_value() > 0.5:
                scheme = sorted(
                    [processed_parts[j] for j in range(len(processed_parts)) if x[i, j].solution_value() > 0.5])
                schemes.append(tuple(scheme))
        return {
            'count': sum(y[i].solution_value() for i in range(num_stock)),
            'summary': Counter(schemes),
            'joints': total_joints
        }
    return None


# --- 界面层 ---
st.set_page_config(page_title="管材切割优化助手", layout="wide")
st.title("🏗️ 管材切割与拼接工程助手")

# 侧边栏参数
st.sidebar.header("系统设置")
kerf = st.sidebar.number_input("切割锯缝宽度 (m)", value=0.003, format="%.3f")
stock_input = st.sidebar.text_input("可选原材规格(m), 用逗号分隔", value="6.0, 9.0, 12.0")
stock_options = [float(s.strip()) for s in stock_input.split(",")]

# 需求输入管理
if 'needs' not in st.session_state: st.session_state.needs = {}

st.subheader("输入需求清单")
c1, c2 = st.columns(2)
new_len = c1.number_input("零件长度(m)", min_value=0.1, step=0.1)
new_cnt = c2.number_input("需求数量(个)", min_value=1, step=1)

btn_c1, btn_c2 = st.columns(2)
if btn_c1.button("✅ 添加/更新需求"):
    st.session_state.needs[new_len] = int(new_cnt)
if btn_c2.button("🗑️ 清空所有需求"):
    st.session_state.needs = {}
    st.rerun()

st.write("当前需求清单 (点击删除):")
for length, count in list(st.session_state.needs.items()):
    col_a, col_b = st.columns([4, 1])
    col_a.write(f"- 零件长度: {length}m | 数量: {count} 个")
    if col_b.button("删除", key=f"del_{length}"):
        del st.session_state.needs[length]
        st.rerun()

# 运行计算
if st.button("🚀 运行最优规划"):
    report = []
    total_parts_len = sum(k * v for k, v in st.session_state.needs.items())

    for s_len in stock_options:
        res = solve_cutting_stock_with_splicing(st.session_state.needs, s_len, kerf)
        if res:
            total_purchased_len = res['count'] * s_len
            waste_rate = ((total_purchased_len - total_parts_len) / total_purchased_len) * 100
            report.append({
                "原材规格(m)": s_len,
                "采购根数": res['count'],
                "需接头数": res['joints'],
                "损耗率(%)": round(waste_rate, 2),
                "方案详情": res['summary']
            })

    df = pd.DataFrame(report).sort_values("损耗率(%)")
    st.table(df.drop(columns=["方案详情"]))

    best = df.iloc[0]
    st.success(f"🏆 推荐采购：{best['原材规格(m)']}m 原材")
    st.write(f"📊 预计损耗率: {best['损耗率(%)']}% | 预计需拼接接头: {best['需接头数']} 个")

    st.subheader("详细切割方案:")
    for scheme, count in best['方案详情'].items():
        st.write(f"模式 [{' + '.join([f'{i}m' for i in scheme])}]: 需执行 {count} 次")