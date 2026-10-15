# -*- coding: utf-8 -*-
"""
Динамика процедуры Капрекара / Kaprekar Dynamics
Запуск: streamlit run kaprekar_dynamics.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations_with_replacement
from math import comb

st.set_page_config(layout="wide", page_title="Kaprekar Dynamics")


# ============ ЯДРО / CORE ============
def K(n, k):
    s = str(n).zfill(k)
    return int(''.join(sorted(s, reverse=True))) - int(''.join(sorted(s)))

def trajectory(n, k):
    path, seen = [n], {n: 0}
    while True:
        nxt = K(path[-1], k)
        if nxt in seen:
            return path, seen[nxt], path[seen[nxt]:]
        seen[nxt] = len(path)
        path.append(nxt)

def digits_of(n, k):
    return np.array([int(c) for c in str(n).zfill(k)])

def d_vector(n, k):
    s = sorted(str(n).zfill(k), reverse=True)
    return np.array([int(s[i]) - int(s[k-1-i]) for i in range(k//2)])

def norm_sum(s):
    return s if s <= 10 else s - 9

def S_vector(n, k):
    d = digits_of(n, k); m = k // 2
    if k % 2 == 0:
        return tuple(norm_sum(d[m-1-i] + d[m+i]) for i in range(m))
    return tuple([d[m]] + [norm_sum(d[m-i] + d[m+i]) for i in range(1, m+1)])


# ============ UI ============
st.title("Динамика процедуры Капрекара / Kaprekar Dynamics")

t1, t2, t3, t4 = st.tabs([
    "1. Стабилизация числа / Stabilization",
    "2. Аномалии разностей / Difference anomalies",
    "3. Пространство n-значных / Space of n-digit numbers",
    "4. Циклы / Cycles",
])


# ============ ВКЛАДКА 1: СТАБИЛИЗАЦИЯ ============
with t1:
    st.header("Стабилизация числа вдоль траектории / Stabilization along trajectory")
    st.caption(
        "**Тепловая карта цифр**: по оси Y — итерация, по оси X — позиция цифры, цвет — значение. "
        "Видно, как из «рваного» начального числа цифры выстраиваются в устойчивый паттерн.\n\n"
        "**Тепловая карта асимметрии**: aᵢ − a_{k+1−i}. Красный/синий — знаки, белый — ноль. "
        "Симметричное число даёт нулевую полосу."
    )
    c1, c2 = st.columns(2)
    n0 = c1.number_input("Число / Number:", value=12899850, step=1, key='s1n')
    k0 = c2.number_input("Длина / Length:", value=8, min_value=2, max_value=12, step=1, key='s1k')

    if st.button("Построить / Build", key='s1b'):
        path, ci, cyc = trajectory(n0, k0)
        st.write(f"Путь: {len(path)} итераций, вход в цикл с шага {ci}, длина цикла {len(cyc)}")

        M = np.array([digits_of(n, k0) for n in path])
        f1 = go.Figure(go.Heatmap(
            z=M,
            x=[f"поз.{i+1}" for i in range(k0)],
            y=[f"итер.{i}  ({n})" for i, n in enumerate(path)],
            colorscale='Viridis', colorbar=dict(title="цифра")))
        f1.update_layout(height=500, title="Цифры по итерациям / Digits across iterations")
        st.plotly_chart(f1, use_container_width=True)

        A = np.array([[d[i] - d[k0-1-i] for i in range(k0)] for d in M])
        f2 = go.Figure(go.Heatmap(
            z=A,
            x=[f"поз.{i+1}" for i in range(k0)],
            y=[f"итер.{i}" for i in range(len(path))],
            colorscale='RdBu', zmid=0, colorbar=dict(title="aᵢ − a_{k+1−i}")))
        f2.update_layout(height=500, title="Асимметрия числа aᵢ − a_{k+1−i} / Asymmetry")
        st.plotly_chart(f2, use_container_width=True)


# ============ ВКЛАДКА 2: АНОМАЛИИ РАЗНОСТЕЙ ============
with t2:
    st.header("Аномалии разностей и их схлопывание / Difference anomalies")
    st.caption(
        "Δᵢ(t) = aᵢ(t+1) − aᵢ(t) — поразрядная разность между соседними итерациями. "
        "Антисимметрия: Δᵢ = −Δ_{k+1−i}. "
        "Нарушение: |Δᵢ + Δ_{k+1−i}| > 0. "
        "Внизу — суммарная метрика нарушения по итерациям: пики = аномалии, спад = схлопывание."
    )
    c1, c2 = st.columns(2)
    n1 = c1.number_input("Число / Number:", value=12899850, step=1, key='s2n')
    k1 = c2.number_input("Длина / Length:", value=8, min_value=2, max_value=12, step=1, key='s2k')

    if st.button("Построить / Build", key='s2b'):
        path, ci, cyc = trajectory(n1, k1)
        D = np.array([digits_of(path[i+1], k1) - digits_of(path[i], k1)
                      for i in range(len(path)-1)])
        m = k1 // 2
        viol = [sum(abs(row[i] + row[k1-1-i]) for i in range(m)) for row in D]

        f1 = go.Figure(go.Heatmap(
            z=D,
            x=[f"поз.{i+1}" for i in range(k1)],
            y=[f"{path[i]}→{path[i+1]}" for i in range(len(path)-1)],
            colorscale='RdBu', zmid=0, colorbar=dict(title="Δ")))
        f1.update_layout(height=500, title="Разности между итерациями Δᵢ(t) / Digit differences")
        st.plotly_chart(f1, use_container_width=True)

        f2 = go.Figure()
        f2.add_trace(go.Scatter(
            x=list(range(len(viol))), y=viol, mode='lines+markers',
            marker=dict(size=10, color=viol, colorscale='Reds', showscale=True)))
        f2.add_vline(x=ci, line_dash="dash", line_color="green",
                     annotation_text=f"вход в цикл (шаг {ci})")
        f2.update_layout(height=400,
                         xaxis_title="итерация (переход t → t+1)",
                         yaxis_title="Σ |Δᵢ + Δ_{k+1−i}|",
                         title="Метрика нарушения антисимметрии / Violation metric")
        st.plotly_chart(f2, use_container_width=True)

        st.dataframe(pd.DataFrame({
            "Итерация": list(range(len(path)-1)),
            "От": path[:-1],
            "К": path[1:],
            "Δ": [str(list(D[i])) for i in range(len(D))],
            "Нарушение": viol,
        }), use_container_width=True)


# ============ ВКЛАДКА 3: ПРОСТРАНСТВО N-ЗНАЧНЫХ ============
with t3:
    st.header("Пространство n-значных чисел / Space of n-digit numbers")
    st.caption(
        "Каждая точка — класс эквивалентности по мультимножеству цифр (K зависит только от него). "
        "Координаты — первые три симметричные разности d₁, d₂, d₃. "
        "Цвет — аттрактор (неподвижная точка или цикл). "
        "Чёрная линия — путь выделенного числа. "
        "Перебирается C(k+9, 9) классов вместо 10^k чисел."
    )
    c1, c2 = st.columns(2)
    k2 = c1.slider("Длина числа / Number length:", 3, 10, 6, key='s3k')
    hl = c2.number_input("Показать путь числа / Highlight path:", value=0, step=1, key='s3h')

    if st.button("Построить / Build", key='s3b'):
        n_classes = comb(k2 + 9, 9)
        progress = st.progress(0.0)
        pts, i = [], 0
        for combo in combinations_with_replacement(range(10), k2):
            if len(set(combo)) == 1:
                continue
            n = int(''.join(str(x) for x in sorted(combo, reverse=True)))
            path, ci, cyc = trajectory(n, k2)
            d = d_vector(n, k2)
            d_pad = list(d[:3]) + [0]*(3-len(d[:3]))
            pts.append({
                'd1': d_pad[0], 'd2': d_pad[1], 'd3': d_pad[2],
                'n': n,
                'attractor': tuple(sorted(cyc))[0] if cyc else None,
                'cycle_len': len(cyc),
            })
            i += 1
            if i % 500 == 0:
                progress.progress(min(i / n_classes, 1.0))
        progress.progress(1.0)

        df = pd.DataFrame(pts)
        uniq = {a: j for j, a in enumerate(df['attractor'].unique())}
        df['attr_id'] = df['attractor'].map(uniq)

        fig = go.Figure(data=[go.Scatter3d(
            x=df['d1'], y=df['d2'], z=df['d3'], mode='markers',
            marker=dict(size=3, color=df['attr_id'], colorscale='Turbo',
                        opacity=0.5, colorbar=dict(title="аттрактор")),
            text=[f"n={r.n}<br>цикл={r.attractor}<br>длина={r.cycle_len}"
                  for r in df.itertuples()],
            hoverinfo='text')])

        if hl > 0 and len(str(hl)) <= k2:
            path, ci, cyc = trajectory(hl, k2)
            dpath = [d_vector(n, k2) for n in path]
            fig.add_trace(go.Scatter3d(
                x=[d[0] if len(d) > 0 else 0 for d in dpath],
                y=[d[1] if len(d) > 1 else 0 for d in dpath],
                z=[d[2] if len(d) > 2 else 0 for d in dpath],
                mode='lines+markers',
                line=dict(color='black', width=8),
                marker=dict(size=8, color='black'),
                name=f"путь {hl}"))

        fig.update_layout(height=750, scene=dict(
            xaxis_title='d₁', yaxis_title='d₂', zaxis_title='d₃'))
        st.plotly_chart(fig, use_container_width=True)

        st.write(f"Классов: {len(df)}. Уникальных аттракторов: {df['attractor'].nunique()}.")
        top = (df.groupby('attractor')
                 .agg(размер_бассейна=('n', 'count'), длина_цикла=('cycle_len', 'first'))
                 .sort_values('размер_бассейна', ascending=False)
                 .reset_index())
        st.dataframe(top, use_container_width=True)


# ============ ВКЛАДКА 4: ЦИКЛЫ ============
with t4:
    st.header("Циклы и неподвижные точки / Cycles and fixed points")
    st.caption("Полный перебор классов эквивалентности. Циклы выписаны полностью, а не по одному представителю.")
    k3 = st.slider("Длина числа / Number length:", 3, 10, 6, key='s4k')

    if st.button("Перебрать / Enumerate", key='s4b'):
        n_classes = comb(k3 + 9, 9)
        progress = st.progress(0.0)
        cycles, i = {}, 0
        for combo in combinations_with_replacement(range(10), k3):
            if len(set(combo)) == 1:
                continue
            n = int(''.join(str(x) for x in sorted(combo, reverse=True)))
            path, ci, cyc = trajectory(n, k3)
            canon = tuple(sorted(cyc))[0]
            cycles.setdefault(canon, {'cycle': cyc, 'count': 0})
            cycles[canon]['count'] += 1
            i += 1
            if i % 500 == 0:
                progress.progress(min(i / n_classes, 1.0))
        progress.progress(1.0)

        total = sum(c['count'] for c in cycles.values())
        rows = [{
            "Тип / Type": "неподвижная / fixed" if len(c['cycle']) == 1
                          else f"цикл длины {len(c['cycle'])} / {len(c['cycle'])}-cycle",
            "Цикл / Cycle": " → ".join(str(x) for x in c['cycle'])
                            + (" → " + str(c['cycle'][0]) if len(c['cycle']) > 1 else ""),
            "Классов / Count": c['count'],
            "Доля / Share": f"{c['count']/total*100:.2f}%",
        } for c in cycles.values()]
        df = pd.DataFrame(rows).sort_values("Классов / Count", ascending=False)
        st.dataframe(df, use_container_width=True)
