# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations_with_replacement
from math import comb

st.set_page_config(layout="wide", page_title="Kaprekar Dynamics")

# ================= ЯДРО =================
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

def d_vec(n, k):
    s = sorted(str(n).zfill(k), reverse=True)
    return tuple(int(s[i]) - int(s[k-1-i]) for i in range(k//2))

def _ns(x):
    return int(x) if x <= 10 else int(x) - 9

def S_of(n, k):
    d = digits_of(n, k); m = k // 2
    if k % 2 == 0:
        return tuple(_ns(d[m-1-i] + d[m+i]) for i in range(m))
    return tuple([int(d[m])] + [_ns(d[m-i] + d[m+i]) for i in range(1, m+1)])

def trajectory_table_html(path, k, ci, cyc):
    html = ('<table style="width:100%;border-collapse:collapse;font-size:13px;">'
            '<tr style="background:#f0f0f0;">'
            '<th style="padding:6px;border:1px solid #ccc;">Итерация / Iter</th>'
            '<th style="padding:6px;border:1px solid #ccc;">Число / Number</th>'
            '<th style="padding:6px;border:1px solid #ccc;">S (центр→край) / S</th>'
            '<th style="padding:6px;border:1px solid #ccc;">d-последовательность / d-seq</th>'
            '</tr>')
    for i, n in enumerate(path):
        in_cycle = i >= ci and len(cyc) > 1
        color = 'color:#c02020;font-weight:600;' if in_cycle else ''
        label = str(i)
        if i == ci and len(cyc) > 1:
            label = f"{i} (вход / entry)"
        elif i == len(path) - 1 and len(cyc) > 1 and i > ci:
            label = f"{i} (конец / end)"
        S_str = ", ".join(str(x) for x in S_of(n, k))
        d_str = ", ".join(str(x) for x in d_vec(n, k))
        html += (f'<tr>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;{color}">{label}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;{color}">{n}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;">{S_str}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;">{d_str}</td>'
                 f'</tr>')
    return html + '</table>'

def diff_table_html(path, k):
    html = ('<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:13px;">'
            '<tr style="background:#f0f0f0;">'
            '<th style="padding:6px;border:1px solid #ccc;">Итерация</th>'
            '<th style="padding:6px;border:1px solid #ccc;">От</th>'
            '<th style="padding:6px;border:1px solid #ccc;">К</th>'
            '<th style="padding:6px;border:1px solid #ccc;">Разности цифр</th>'
            '<th style="padding:6px;border:1px solid #ccc;">Нарушение</th>'
            '<th style="padding:6px;border:1px solid #ccc;">Центр равны</th>'
            '</tr>')
    for t in range(len(path) - 1):
        a_s, b_s = str(path[t]).zfill(k), str(path[t+1]).zfill(k)
        diffs = [int(b_s[i]) - int(a_s[i]) for i in range(k)]
        viol = any(diffs[i] + diffs[k-1-i] != 0 for i in range(k))
        if k % 2 == 0:
            ce = a_s[k//2-1] == a_s[k//2]
        else:
            ce = (k >= 3) and (a_s[k//2-1] == a_s[k//2+1])
        chips = ''
        for d in diffs:
            c = '#ffb0b0' if d < 0 else ('#b0e0b0' if d > 0 else '#e6e6e6')
            chips += (f'<span style="display:inline-block;padding:2px 6px;margin:1px;'
                      f'background:{c};border-radius:8px;">{d:+d}</span>')
        bg = '#fff5f5' if viol else 'white'
        html += (f'<tr style="background:{bg};">'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;">{t}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;">{path[t]}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;">{path[t+1]}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;">{chips}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;">'
                 f'{"⚠️" if viol else "—"}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;">'
                 f'{"✓" if ce else "—"}</td></tr>')
    return html + '</table>'

@st.cache_data(show_spinner=False)
def enumerate_classes(k):
    cycles = {}
    for combo in combinations_with_replacement(range(10), k):
        if len(set(combo)) == 1:
            continue
        n = int(''.join(str(x) for x in sorted(combo, reverse=True)))
        path, ci, cyc = trajectory(n, k)
        canon = tuple(sorted(cyc))[0]
        if canon not in cycles:
            cycles[canon] = {'cycle': list(cyc), 'count': 0}
        cycles[canon]['count'] += 1
    return cycles

# ================= UI =================
st.title("Динамика процедуры Капрекара / Kaprekar Dynamics")

tab1, tab2, tab3 = st.tabs([
    "1. Стабилизация и аномалии / Stabilization & anomalies",
    "2. Пространство n-значных / Space of n-digit numbers",
    "3. Сравнение контейнеров / Container comparison",
])

# ================== ВКЛАДКА 1 ==================
with tab1:
    st.header("Стабилизация и аномалии / Stabilization and anomalies")

    col_in, col_btn = st.columns([4, 1])
    with col_in:
        n0 = st.number_input("Число / Number:", value=74943, step=1, key='n0_in')
    with col_btn:
        st.write("")
        st.write("")
        build1 = st.button("Построить / Build", key='b1', use_container_width=True)

    if build1:
        k0 = len(str(n0))
        path, ci, cyc = trajectory(n0, k0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Длина пути / Path length", len(path))
        c2.metric("Вход в цикл / Cycle entry", ci)
        c3.metric("Длина цикла / Cycle length", len(cyc))

        if len(cyc) > 1:
            st.subheader("Полный цикл / Full cycle")
            st.code(" → ".join(str(x) for x in cyc) + " → " + str(cyc[0]))
        else:
            st.subheader("Неподвижная точка / Fixed point")
            st.code(str(cyc[0]))

        st.subheader("Траектория / Trajectory")
        with st.expander("Что такое d-последовательность? / What is d-sequence?", expanded=False):
            st.markdown(
                "**d-последовательность** — это симметричные разности цифр, отсортированных по убыванию.\n\n"
                "Пример для числа **74943**:\n"
                "1. Сортируем цифры по убыванию: `9, 7, 4, 4, 3`.\n"
                "2. Считаем разности пар от краёв к центру:\n"
                "   - d₁ = 9 − 3 = **6** (первая цифра минус последняя)\n"
                "   - d₂ = 7 − 4 = **3** (вторая цифра минус предпоследняя)\n"
                "3. Получаем d = `(6, 3)`.\n\n"
                "**Зачем это нужно?** При вычислении K(n) мы вычитаем `34479` из `97443`. "
                "Разности d₁ и d₂ — это максимальные разности на этих позициях до заимствований. "
                "Они определяют структуру результата."
            )
        st.markdown(trajectory_table_html(path, k0, ci, cyc), unsafe_allow_html=True)

        st.subheader("Таблица разностей / Difference table")
        st.markdown(diff_table_html(path, k0), unsafe_allow_html=True)

        st.subheader("3D-поток разностей / 3D flow of differences")
        st.caption(
            "По X — позиция цифры. По Y — итерация. По Z — значение Δᵢ (разность между итерациями). "
            "Цветные линии соединяют значения **одной и той же позиции** на соседних итерациях. "
            "Красные — Δᵢ < 0, зелёные — Δᵢ > 0, серые — 0. Видно, как «волна» разности течёт во времени.\n\n"
            "Colored lines connect the same position across iterations."
        )
        fig3 = go.Figure()
        diffs_all = []
        for t in range(len(path) - 1):
            a = digits_of(path[t], k0)
            b = digits_of(path[t+1], k0)
            diffs_all.append(b - a)

        # Линии, соединяющие одну позицию во времени
        for i in range(k0):
            zs = [int(diffs_all[t][i]) for t in range(len(diffs_all))]
            for t in range(len(zs)-1):
                col = '#d04040' if zs[t] < 0 else ('#30a030' if zs[t] > 0 else '#808080')
                fig3.add_trace(go.Scatter3d(
                    x=[i, i], y=[t, t+1], z=[zs[t], zs[t+1]],
                    mode='lines+markers',
                    line=dict(color=col, width=4),
                    marker=dict(size=5, color=col),
                    showlegend=False, hoverinfo='skip'))

        fig3.update_layout(
            scene=dict(
                xaxis_title='Позиция / Position',
                yaxis_title='Итерация / Iteration',
                zaxis_title='Δᵢ'),
            height=680, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)

# ================== ВКЛАДКА 2 ==================
with tab2:
    st.header("Пространство n-значных чисел / Space of n-digit numbers")
    st.caption(
        "Каждая точка — класс эквивалентности по мультимножеству цифр. "
        "Число классов = C(k+9, 9)."
    )
    c1, c2 = st.columns(2)
    with c1:
        k2 = st.slider("Длина числа / Number length:", 3, 10, 6, key='k2')
        n_classes = comb(k2 + 9, 9)
        st.info(f"Классов при полном переборе / Classes: **{n_classes:,}**")
    with c2:
        hl = st.number_input(f"Трассируемое число ({k2} цифр) / Trace number:", value=0, step=1, key='hl')
        if hl > 0 and len(str(hl)) != k2:
            st.warning(f"Введите число ровно из {k2} цифр.")

    if st.button("Построить / Build", key='b2'):
        with st.spinner("Перебор..."):
            cycles = enumerate_classes(k2)
            attn = {c: i for i, c in enumerate(cycles.keys())}

            xs, ys, zs, colors, texts = [], [], [], [], []
            for combo in combinations_with_replacement(range(10), k2):
                if len(set(combo)) == 1: continue
                n = int(''.join(str(x) for x in sorted(combo, reverse=True)))
                d = list(d_vec(n, k2)[:3]) + [0]*(3-len(d_vec(n, k2)[:3]))
                path, ci, cyc = trajectory(n, k2)
                canon = tuple(sorted(cyc))[0]
                xs.append(d[0]); ys.append(d[1]); zs.append(d[2])
                colors.append(attn[canon])
                texts.append(f"n={n}<br>цикл={canon}<br>длина={len(cyc)}")

        st.subheader("Циклы и неподвижные числа / Cycles and fixed points")
        total = sum(info['count'] for info in cycles.values())
        rows = [{"Цикл": " → ".join(str(x) for x in info['cycle']),
                 "Длина": len(info['cycle']),
                 "Классов": info['count'],
                 "Доля": f"{info['count']/total*100:.2f}%"}
                for info in sorted(cycles.values(), key=lambda x: -x['count'])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("3D-пространство / 3D space")
        fig = go.Figure(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='markers',
            marker=dict(size=3, color=colors, colorscale='Turbo', opacity=0.55),
            text=texts, hoverinfo='text'))

        if hl > 0 and len(str(hl)) == k2:
            path, ci, cyc = trajectory(hl, k2)
            pts = [(d_vec(n, k2)[0] if len(d_vec(n, k2))>0 else 0,
                    d_vec(n, k2)[1] if len(d_vec(n, k2))>1 else 0,
                    d_vec(n, k2)[2] if len(d_vec(n, k2))>2 else 0, t, n)
                   for t, n in enumerate(path)]
            pre = [p for p in pts if p[3] < ci] + ([pts[ci]] if ci < len(pts) else [])
            cyc_pts = [p for p in pts if p[3] >= ci]
            if pre:
                fig.add_trace(go.Scatter3d(x=[p[0] for p in pre], y=[p[1] for p in pre], z=[p[2] for p in pre],
                    mode='lines+markers', line=dict(color='orange', width=5), marker=dict(size=6, color='orange'),
                    text=[f"итер {p[3]}: {p[4]}" for p in pre], hoverinfo='text', name='путь'))
            if cyc_pts:
                fig.add_trace(go.Scatter3d(x=[p[0] for p in cyc_pts], y=[p[1] for p in cyc_pts], z=[p[2] for p in cyc_pts],
                    mode='lines+markers', line=dict(color='red', width=8), marker=dict(size=9, color='red'),
                    text=[f"итер {p[3]}: {p[4]}" for p in cyc_pts], hoverinfo='text', name='цикл'))

        fig.update_layout(height=750, scene=dict(xaxis_title='d1', yaxis_title='d2', zaxis_title='d3'))
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Почему пирамида?** Точки удовлетворяют d1 >= d2 >= d3 >= 0 — это математическое "
            "следствие того, что d_i — разности убывающих цифр. Для k=3 (только d1) — линия; "
            "для k=4,5 (d1, d2) — плоский треугольник; для k>=6 (d1, d2, d3) — пирамида."
        )

# ================== ВКЛАДКА 3 ==================
with tab3:
    st.header("Сравнение контейнеров / Container comparison")
    st.caption("Поиск чисел с заданным контейнером (например, 86...32) для разных длин и сравнение их d-траекторий.")

    c1, c2 = st.columns(2)
    with c1:
        cont_start = st.text_input("Начальный контейнер / Start container:", "86")
    with c2:
        cont_end = st.text_input("Конечный контейнер / End container:", "32")

    if st.button("Сравнить / Compare", key='b3'):
        fig = go.Figure()
        for k in range(4, 11):
            # Ищем все числа с таким контейнером
            found = False
            for combo in combinations_with_replacement(range(10), k):
                if len(set(combo)) == 1: continue
                n = int(''.join(str(x) for x in sorted(combo, reverse=True)))
                n_str = str(n).zfill(k)
                if n_str.startswith(cont_start) and n_str.endswith(cont_end):
                    path, ci, cyc = trajectory(n, k)
                    d_path = [d_vec(num, k) for num in path]
                    # Найдем длину максимальной d-последовательности для оси X
                    max_m = max(len(d) for d in d_path)
                    for t, d in enumerate(d_path):
                        pad_d = list(d) + [0]*(max_m - len(d))
                        fig.add_trace(go.Scatter3d(
                            x=[t], y=list(range(max_m)), z=pad_d,
                            mode='lines+markers',
                            line=dict(width=3),
                            marker=dict(size=4),
                            name=f"k={k}, итер {t}"
                        ))
                    found = True
                    break # Берем первое попавшееся для примера
            if not found:
                st.write(f"Для k={k} контейнер {cont_start}...{cont_end} не найден в классе.")

        fig.update_layout(
            scene=dict(xaxis_title='Итерация', yaxis_title='Позиция d', zaxis_title='Значение d'),
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)₂ ≥ d₃ ≥ 0 is a mathematical consequence."
        )
