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
    return tuple(int(s[i]) - int(s[k - 1 - i]) for i in range(k // 2))


def z_of(n, k):
    """Глубина каскада заимствований: число нулей в конце d-вектора."""
    d = d_vec(n, k)
    z = 0
    for v in reversed(d):
        if v == 0:
            z += 1
        else:
            break
    return z


def _ns(x):
    """Нормировка для S-вектора: 10 остаётся 10, 18 → 9."""
    return int(x) if x <= 10 else int(x) - 9


def S_of(n, k):
    """SP1: вектор симметричных сумм от центра к краю."""
    d = digits_of(n, k)
    m = k // 2
    if k % 2 == 0:
        return tuple(_ns(d[m - 1 - i] + d[m + i]) for i in range(m))
    return tuple([int(d[m])] + [_ns(d[m - i] + d[m + i]) for i in range(1, m + 1)])


def SP2(n, k):
    """SP2: поразрядные суммы половин (сырые, без редукции)."""
    a = digits_of(n, k)
    m = k // 2
    if k % 2 == 0:
        return tuple(int(a[i] + a[m + i]) for i in range(m))
    return tuple(int(a[i] + a[m + 1 + i]) for i in range(m))


def SP3(n, k):
    """SP3: поразрядное вычитание (левая − правая половина)."""
    a = digits_of(n, k)
    m = k // 2
    if k % 2 == 0:
        return tuple(int(a[i] - a[m + i]) for i in range(m))
    return tuple(int(a[i] - a[m + 1 + i]) for i in range(m))


def dr(x):
    """Цифровой корень."""
    if x == 0:
        return 0
    return 1 + (x - 1) % 9


def fmt_num(n, k):
    return str(n).zfill(k)


# ================= ТАБЛИЦЫ =================

def trajectory_table_html(path, k, ci, cyc):
    html = ('<table style="width:100%;border-collapse:collapse;font-size:13px;">'
            '<tr style="background:#f0f0f0;">'
            '<th style="padding:6px;border:1px solid #ccc;">Итерация</th>'
            '<th style="padding:6px;border:1px solid #ccc;">Число</th>'
            '<th style="padding:6px;border:1px solid #ccc;">S (центр→край)</th>'
            '<th style="padding:6px;border:1px solid #ccc;">d-последовательность</th>'
            '<th style="padding:6px;border:1px solid #ccc;">SP2</th>'
            '<th style="padding:6px;border:1px solid #ccc;">SP3</th>'
            '<th style="padding:6px;border:1px solid #ccc;">z</th>'
            '</tr>')
    for i, n in enumerate(path):
        in_cycle = i >= ci and len(cyc) > 1
        color = 'color:#c02020;font-weight:600;' if in_cycle else ''
        label = str(i)
        if i == ci and len(cyc) > 1:
            label = f"{i} (вход)"
        elif i == len(path) - 1 and len(cyc) > 1 and i > ci:
            label = f"{i} (конец)"
        S_str = ", ".join(str(x) for x in S_of(n, k))
        d_str = ", ".join(str(x) for x in d_vec(n, k))
        sp2 = SP2(n, k)
        sp3 = SP3(n, k)
        sp2_str = ", ".join(f"{v}({dr(v)})" for v in sp2)
        sp3_str = ", ".join(f"{v:+d}" for v in sp3)
        z_val = z_of(n, k)
        html += (f'<tr>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;{color}">{label}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;{color}">{fmt_num(n, k)}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;">{S_str}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;">{d_str}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;font-family:monospace;">{sp2_str}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;font-family:monospace;">{sp3_str}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;">{z_val}</td>'
                 f'</tr>')
    return html + '</table>'


def diff_table_html(path, k):
    html = ('<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:13px;">'
            '<tr style="background:#f0f0f0;">'
            '<th style="padding:6px;border:1px solid #ccc;">Итерация</th>'
            '<th style="padding:6px;border:1px solid #ccc;">От</th>'
            '<th style="padding:6px;border:1px solid #ccc;">К</th>'
            '<th style="padding:6px;border:1px solid #ccc;">Разности цифр</th>'
            '<th style="padding:6px;border:1px solid #ccc;">Антисимм.</th>'
            '<th style="padding:6px;border:1px solid #ccc;">z(вход)</th>'
            '</tr>')
    for t in range(len(path) - 1):
        a_s, b_s = str(path[t]).zfill(k), str(path[t + 1]).zfill(k)
        diffs = [int(b_s[i]) - int(a_s[i]) for i in range(k)]
        pair_viol = any(diffs[i] + diffs[k - 1 - i] != 0 for i in range(k // 2))
        z_in = z_of(path[t], k)
        chips = ''
        for d in diffs:
            c = '#ffb0b0' if d < 0 else ('#b0e0b0' if d > 0 else '#e6e6e6')
            chips += (f'<span style="display:inline-block;padding:2px 6px;margin:1px;'
                      f'background:{c};border-radius:8px;">{d:+d}</span>')
        bg = '#fff5f5' if pair_viol else 'white'
        html += (f'<tr style="background:{bg};">'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;">{t}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;">{fmt_num(path[t], k)}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;">{fmt_num(path[t + 1], k)}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;">{chips}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;">'
                 f'{"⚠️" if pair_viol else "✓"}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;">{z_in}</td>'
                 f'</tr>')
    return html + '</table>'


def diff_matrix_fig(path, k):
    if len(path) < 2:
        return None, []
    base = digits_of(path[0], k)
    M = np.array([digits_of(path[t], k) - base for t in range(len(path))])
    status = []
    for t in range(len(path)):
        pair_viol = any(M[t][i] + M[t][k - 1 - i] != 0 for i in range(k // 2))
        status.append((pair_viol, z_of(path[t], k)))
    y_labels = []
    for t in range(len(path)):
        mark = " ⚠" if status[t][0] else ""
        z_mark = f" z={status[t][1]}" if status[t][1] > 0 else ""
        y_labels.append(f"итер {t}{mark}{z_mark}")
    fig = go.Figure(go.Heatmap(
        z=M,
        x=[f"поз {i}" for i in range(k)],
        y=y_labels,
        colorscale='RdBu', zmid=0,
        colorbar=dict(title="Δ"),
        hovertemplate='поз %{x}<br>%{y}<br>Δ=%{z}<extra></extra>',
    ))
    fig.update_layout(
        height=max(300, 22 * len(path)),
        margin=dict(l=10, r=10, t=30, b=10),
        title="Матрица разностей: все итерации vs первая",
    )
    return fig, status


# ================= ПЕРЕБОР =================

@st.cache_data(show_spinner="Перебираю классы эквивалентности...")
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


@st.cache_data(show_spinner="Строю индекс соседей...")
def build_neighbor_index(k):
    cycles = enumerate_classes(k)
    index = {}
    for canon, info in cycles.items():
        cyc = info['cycle']
        if len(cyc) < 2:
            continue
        full = cyc + [cyc[0]]
        last_diffs = tuple(
            int(str(full[t + 1]).zfill(k)[-1]) - int(str(full[t]).zfill(k)[-1])
            for t in range(len(cyc))
        )
        index.setdefault((len(cyc), last_diffs), []).append(canon)
    return index


def find_neighbors(n, k):
    path, ci, cyc = trajectory(n, k)
    if len(cyc) < 2:
        return None, None, None
    full = cyc + [cyc[0]]
    first_diffs = tuple(
        int(str(full[t + 1]).zfill(k)[0]) - int(str(full[t]).zfill(k)[0])
        for t in range(len(cyc))
    )
    index = build_neighbor_index(k)
    return cyc, first_diffs, index.get((len(cyc), first_diffs), [])


@st.cache_data(show_spinner="Считаю частоты контейнеров...")
def container_stats(k, cont_start, cont_end, min_len=4):
    if k < min_len:
        return {}
    cycles = enumerate_classes(k)
    stats = {}
    for canon, info in cycles.items():
        if len(info['cycle']) < 2:
            continue
        hits = sum(
            1 for num in info['cycle']
            if str(num).zfill(k).startswith(cont_start)
            and str(num).zfill(k).endswith(cont_end)
        )
        if hits:
            stats[canon] = hits
    return stats


# ================= UI =================

st.title("Динамика процедуры Капрекара / Kaprekar Dynamics")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Стабилизация и аномалии",
    "2. Пространство n-значных",
    "3. Сравнение контейнеров",
    "4. Поиск соседей",
])


# ================== ВКЛАДКА 1 ==================

with tab1:
    st.header("Стабилизация и аномалии / Stabilization and anomalies")

    if 'n0_build' not in st.session_state:
        st.session_state.n0_build = 12899850

    col_in, col_btn = st.columns([4, 1])
    with col_in:
        n_input = st.number_input("Число / Number:", value=12899850, step=1, key='n0_in')
    with col_btn:
        st.write("")
        st.write("")
        if st.button("Построить / Build", key='b1', use_container_width=True):
            st.session_state.n0_build = n_input

    n0 = st.session_state.n0_build
    k0 = len(str(n0))
    path, ci, cyc = trajectory(n0, k0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Длина пути / Path", len(path))
    c2.metric("Вход в цикл / Cycle entry", ci)
    c3.metric("Длина цикла / Cycle len", len(cyc))

    if len(cyc) > 1:
        st.subheader("Полный цикл / Full cycle")
        st.code(" → ".join(fmt_num(x, k0) for x in cyc) + " → " + fmt_num(cyc[0], k0))
    else:
        st.subheader("Неподвижная точка / Fixed point")
        st.code(fmt_num(cyc[0], k0))

    st.subheader("Траектория / Trajectory")
    st.markdown(trajectory_table_html(path, k0, ci, cyc), unsafe_allow_html=True)

    st.subheader("Таблица разностей / Difference table")
    st.markdown(diff_table_html(path, k0), unsafe_allow_html=True)

    st.subheader("Матрица антисимметрии / Antisymmetry matrix")
    fig_mat, status = diff_matrix_fig(path, k0)
    if fig_mat is not None:
        st.plotly_chart(fig_mat, use_container_width=True, key="mat_main")
        viol_rows = [t for t, (v, _) in enumerate(status) if v]
        if viol_rows:
            st.warning(f"Нарушения антисимметрии: итерации {viol_rows}. "
                       f"Проверьте z(вход) в diff-таблице.")
        else:
            st.success("Антисимметрия выполнена для всех итераций.")

    st.subheader("3D-Динамика / 3D Dynamics")
    view_mode = st.radio(
        "Проекция / Projection:",
        ("Поверхность (Surface)", "Только линии (Lines)",
         "Спираль (Helix)", "Замыкание цикла (Torus)"),
        horizontal=True, key='mode3d',
    )

    fig3 = go.Figure()
    Z_diffs = np.array([digits_of(path[t + 1], k0) - digits_of(path[t], k0)
                        for t in range(len(path) - 1)])
    x_positions = np.arange(k0) - (k0 - 1) / 2

    if view_mode == "Поверхность (Surface)":
        st.caption("Седловидная поверхность антисимметрии.")
        fig3.add_trace(go.Surface(
            x=x_positions, y=np.arange(len(Z_diffs)), z=Z_diffs,
            colorscale='RdBu', cmid=0, opacity=0.9,
            colorbar=dict(title="Разность (Δ)"),
        ))
        fig3.update_layout(scene=dict(xaxis_title='Позиция',
                                      yaxis_title='Итерация',
                                      zaxis_title='Δ'))

    elif view_mode == "Только линии (Lines)":
        st.caption("Траектории отдельных позиций. Видно, как каждая цифра «прыгает».")
        for t in range(len(Z_diffs)):
            colors = ['#d04040' if v < 0 else ('#30a030' if v > 0 else '#808080')
                      for v in Z_diffs[t]]
            fig3.add_trace(go.Scatter3d(
                x=x_positions, y=[t] * k0, z=Z_diffs[t],
                mode='lines+markers',
                line=dict(color='black', width=3),
                marker=dict(size=5, color=colors),
                showlegend=False, hoverinfo='skip',
            ))
        fig3.update_layout(scene=dict(xaxis_title='Позиция',
                                      yaxis_title='Итерация',
                                      zaxis_title='Δ'))

    elif view_mode == "Спираль (Helix)":
        st.caption("Ось времени закручена в спираль. Каждая линия — позиция цифры.")
        turns = 2
        theta = np.linspace(0, turns * 2 * np.pi, len(Z_diffs))
        radius = 10
        for i in range(k0):
            z_vals = np.array([Z_diffs[t][i] for t in range(len(Z_diffs))])
            r_vals = radius + z_vals
            fig3.add_trace(go.Scatter3d(
                x=r_vals * np.cos(theta), y=r_vals * np.sin(theta),
                z=np.arange(len(Z_diffs)),
                mode='lines+markers',
                line=dict(width=3),
                marker=dict(size=3, color=['#d04040' if v < 0 else '#30a030'
                                            for v in z_vals]),
                name=f"Позиция {i + 1}",
            ))
        fig3.update_layout(scene=dict(
            xaxis_title='X (Спираль)', yaxis_title='Y (Спираль)',
            zaxis_title='Итерация',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ))

    elif view_mode == "Замыкание цикла (Torus)":
        if len(cyc) == 1:
            st.warning("Неподвижная точка. Цикл не замыкается в Тор.")
        else:
            st.caption("Только итерации цикла. Гладкий Тор, где разности образуют рельеф.")
            cycle_path = cyc + [cyc[0]]
            Z_cyc = np.array([digits_of(cycle_path[t + 1], k0) - digits_of(cycle_path[t], k0)
                              for t in range(len(cycle_path) - 1)])
            Z_pad = np.vstack([Z_cyc, Z_cyc[0:1]])
            Z_pad = np.hstack([Z_pad, Z_pad[:, 0:1]])
            u = np.linspace(0, 2 * np.pi, Z_pad.shape[0])
            v = np.linspace(0, 2 * np.pi, Z_pad.shape[1])
            U, V = np.meshgrid(u, v, indexing='ij')
            R, r_base = 15, 5
            r = r_base + Z_pad
            X_t = (R + r * np.cos(V)) * np.cos(U)
            Y_t = (R + r * np.cos(V)) * np.sin(U)
            Z_t = r * np.sin(V)
            fig3.add_trace(go.Surface(
                x=X_t, y=Y_t, z=Z_t, surfacecolor=Z_pad,
                colorscale='RdBu', cmid=0,
                colorbar=dict(title="Разность (Δ)"),
            ))
            fig3.update_layout(scene=dict(
                xaxis_title='X (Тор)', yaxis_title='Y (Тор)',
                zaxis_title='Рельеф разностей',
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
            ))

    st.plotly_chart(fig3, use_container_width=True, key=f"plot_{view_mode}")


# ================== ВКЛАДКА 2 ==================

with tab2:
    st.header("Пространство n-значных чисел / Space of n-digit numbers")
    st.caption("Каждая точка — класс эквивалентности по мультимножеству цифр. "
               "Число классов = C(k+9, 9).")

    c1, c2 = st.columns(2)
    with c1:
        k2 = st.slider("Длина числа / Number length:", 2, 10, 6, key='k2')
        n_classes = comb(k2 + 9, 9)
        st.info(f"Классов при полном переборе: **{n_classes:,}**")
    with c2:
        hl = st.number_input(f"Трассируемое число ({k2} цифр):",
                             value=0, step=1, key='hl')
        if hl > 0 and len(str(hl)) != k2:
            st.warning(f"Введите число ровно из {k2} цифр.")

    if st.button("Построить / Build", key='b2'):
        with st.spinner("Перебор..."):
            cycles = enumerate_classes(k2)
        attn = {c: i for i, c in enumerate(cycles.keys())}
        xs, ys, zs, colors, texts = [], [], [], [], []
        for combo in combinations_with_replacement(range(10), k2):
            if len(set(combo)) == 1:
                continue
            n = int(''.join(str(x) for x in sorted(combo, reverse=True)))
            d = list(d_vec(n, k2)[:3]) + [0] * (3 - len(d_vec(n, k2)[:3]))
            path_i, ci_i, cyc_i = trajectory(n, k2)
            canon = tuple(sorted(cyc_i))[0]
            xs.append(d[0]); ys.append(d[1]); zs.append(d[2])
            colors.append(attn[canon])
            texts.append(f"n={fmt_num(n, k2)}<br>цикл={fmt_num(canon, k2)}<br>длина={len(cyc_i)}")

        st.subheader("Циклы и неподвижные числа")
        total = sum(info['count'] for info in cycles.values())
        rows = [{"Цикл": " → ".join(fmt_num(x, k2) for x in info['cycle']),
                 "Длина": len(info['cycle']),
                 "Классов": info['count'],
                 "Доля": f"{info['count'] / total * 100:.2f}%"}
                for info in sorted(cycles.values(), key=lambda x: -x['count'])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("3D-пространство")
        fig = go.Figure(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='markers',
            marker=dict(size=3, color=colors, colorscale='Turbo', opacity=0.55),
            text=texts, hoverinfo='text'))
        if hl > 0 and len(str(hl)) == k2:
            path_t, ci_t, cyc_t = trajectory(hl, k2)
            pts = []
            for t, n in enumerate(path_t):
                dv = d_vec(n, k2)
                pts.append((dv[0] if len(dv) > 0 else 0,
                            dv[1] if len(dv) > 1 else 0,
                            dv[2] if len(dv) > 2 else 0, t, n))
            pre = [p for p in pts if p[3] < ci_t] + ([pts[ci_t]] if ci_t < len(pts) else [])
            cyc_pts = [p for p in pts if p[3] >= ci_t]
            if pre:
                fig.add_trace(go.Scatter3d(
                    x=[p[0] for p in pre], y=[p[1] for p in pre], z=[p[2] for p in pre],
                    mode='lines+markers',
                    line=dict(color='orange', width=5),
                    marker=dict(size=6, color='orange'),
                    text=[f"итер {p[3]}: {fmt_num(p[4], k2)}" for p in pre],
                    hoverinfo='text', name='путь'))
            if cyc_pts:
                fig.add_trace(go.Scatter3d(
                    x=[p[0] for p in cyc_pts], y=[p[1] for p in cyc_pts], z=[p[2] for p in cyc_pts],
                    mode='lines+markers',
                    line=dict(color='red', width=8),
                    marker=dict(size=9, color='red'),
                    text=[f"итер {p[3]}: {fmt_num(p[4], k2)}" for p in cyc_pts],
                    hoverinfo='text', name='цикл'))
        fig.update_layout(height=750,
                          scene=dict(xaxis_title='d1', yaxis_title='d2', zaxis_title='d3'))
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Почему пирамида?** Точки удовлетворяют d1 ≥ d2 ≥ d3 ≥ 0. "
            "Это следствие того, что d — разности убывающих цифр. "
            "Для k=2 (только d1) — точки на оси; k=3 — линия; "
            "k=4,5 — плоский треугольник; k≥6 — пирамида."
        )


# ================== ВКЛАДКА 3 ==================

with tab3:
    st.header("Сравнение контейнеров / Container comparison")
    st.caption("Частотный анализ контейнеров во всех циклах для разных длин.")

    c1, c2 = st.columns(2)
    with c1:
        cont_start = st.text_input("Начальный контейнер:", "86")
    with c2:
        cont_end = st.text_input("Конечный контейнер:", "32")

    if st.button("Сравнить / Compare", key='b3'):
        all_rows = []
        for k in range(4, 11):
            stats = container_stats(k, cont_start, cont_end)
            if not stats:
                all_rows.append({"k": k, "цикл": "—",
                                 "вхождений": 0, "примечание": "не найдено"})
                continue
            for canon, hits in stats.items():
                cyc_info = enumerate_classes(k)[canon]
                all_rows.append({
                    "k": k,
                    "цикл": " → ".join(fmt_num(x, k) for x in cyc_info['cycle']),
                    "длина": len(cyc_info['cycle']),
                    "вхождений": hits,
                })
        st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)


# ================== ВКЛАДКА 4 ==================

with tab4:
    st.header("Поиск соседей / Neighbor search")
    st.caption(
        "Полный перебор всех циклов: ищем циклы, у которых правая граница разностей "
        "первой цифры совпадает с левой границей разностей нашего цикла."
    )

    c1, c2 = st.columns(2)
    with c1:
        n_q = st.number_input("Наше число:", value=12899850, step=1, key='nq')
    with c2:
        k_q = len(str(n_q))
        st.metric("Длина / Length", k_q)

    if st.button("Найти соседей / Find neighbors", key='b4'):
        with st.spinner("Строю индекс и ищу..."):
            cyc_q, fd_q, neighbors = find_neighbors(n_q, k_q)

        if cyc_q is None:
            st.info("Неподвижная точка: соседи не определены (цикл длины 1).")
        else:
            st.write(f"**Наш цикл** ({len(cyc_q)}): "
                     + " → ".join(fmt_num(x, k_q) for x in cyc_q))
            st.write(f"**first_diffs** = {fd_q}")

            if neighbors:
                st.success(f"Найдено соседей: {len(neighbors)}")
                for nb in neighbors[:10]:
                    nb_info = enumerate_classes(k_q)[nb]
                    st.code(" → ".join(fmt_num(x, k_q) for x in nb_info['cycle']))
                if len(neighbors) > 10:
                    st.caption(f"... и ещё {len(neighbors) - 10} соседей.")
            else:
                st.warning("Для данного цикла соседей нет (уникальная сигнатура).")
