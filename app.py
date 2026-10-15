# -*- coding: utf-8 -*-
"""
Kaprekar Dynamics — единый файл
streamlit run kaprekar_dynamics.py
"""
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
    return np.array([int(s[i]) - int(s[k-1-i]) for i in range(k//2)])

def _ns(s):
    return s if s <= 10 else s - 9

def S_of(n, k):
    d = digits_of(n, k); m = k // 2
    if k % 2 == 0:
        return tuple(_ns(d[m-1-i] + d[m+i]) for i in range(m))
    return tuple([int(d[m])] + [_ns(d[m-i] + d[m+i]) for i in range(1, m+1)])


# ================= ОРБИТЫ =================
def orbit_fig(S, title=""):
    L = len(S)
    fig = go.Figure()
    for r in range(1, L):
        th = np.linspace(0, 2*np.pi, 120)
        fig.add_trace(go.Scatter(
            x=r*np.cos(th), y=r*np.sin(th), mode='lines',
            line=dict(color='rgba(160,160,160,0.5)', width=1),
            showlegend=False, hoverinfo='skip'))
    for i, v in enumerate(S):
        col = '#3070c0' if v == 8 else ('#d04040' if v == 10 else '#888888')
        fig.add_trace(go.Scatter(
            x=[0], y=[i], mode='markers+text',
            marker=dict(size=26, color=col, line=dict(color='black', width=1.5)),
            text=[str(v)], textposition='middle center',
            textfont=dict(color='white', size=13),
            showlegend=False, hoverinfo='text',
            hovertext=f"орбита {i}: {v}"))
    fig.update_layout(
        xaxis=dict(visible=False, scaleanchor='y', range=[-L-0.5, L+0.5]),
        yaxis=dict(visible=False, range=[-1, L]),
        height=270, margin=dict(l=5, r=5, t=32, b=5),
        title=dict(text=title, font=dict(size=11)), showlegend=False)
    return fig


# ================= ТАБЛИЦА РАЗНОСТЕЙ =================
def diff_table_html(path, k):
    html = ('<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:13px;">'
            '<tr style="background:#f0f0f0;">'
            '<th style="padding:6px;border:1px solid #ccc;">Итерация<br>Iter</th>'
            '<th style="padding:6px;border:1px solid #ccc;">От<br>From</th>'
            '<th style="padding:6px;border:1px solid #ccc;">К<br>To</th>'
            '<th style="padding:6px;border:1px solid #ccc;">Разности цифр<br>Digit differences</th>'
            '<th style="padding:6px;border:1px solid #ccc;">Нарушение антисимметрии<br>Antisymmetry violation</th>'
            '<th style="padding:6px;border:1px solid #ccc;">Центр. цифры равны<br>Center digits equal</th>'
            '</tr>')
    for t in range(len(path)-1):
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
                 f'{"⚠️" if viol else "✓"}</td>'
                 f'<td style="padding:4px;border:1px solid #ccc;text-align:center;">'
                 f'{"✓" if ce else "—"}</td></tr>')
    return html + '</table>'


# ================= КЭШ ПЕРЕБОРА =================
@st.cache_data(show_spinner=False)
def enumerate_classes(k):
    """
    Полный перебор классов эквивалентности по мультимножеству цифр.
    Их число C(k+9, 9). Для k=9 — 48620, для k=10 — 92378.
    """
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

t1, t2, t3 = st.tabs([
    "1. Стабилизация и аномалии / Stabilization & anomalies",
    "2. Пространство n-значных / Space of n-digit numbers",
    "3. Циклы и неподвижные / Cycles and fixed points",
])


# ---------------- ВКЛАДКА 1 ----------------
with t1:
    st.header("Стабилизация и аномалии / Stabilization and anomalies")
    n0 = st.number_input("Число / Number:", value=12899850, step=1)
    k0 = len(str(n0))
    st.caption(f"Автоопределённая длина / Detected length: **{k0}**")

    if st.button("Построить / Build", key='b1'):
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
        rows = []
        for i, n in enumerate(path):
            stage = "старт" if i == 0 else (
                "вход в цикл" if i == ci else (
                    "в цикле" if i > ci else "переход"))
            rows.append({
                "Итерация / Iter": i,
                "Число / Number": n,
                "S (центр→край) / S (center→edge)": str(S_of(n, k0)),
                "d-последовательность / d-sequence": str(tuple(int(x) for x in d_vec(n, k0))),
                "Стадия / Stage": stage,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("Орбиты S: как из центра собираются суммы / Orbits of S")
        st.caption(
            "Концентрические окружности — орбиты. Внутри — суммы от центра числа, снаружи — "
            "крайние пары. Цвет: синий = 8, красный = 10, серый = 9. "
            "На устойчивом состоянии видно: 8 у центра, 10 на краю.\n\n"
            "Concentric rings are orbits. Innermost = sums near center, outermost = extreme pair. "
            "Color: blue = 8, red = 10, gray = 9."
        )
        for start in range(0, len(path), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                idx = start + j
                if idx >= len(path):
                    break
                n = path[idx]
                tag = " [Ц]" if idx >= ci else ""
                with col:
                    st.plotly_chart(
                        orbit_fig(S_of(n, k0), f"итер.{idx}: {n}{tag}"),
                        use_container_width=True)

        st.subheader("3D-антисимметрия разностей / 3D antisymmetry of differences")
        st.caption(
            "Каждая точка: (Δᵢ + Δ_{k+1−i}) для симметричной пары позиций, на данной итерации. "
            "z = 0 → антисимметрия выполнена. z ≠ 0 → нарушение.\n\n"
            "Each point: sum of symmetric differences. z = 0 → antisymmetry holds. z ≠ 0 → violation."
        )
        xs, ys, zs, colors = [], [], [], []
        for t in range(len(path)-1):
            a = digits_of(path[t], k0)
            b = digits_of(path[t+1], k0)
            diffs = b - a
            for i in range(k0 // 2):
                s = int(diffs[i] + diffs[k0-1-i])
                xs.append(i)
                ys.append(t)
                zs.append(s)
                colors.append('#d04040' if s != 0 else '#3070c0')
        fig3 = go.Figure(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='markers',
            marker=dict(size=7, color=colors, opacity=0.85),
            text=[f"пара {x}, итер {y}, сумма Δ={z}" for x, y, z in zip(xs, ys, zs)],
            hoverinfo='text'))
        fig3.update_layout(
            scene=dict(xaxis_title='Позиция пары от центра / Pair pos',
                       yaxis_title='Итерация / Iter',
                       zaxis_title='Δᵢ + Δ_{k+1−i}'),
            height=600, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Таблица разностей / Difference table")
        st.markdown(diff_table_html(path, k0), unsafe_allow_html=True)


# ---------------- ВКЛАДКА 2 ----------------
with t2:
    st.header("Пространство n-значных чисел / Space of n-digit numbers")
    st.caption(
        "Каждая точка — **класс эквивалентности по мультимножеству цифр**. "
        "K(n) зависит только от мультимножества, поэтому перебираются все классы, а не все 10^k чисел.\n\n"
        "Число классов = C(k+9, 9). Для k=9: **C(18, 9) = 48 620**. Для k=10: **C(19, 9) = 92 378**. "
        "Один класс — все числа с одинаковым набором цифр (например, класс {0,1,2,3} содержит 4! = 24 числа).\n\n"
        "Each point is an **equivalence class by multiset of digits**. The number of classes is C(k+9, 9)."
    )

    c1, c2 = st.columns(2)
    with c1:
        k2 = st.slider("Длина числа / Number length:", 3, 10, 6, key='s3k')
        n_classes = comb(k2 + 9, 9)
        st.info(f"Классов при полном переборе: **{n_classes:,}** / Classes: **{n_classes:,}**")
    with c2:
        hl = st.number_input(
            "Трассируемое число / Trace number (0 = не показывать / 0 = hide):",
            value=0, step=1, key='s3h')
        if hl > 0:
            hl_str = str(hl).zfill(k2)
            st.caption(f"Будет отрисован путь числа **{hl_str}** в d-пространстве. "
                       f"Если цифр меньше {k2}, число дополняется ведущими нулями.")

    if st.button("Построить / Build", key='b3'):
        with st.spinner("Перебор классов / Enumerating classes..."):
            cycles = enumerate_classes(k2)
            attn = {c: i for i, c in enumerate(cycles.keys())}

            xs, ys, zs, colors, texts = [], [], [], [], []
            for combo in combinations_with_replacement(range(10), k2):
                if len(set(combo)) == 1:
                    continue
                n = int(''.join(str(x) for x in sorted(combo, reverse=True)))
                path, ci, cyc = trajectory(n, k2)
                d = list(d_vec(n, k2)[:3])
                d += [0] * (3 - len(d))
                canon = tuple(sorted(cyc))[0]
                xs.append(d[0]); ys.append(d[1]); zs.append(d[2])
                colors.append(attn[canon])
                texts.append(f"n={n}<br>цикл={canon}<br>длина={len(cyc)}")

        fig = go.Figure(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='markers',
            marker=dict(size=3, color=colors, colorscale='Turbo',
                        opacity=0.55, colorbar=dict(title="аттрактор")),
            text=texts, hoverinfo='text'))

        if hl > 0 and len(str(hl)) <= k2:
            path, ci, cyc = trajectory(hl, k2)
            dpath = [list(d_vec(n, k2)[:3]) + [0]*(3-len(d_vec(n, k2)[:3])) for n in path]
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in dpath], y=[p[1] for p in dpath], z=[p[2] for p in dpath],
                mode='lines+markers',
                line=dict(color='black', width=6),
                marker=dict(size=6, color='black'),
                name=f"путь {hl}"))

        fig.update_layout(
            height=750,
            scene=dict(xaxis_title='d₁', yaxis_title='d₂', zaxis_title='d₃'),
            margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Почему пирамида?** Точки удовлетворяют d₁ ≥ d₂ ≥ d₃ ≥ 0 — это математическое "
            "ограничение на симметричные разности, а не артефакт кода. Для k=3 (только d₁) — "
            "линия; для k=4,5 (d₁,d₂) — плоский треугольник; для k≥6 (d₁,d₂,d₃) — пирамида.\n\n"
            "**Why a pyramid?** The constraint d₁ ≥ d₂ ≥ d₃ ≥ 0 is mathematical, not a code artefact."
        )

        st.subheader("Аттракторы / Attractors")
        tbl = pd.DataFrame([
            {"Цикл / Cycle": " → ".join(str(x) for x in info['cycle']),
             "Длина / Length": len(info['cycle']),
             "Классов / Classes": info['count'],
             "Доля / Share": f"{info['count'] / sum(v['count'] for v in cycles.values()) * 100:.2f}%"}
            for info in sorted(cycles.values(), key=lambda x: -x['count'])
        ])
        st.dataframe(tbl, use_container_width=True)


# ---------------- ВКЛАДКА 3 ----------------
with t3:
    st.header("Циклы и неподвижные точки / Cycles and fixed points")
    st.caption("Полный перебор классов эквивалентности. Циклы выписаны целиком.")
    k3 = st.slider("Длина числа / Number length:", 3, 10, 6, key='s4k')

    if st.button("Перебрать / Enumerate", key='b4'):
        with st.spinner("Перебор / Enumerating..."):
            cycles = enumerate_classes(k3)
            total = sum(v['count'] for v in cycles.values())
            rows = []
            for info in sorted(cycles.values(), key=lambda x: -x['count']):
                cyc = info['cycle']
                rows.append({
                    "Тип / Type": "неподвижная / fixed" if len(cyc) == 1
                                  else f"цикл длины {len(cyc)} / {len(cyc)}-cycle",
                    "Цикл (полностью) / Cycle (full)":
                        " → ".join(str(x) for x in cyc) + (" → " + str(cyc[0]) if len(cyc) > 1 else ""),
                    "Классов / Classes": info['count'],
                    "Доля / Share": f"{info['count'] / total * 100:.2f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
