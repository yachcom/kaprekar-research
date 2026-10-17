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
        st.code(" -> ".join(str(x) for x in cyc) + " -> " + str(cyc[0]))
    else:
        st.subheader("Неподвижная точка / Fixed point")
        st.code(str(cyc[0]))

    st.subheader("Траектория / Trajectory")
    st.markdown(trajectory_table_html(path, k0, ci, cyc), unsafe_allow_html=True)

    st.subheader("Таблица разностей / Difference table")
    st.markdown(diff_table_html(path, k0), unsafe_allow_html=True)

    st.subheader("3D-Динамика / 3D Dynamics")
    
    view_mode = st.radio(
        "Проекция / Projection:",
        ("Поверхность (Surface)", "Только линии (Lines)", "Спираль (Helix)", "Замыкание цикла (Torus)", "Расширение поля (Neighbors)"),
        horizontal=True, key='mode3d'
    )

    fig3 = go.Figure()

    # Считаем разности для всей траектории
    Z_diffs = np.array([digits_of(path[t+1], k0) - digits_of(path[t], k0) for t in range(len(path)-1)])
    x_positions = np.arange(k0) - (k0 - 1) / 2

    # --- РЕЖИМ 1: ПОВЕРХНОСТЬ ---
    if view_mode == "Поверхность (Surface)":
        st.caption("Седловидная поверхность антисимметрии Δi = -Δ(k+1-i).")
        fig3.add_trace(go.Surface(
            x=x_positions, y=np.arange(len(Z_diffs)), z=Z_diffs,
            colorscale='RdBu', cmid=0, opacity=0.9,
            colorbar=dict(title="Разность (Δ)")
        ))
        fig3.update_layout(scene=dict(xaxis_title='Позиция', yaxis_title='Итерация', zaxis_title='Δ'))

    # --- РЕЖИМ 2: ТОЛЬКО ЛИНИИ ---
    elif view_mode == "Только линии (Lines)":
        st.caption("Траектории отдельных позиций. Видно, как каждая цифра 'прыгает'.")
        for t in range(len(Z_diffs)):
            colors = ['#d04040' if v < 0 else ('#30a030' if v > 0 else '#808080') for v in Z_diffs[t]]
            fig3.add_trace(go.Scatter3d(
                x=x_positions, y=[t]*k0, z=Z_diffs[t],
                mode='lines+markers',
                line=dict(color='black', width=3),
                marker=dict(size=5, color=colors),
                showlegend=False, hoverinfo='skip'
            ))
        for i in range(k0):
            z_vals = [Z_diffs[t][i] for t in range(len(Z_diffs))]
            fig3.add_trace(go.Scatter3d(
                x=[x_positions[i]]*len(Z_diffs), y=np.arange(len(Z_diffs)), z=z_vals,
                mode='lines', line=dict(color='gray', width=2, dash='dot'),
                showlegend=False, hoverinfo='skip'
            ))
        fig3.update_layout(scene=dict(xaxis_title='Позиция', yaxis_title='Итерация', zaxis_title='Δ'))

    # --- РЕЖИМ 3: СПИРАЛЬ (HELIX) ---
    elif view_mode == "Спираль (Helix)":
        st.caption("Ось итераций закручена в спираль. Разности наматываются на цилиндр.")
        
        # Параметры спирали
        turns = 2  # сколько оборотов на весь путь
        theta = np.linspace(0, turns * 2 * np.pi, len(Z_diffs))
        radius = 6
        
        for i in range(k0):
            z_vals = [Z_diffs[t][i] for t in range(len(Z_diffs))]
            # Добавляем разность к радиусу, чтобы она "выпирала"
            r_vals = radius + z_vals 
            
            x_helix = r_vals * np.cos(theta)
            y_helix = r_vals * np.sin(theta)
            
            fig3.add_trace(go.Scatter3d(
                x=x_helix, y=y_helix, z=np.arange(len(Z_diffs)),
                mode='lines+markers',
                line=dict(width=4),
                marker=dict(size=4, color=['#d04040' if v < 0 else '#30a030' for v in z_vals]),
                name=f"Позиция {i+1}"
            ))
            
        fig3.update_layout(scene=dict(
            xaxis_title='X (Спираль)', yaxis_title='Y (Спираль)', zaxis_title='Итерация',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
        ))

    # --- РЕЖИМ 4: ТОР (TORUS) ---
    elif view_mode == "Замыкание цикла (Torus)":
        if len(cyc) == 1:
            st.warning("Неподвижная точка. Цикл не замыкается в Тор.")
        else:
            st.caption("Показан только цикл. Время свернуто в кольцо, образуя истинный тор.")
            cycle_path = cyc + [cyc[0]]
            Z_cyc = np.array([digits_of(cycle_path[t+1], k0) - digits_of(cycle_path[t], k0) for t in range(len(cycle_path)-1)])
            
            # Формируем сетку для тора
            T_len = len(Z_cyc)
            theta = np.linspace(0, 2*np.pi, T_len)
            
            # Тороидальные координаты
            R = 10 # радиус тора
            # X = (R + Delta) * cos(theta)
            # Y = (R + Delta) * sin(theta)
            # Z = позиция цифры
            
            X_torus = np.zeros((T_len, k0))
            Y_torus = np.zeros((T_len, k0))
            Z_torus = np.zeros((T_len, k0))
            
            for t in range(T_len):
                for i in range(k0):
                    # Разность добавляем к радиусу трубы
                    r = R + Z_cyc[t][i] 
                    X_torus[t, i] = r * np.cos(theta[t])
                    Y_torus[t, i] = r * np.sin(theta[t])
                    Z_torus[t, i] = x_positions[i] # Позиция цифры вдоль оси Z тора
            
            # Замыкаем поверхность (добавляем первую строку в конец)
            X_torus = np.vstack([X_torus, X_torus[0]])
            Y_torus = np.vstack([Y_torus, Y_torus[0]])
            Z_torus = np.vstack([Z_torus, Z_torus[0]])
            Z_color = np.vstack([Z_cyc, Z_cyc[0]])

            fig3.add_trace(go.Surface(
                x=X_torus, y=Y_torus, z=Z_torus,
                surfacecolor=Z_color,
                colorscale='RdBu', cmid=0,
                colorbar=dict(title="Δ")
            ))
            
            fig3.update_layout(scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Позиция цифры',
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
            ))

    # --- РЕЖИМ 5: РАСШИРЕНИЕ ПОЛЯ (NEIGHBORS) ---
    elif view_mode == "Расширение поля (Neighbors)":
        st.caption(
            "Алгоритм ищет числа, чья **левая** колонка разностей стыкуется с **правой** колонкой текущего числа. "
            "Соединяя их, мы строим непрерывное поле в плоскости."
        )
        
        # Мы будем искать число, у которого первая позиция (x_positions[0]) ведет себя так же,
        # как последняя позиция (x_positions[-1]) нашего исходного числа.
        
        # Собираем правую колонку текущего числа (последняя позиция)
        target_col = [Z_diffs[t][-1] for t in range(len(Z_diffs))]
        
        st.write(f"Ищем число, у которого крайняя левая разность образует последовательность: `{target_col[:5]}...`")
        
        # Простой поиск (для примера переберем 5000 чисел)
        import random
        random.seed(42)
        found_n = None
        
        for _ in range(5000):
            # Генерируем случайное число той же длины, начинающееся с правильных цифр
            n_cand = random.randint(10**(k0-1), 10**k0 - 1)
            p_cand, _, _ = trajectory(n_cand, k0)
            # Проверяем только первые 5 шагов
            z_cand = [digits_of(p_cand[t+1], k0) - digits_of(p_cand[t], k0) for t in range(min(5, len(p_cand)-1))]
            
            # Совпадает ли первая позиция с правой колонкой искомого?
            if [z[0] for z in z_cand] == target_col[:5]:
                found_n = n_cand
                break
                
        if found_n:
            st.success(f"Найдено число-сосед: **{found_n}**. Оно идеально продолжает поле влево!")
            p2, _, _ = trajectory(found_n, k0)
            Z2 = np.array([digits_of(p2[t+1], k0) - digits_of(p2[t], k0) for t in range(len(p2)-1)])
            
            # Сдвигаем координаты второго числа влево
            offset = k0 - 1 # сдвиг на ширину числа
            
            # Исходное число
            fig3.add_trace(go.Surface(
                x=x_positions, y=np.arange(len(Z_diffs)), z=Z_diffs,
                colorscale='RdBu', cmid=0, opacity=0.9, showscale=False
            ))
            # Сосед
            fig3.add_trace(go.Surface(
                x=x_positions - offset, y=np.arange(len(Z2)), z=Z2,
                colorscale='RdBu', cmid=0, opacity=0.9, showscale=False
            ))
            
            st.info("Вы видите, как правый край левой поверхности стыкуется с левым краем правой.")
        else:
            st.warning("Не удалось найти идеального соседа за 5000 попыток. (Алгоритм поиска можно усложнить).")
            
        fig3.update_layout(scene=dict(xaxis_title='Позиция', yaxis_title='Итерация', zaxis_title='Δ'))

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
            "**Почему пирамида?** Точки удовлетворяют d1 >= d2 >= d3 >= 0. "
            "Это математическое следствие того, что d - это разности убывающих цифр. "
            "Для k=3 (только d1) - линия; для k=4,5 (d1, d2) - плоский треугольник; для k>=6 (d1, d2, d3) - пирамида."
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
        st.plotly_chart(fig, use_container_width=True)
