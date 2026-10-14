# -*- coding: utf-8 -*-
"""
Kaprekar Structural Research Tool
----------------------------------
Инструмент для честного вычислительного анализа процедуры Капрекара.
Все расчёты выполняются на лету, без заглушек и захардкоженных результатов.

Запуск:
pip install streamlit numpy pandas plotly
streamlit run kaprekar_tool.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
from collections import defaultdict

# =========================================================
# МОДУЛЬ 1: ЯДРО ПРОЦЕДУРЫ КАПРЕКАРА
# =========================================================

def kaprekar_step(number: int, length: int) -> int:
"""Один шаг процедуры Капрекара с сохранением разрядности (ведущие нули)."""
digits = str(number).zfill(length)
desc = ''.join(sorted(digits, reverse=True))
asc = ''.join(sorted(digits))
return int(desc) - int(asc)


def trace_full_path(seed: int, length: int, max_steps: int = 50):
"""
Полная трассировка числа: возвращает весь путь (все итерации),
индекс входа в цикл и сам цикл.
"""
path = [seed]
seen = {seed: 0}
curr = seed
for step in range(1, max_steps + 1):
nxt = kaprekar_step(curr, length)
if nxt in seen:
entry = seen[nxt]
cycle = path[entry:]
return {
'path': path,
'cycle': cycle,
'cycle_start_index': entry,
'converged': True
}
seen[nxt] = step
path.append(nxt)
curr = nxt
return {'path': path, 'cycle': [], 'cycle_start_index': -1, 'converged': False}


# =========================================================
# МОДУЛЬ 2: СТРУКТУРНЫЙ АНАЛИЗ
# =========================================================

def get_symmetry_vector(number: int, length: int):
"""Вектор S симметричных сумм: s_i = a_i + a_(k-i+1), от краёв к центру."""
digits = [int(d) for d in str(number).zfill(length)]
m = length // 2
return [digits[i] + digits[length - 1 - i] for i in range(m)]


def get_containers(number: int, length: int):
"""Контейнеры: первая пара цифр и последняя пара цифр."""
digits = str(number).zfill(length)
if length < 4:
return None, None
return digits[0:2], digits[-2:]


def compute_differences_between_iterations(path: list, length: int):
"""
Для последовательных итераций вычисляет поразрядные разности
и проверяет антисимметрию: sgn(левая половина) == -sgn(правая половина).
"""
rows = []
half = length // 2
for idx in range(len(path) - 1):
a = str(path[idx]).zfill(length)
b = str(path[idx + 1]).zfill(length)
diffs = [int(b[i]) - int(a[i]) for i in range(length)]

left = diffs[:half]
right = diffs[length - half:]
right_reversed = right[::-1]

violation = False
for l_val, r_val in zip(left, right_reversed):
if l_val != 0 and r_val != 0 and np.sign(l_val) == np.sign(r_val):
violation = True
break

mid1 = int(a[half - 1]) if half - 1 >= 0 else None
mid2 = int(a[half]) if half < length else None
center_equal = (mid1 == mid2) if (mid1 is not None and mid2 is not None) else False

rows.append({
'Итерация': idx,
'От': path[idx],
'К': path[idx + 1],
'Разности по разрядам': diffs,
'Нарушение антисимметрии': violation,
'Центральные цифры равны': center_equal
})
return pd.DataFrame(rows)


def classify_nine_pattern(number: int, length: int) -> bool:
"""
Правило 'все девятки': исходное число состоит из цифр,
все из которых равны максимальной цифре d, кроме ровно одной цифры d-1.
"""
digits = [int(x) for x in str(number).zfill(length)]
counts = defaultdict(int)
for d in digits:
counts[d] += 1
if len(counts) != 2:
return False
vals = sorted(counts.keys())
if vals[1] - vals[0] != 1:
return False
return counts[vals[0]] == 1


# =========================================================
# МОДУЛЬ 3: МАССОВОЕ СКАНИРОВАНИЕ ПО РАЗРЯДНОСТИ (5-9 знаков)
# =========================================================

def scan_full_range(length: int, sample_size: int = 20000, max_steps: int = 50,
progress_callback=None):
"""
Сканирует числа заданной длины.
length <= 6 -> полный перебор.
length >= 7 -> случайная выборка размера sample_size.
"""
low = 10 ** (length - 1)
high = 10 ** length

full_scan = length <= 6
if full_scan:
numbers = range(low, high)
total = high - low
else:
numbers = (random.randint(low, high - 1) for _ in range(sample_size))
total = sample_size

results = []
for i, n in enumerate(numbers):
digits = str(n).zfill(length)
if len(set(digits)) == 1:
continue
res = trace_full_path(n, length, max_steps=max_steps)
if res['converged'] and res['cycle']:
final = res['cycle'][-1]
s_vec = tuple(get_symmetry_vector(final, length))
results.append({
'seed': n,
'final_attractor': final,
's_vector': s_vec,
'cycle_length': len(set(res['cycle'])),
'steps_to_converge': res['cycle_start_index']
})
if progress_callback and i % max(1, total // 100) == 0:
progress_callback(min(i / total, 1.0))

return pd.DataFrame(results), full_scan, total


# =========================================================
# МОДУЛЬ 4: ПОИСК ПО КОНТЕЙНЕРАМ
# =========================================================

def search_by_containers(length: int, start_container: str, end_container: str,
sample_limit: int = 50000):
"""
Ищет числа заданной длины с заданной начальной и конечной парой цифр.
"""
middle_len = length - 4
found = []

if middle_len < 0:
return found

if middle_len == 0:
full = start_container + end_container
return [int(full)]

if middle_len <= 5:
for mid in range(10 ** middle_len):
mid_str = str(mid).zfill(middle_len)
found.append(int(start_container + mid_str + end_container))
else:
seen = set()
attempts = 0
while len(found) < sample_limit and attempts < sample_limit * 3:
mid_str = ''.join(str(random.randint(0, 9)) for _ in range(middle_len))
if mid_str not in seen:
seen.add(mid_str)
found.append(int(start_container + mid_str + end_container))
attempts += 1

return found


def analyze_container_group(numbers: list, length: int, max_steps: int = 50):
"""Прогоняет список чисел через процедуру и возвращает сводку аттракторов."""
rows = []
for n in numbers:
digits = str(n).zfill(length)
if len(set(digits)) == 1:
continue
res = trace_full_path(n, length, max_steps=max_steps)
if res['converged'] and res['cycle']:
rows.append({
'seed': n,
'final_attractor': res['cycle'][-1],
's_vector': tuple(get_symmetry_vector(res['cycle'][-1], length))
})
return pd.DataFrame(rows)


# =========================================================
# МОДУЛЬ 5: 3D ВИЗУАЛИЗАЦИЯ — СПИРАЛЬ ФАЗОВОГО ПРОСТРАНСТВА
# =========================================================

def build_3d_phase_spiral(path: list, length: int):
"""
3D-спираль траектории Капрекара.
Угол: позиция пары (от центра к краю).
Радиус: значение симметричной суммы s_i (нормировано вокруг 9).
Высота (Z): номер итерации.
"""
fig = go.Figure()
m = length // 2

all_x, all_y, all_z, all_colors, all_text = [], [], [], [], []

for step_idx, number in enumerate(path):
s_vector = get_symmetry_vector(number, length)
for pair_idx, s_val in enumerate(s_vector):
# угол: от центра (pair_idx=0 условно центр) к краю
angle = 2 * np.pi * pair_idx / max(m, 1)
radius = s_val # сама сумма как радиус
x = radius * np.cos(angle)
y = radius * np.sin(angle)
z = step_idx

all_x.append(x)
all_y.append(y)
all_z.append(z)
all_text.append(f"Число: {number}<br>Пара #{pair_idx+1}<br>Сумма S: {s_val}<br>Итерация: {step_idx}")

# Цветовая индикация коридора [8,10] -- эмпирически найденного диапазона
if 8 <= s_val <= 10:
all_colors.append('rgb(30,120,220)') # внутри коридора
else:
all_colors.append('rgb(220,60,40)') # выход за коридор

fig.add_trace(go.Scatter3d(
x=all_x, y=all_y, z=all_z,
mode='markers+lines',
marker=dict(size=5, color=all_colors),
line=dict(color='rgba(150,150,150,0.4)', width=2),
text=all_text,
hoverinfo='text',
name='Траектория'
))

fig.update_layout(
title="Фазовая спираль: угол=позиция пары от центра, радиус=сумма S, высота=итерация",
scene=dict(
xaxis_title='X (проекция суммы S)',
yaxis_title='Y (проекция суммы S)',
zaxis_title='Итерация'
),
height=700,
margin=dict(l=0, r=0, t=40, b=0)
)
return fig


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="Kaprekar Structural Research Tool", layout="wide")
st.title("Исследовательский инструмент процедуры Капрекара")
st.caption("Все вычисления выполняются в реальном времени на основе введённых данных. "
"Диапазон 2-4 знака не включён в сканирование, так как их поведение "
"полностью детерминировано и тривиально вычислимо на бумаге.")

tab1, tab2, tab3, tab4 = st.tabs([
"Карточка числа",
"Массовое сканирование (5-9 знаков)",
"Поиск по контейнерам",
"Сравнение групп чисел"
])

# ---------------------------------------------------------
# TAB 1: КАРТОЧКА ЧИСЛА
# ---------------------------------------------------------
with tab1:
st.header("Полная карточка числа")
st.write("Отображает весь путь числа (все итерации, а не только цикл), "
"вектор симметричных сумм S на каждом шаге, разности между итерациями "
"и проверку антисимметрии.")

col_in1, col_in2 = st.columns(2)
with col_in1:
card_number = st.number_input("Введите число:", min_value=10, value=83952, step=1)
with col_in2:
card_length = st.number_input("Разрядность (с учётом ведущих нулей, если нужно):",
min_value=5, max_value=9,
value=len(str(card_number)), step=1)

if st.button("Построить карточку"):
if len(str(card_number)) > card_length:
st.error("Введённое число длиннее указанной разрядности.")
else:
result = trace_full_path(card_number, card_length, max_steps=60)

colA, colB, colC = st.columns(3)
colA.metric("Длина полного пути", len(result['path']))
colB.metric("Шаг входа в цикл", result['cycle_start_index'] if result['converged'] else "не найден")
colC.metric("Длина цикла", len(set(result['cycle'])) if result['converged'] else "—")

st.subheader("Полный путь (все итерации)")
path_df = pd.DataFrame({
'Итерация': range(len(result['path'])),
'Число': result['path'],
'В цикле': [
(i >= result['cycle_start_index']) if result['converged'] else False
for i in range(len(result['path']))
]
})
st.dataframe(path_df, use_container_width=True)

st.subheader("Вектор симметричных сумм S по каждой итерации")
s_rows = []
for i, num in enumerate(result['path']):
s_vec = get_symmetry_vector(num, card_length)
s_rows.append({'Итерация': i, 'Число': num, 'Вектор S': s_vec})
s_df = pd.DataFrame(s_rows)
st.dataframe(s_df, use_container_width=True)

# График вектора S с коридором [8,10]
st.subheader("Динамика сумм S с визуальным коридором [8,10]")
fig_s = go.Figure()
for i, num in enumerate(result['path']):
s_vec = get_symmetry_vector(num, card_length)
fig_s.add_trace(go.Scatter(
x=list(range(len(s_vec))), y=s_vec,
mode='lines+markers', name=f'Итер. {i} ({num})'
))
fig_s.add_hrect(y0=8, y1=10, fillcolor="LightBlue", opacity=0.3,
line_width=0, annotation_text="Эмпирический коридор [8,10]")
fig_s.update_layout(xaxis_title="Номер пары (от края к центру)",
yaxis_title="Сумма симметричной пары S",
height=450)
st.plotly_chart(fig_s, use_container_width=True)

st.subheader("Разности между итерациями и проверка антисимметрии")
diff_df = compute_differences_between_iterations(result['path'], card_length)
if not diff_df.empty:
def highlight_violation(row):
color = 'background-color: #ffcccc' if row['Нарушение антисимметрии'] else ''
return [color] * len(row)
st.dataframe(diff_df.style.apply(highlight_violation, axis=1), use_container_width=True)

violations = diff_df[diff_df['Нарушение антисимметрии']]
if not violations.empty:
st.warning(f"Обнаружено нарушений антисимметрии: {len(violations)}. "
f"Проверьте колонку 'Центральные цифры равны' — согласно гипотезе, "
f"нарушение связано с совпадением центральных цифр исходного числа на данном шаге.")
else:
st.success("Нарушений антисимметрии на данном пути не обнаружено.")

st.subheader("Контейнеры на каждой итерации")
cont_rows = []
for i, num in enumerate(result['path']):
start_c, end_c = get_containers(num, card_length)
cont_rows.append({'Итерация': i, 'Число': num,
'Начальный контейнер': start_c, 'Конечный контейнер': end_c})
st.dataframe(pd.DataFrame(cont_rows), use_container_width=True)

st.subheader("Проверка правила 'все девятки' для исходного числа")
is_nine_pattern = classify_nine_pattern(card_number, card_length)
if is_nine_pattern:
st.info("Исходное число соответствует правилу: все цифры одинаковы и максимальны, "
"кроме одной цифры, меньшей на 1. Ожидается сходимость к числу из девяток.")
else:
st.write("Исходное число не соответствует правилу 'все девятки'.")

st.subheader("3D фазовая спираль траектории")
st.caption("Угол — позиция симметричной пары от центра к краю, "
"радиус — значение суммы S, высота — номер итерации. "
"Синий цвет — значение внутри эмпирического коридора [8,10], красный — вне его.")
fig3d = build_3d_phase_spiral(result['path'], card_length)
st.plotly_chart(fig3d, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: МАССОВОЕ СКАНИРОВАНИЕ ПО РАЗРЯДНОСТИ
# ---------------------------------------------------------
with tab2:
st.header("Массовое сканирование чисел заданной разрядности")
st.write("Для 5-6 знаков выполняется полный перебор всех чисел. "
"Для 7-9 знаков — случайная репрезентативная выборка (полный перебор "
"для этих разрядностей вычислительно нецелесообразен в данной среде).")

scan_length = st.selectbox("Разрядность для сканирования:", [5, 6, 7, 8, 9], index=1)

if scan_length >= 7:
sample_size = st.slider("Размер случайной выборки:", 1000, 50000, 20000, step=1000)
else:
sample_size = None
total_possible = 9 * (10 ** (scan_length - 1))
st.write(f"Будет выполнен полный перебор: примерно {total_possible:,} чисел.")

if st.button("Запустить сканирование"):
progress_bar = st.progress(0.0)
def update_progress(p):
progress_bar.progress(p)

with st.spinner("Выполняется сканирование..."):
scan_df, was_full_scan, total_scanned = scan_full_range(
scan_length,
sample_size=sample_size if sample_size else 20000,
max_steps=60,
progress_callback=update_progress
)
progress_bar.progress(1.0)

scan_type_label = "ПОЛНЫЙ ПЕРЕБОР" if was_full_scan else f"ВЫБОРКА (n={total_scanned})"
st.write(f"Тип сканирования: **{scan_type_label}**. Обработано результатов: {len(scan_df)}")

if not scan_df.empty:
st.subheader("Распределение числовых аттракторов")
attractor_counts = scan_df['final_attractor'].value_counts().reset_index()
attractor_counts.columns = ['Аттрактор', 'Количество']
st.dataframe(attractor_counts, use_container_width=True)

st.subheader("Распределение вариантов вектора S (ключевая проверка гипотезы о пике на 6 и 8 знаках)")
s_vector_counts = scan_df['s_vector'].value_counts().reset_index()
s_vector_counts.columns = ['Вектор S', 'Количество']
st.dataframe(s_vector_counts, use_container_width=True)
st.metric("Количество РАЗЛИЧНЫХ вариантов вектора S", scan_df['s_vector'].nunique())

fig_bar = go.Figure(go.Bar(
x=[str(v) for v in s_vector_counts['Вектор S']],
y=s_vector_counts['Количество']
))
fig_bar.update_layout(title=f"Варианты вектора S для {scan_length}-значных чисел",
xaxis_title="Вектор S", yaxis_title="Частота",
height=400)
st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Распределение количества шагов до сходимости")
fig_hist = go.Figure(go.Histogram(x=scan_df['steps_to_converge']))
fig_hist.update_layout(xaxis_title="Шагов до входа в цикл", yaxis_title="Частота", height=350)
st.plotly_chart(fig_hist, use_container_width=True)

st.session_state['last_scan_df'] = scan_df
st.session_state['last_scan_length'] = scan_length
else:
st.warning("Не найдено сходящихся результатов в выбранной выборке.")

# ---------------------------------------------------------
# TAB 3: ПОИСК ПО КОНТЕЙНЕРАМ
# ---------------------------------------------------------
with tab3:
st.header("Поиск чисел по заданным контейнерам")
st.write("Контейнер — пара цифр на краю числа. Введите начальный и конечный контейнер, "
"программа найдёт числа заданной длины с такими границами и покажет, "
"к каким аттракторам они сходятся.")

col_c1, col_c2, col_c3 = st.columns(3)
with col_c1:
cont_length = st.selectbox("Разрядность:", [5, 6, 7, 8, 9], index=1, key='cont_len')
with col_c2:
start_container_input = st.text_input("Начальный контейнер (2 цифры):", "86")
with col_c3:
end_container_input = st.text_input("Конечный контейнер (2 цифры):", "32")

valid_input = (len(start_container_input) == 2 and start_container_input.isdigit() and
len(end_container_input) == 2 and end_container_input.isdigit())

if not valid_input:
st.warning("Введите корректные двузначные контейнеры (например, '86' и '32').")

if cont_length >= 8:
st.caption("Для 8-9 знаков средняя часть числа формируется случайной выборкой, "
"а не полным перебором, из соображений производительности.")

if st.button("Найти числа с такими контейнерами") and valid_input:
with st.spinner("Поиск и анализ чисел..."):
candidates = search_by_containers(cont_length, start_container_input, end_container_input,
sample_limit=30000)
group_df = analyze_container_group(candidates, cont_length, max_steps=60)

st.write(f"Сгенерировано кандидатов: {len(candidates)}. "
f"Сошлось к устойчивому циклу/точке: {len(group_df)}")

if not group_df.empty:
st.subheader("Распределение аттракторов внутри данной группы контейнеров")
attractor_dist = group_df['final_attractor'].value_counts().reset_index()
attractor_dist.columns = ['Аттрактор', 'Количество']
st.dataframe(attractor_dist, use_container_width=True)

dominant_share = attractor_dist['Количество'].iloc[0] / len(group_df) * 100
st.metric("Доля чисел, сошедшихся к доминирующему аттрактору", f"{dominant_share:.1f}%")

if dominant_share > 70:
st.success("Высокая концентрация на одном аттракторе — данный контейнер, "
"по всей видимости, статистически связан с конкретным циклом/точкой.")
else:
st.info("Числа с данным контейнером расходятся по разным аттракторам — "
"жёсткой связи контейнера с единственным результатом не обнаружено.")

st.subheader("Распределение вектора S внутри найденной группы")
s_dist = group_df['s_vector'].value_counts().reset_index()
s_dist.columns = ['Вектор S', 'Количество']
st.dataframe(s_dist, use_container_width=True)

st.subheader("Полная таблица результатов")
st.dataframe(group_df, use_container_width=True)
else:
st.warning("Ни одно число из сгенерированных кандидатов не сошлось к устойчивому результату "
"за отведённое число шагов.")

# ---------------------------------------------------------
# TAB 4: СРАВНЕНИЕ ГРУПП ЧИСЕЛ
# ---------------------------------------------------------
with tab4:
st.header("Сравнительный анализ произвольного набора чисел")
st.write("Введите несколько чисел одной разрядности через запятую, чтобы сравнить "
"их аттракторы, векторы S, контейнеры и скорость сходимости.")

col_cmp1, col_cmp2 = st.columns(2)
with col_cmp1:
compare_length = st.selectbox("Разрядность сравниваемых чисел:", [5, 6, 7, 8, 9], index=1, key='cmp_len')
with col_cmp2:
compare_input = st.text_input("Числа через запятую:", "83952, 71973, 74943, 62964")

if st.button("Сравнить"):
raw_items = [x.strip() for x in compare_input.split(",") if x.strip()]
valid_numbers = []
invalid_items = []
for item in raw_items:
if item.isdigit() and len(item) <= compare_length:
valid_numbers.append(int(item))
else:
invalid_items.append(item)

if invalid_items:
st.warning(f"Пропущены некорректные значения (не число или длиннее заданной разрядности): {invalid_items}")

if valid_numbers:
comp_rows = []
all_paths = {}
for n in valid_numbers:
res = trace_full_path(n, compare_length, max_steps=60)
all_paths[n] = res
start_c, end_c = get_containers(n, compare_length)
comp_rows.append({
'Исходное число': n,
'Итоговый аттрактор': res['cycle'][-1] if res['converged'] else "не сошлось",
'Шагов до цикла': res['cycle_start_index'] if res['converged'] else "—",
'Длина цикла': len(set(res['cycle'])) if res['converged'] else "—",
'Вектор S аттрактора': tuple(get_symmetry_vector(res['cycle'][-1], compare_length)) if res['converged'] else None,
'Начальный контейнер': start_c,
'Конечный контейнер': end_c
})

comp_df = pd.DataFrame(comp_rows)
st.subheader("Сводная сравнительная таблица")
st.dataframe(comp_df, use_container_width=True)

st.subheader("Группировка по совпадающему итоговому аттрактору")
grouped = comp_df.groupby('Итоговый аттрактор')['Исходное число'].apply(list).reset_index()
grouped.columns = ['Аттрактор', 'Числа, сходящиеся к нему']
st.dataframe(grouped, use_container_width=True)

st.subheader("Сравнение скорости сходимости (шаги до цикла)")
fig_cmp = go.Figure(go.Bar(
x=[str(n) for n in comp_df['Исходное число']],
y=[s if isinstance(s, int) else 0 for s in comp_df['Шагов до цикла']]
))
fig_cmp.update_layout(xaxis_title="Исходное число", yaxis_title="Шагов до входа в цикл", height=400)
st.plotly_chart(fig_cmp, use_container_width=True)

st.subheader("Наложенные траектории вектора S (первая симметричная сумма) по шагам")
fig_overlay = go.Figure()
for n in valid_numbers:
path = all_paths[n]['path']
first_s_values = [get_symmetry_vector(num, compare_length)[0] for num in path]
fig_overlay.add_trace(go.Scatter(
x=list(range(len(first_s_values))), y=first_s_values,
mode='lines+markers', name=str(n)
))
fig_overlay.add_hrect(y0=8, y1=10, fillcolor="LightBlue", opacity=0.25, line_width=0)
fig_overlay.update_layout(xaxis_title="Итерация", yaxis_title="Первая сумма S (крайняя пара)",
height=450)
st.plotly_chart(fig_overlay, use_container_width=True)
else:
st.error("Не найдено ни одного корректного числа для сравнения.")
