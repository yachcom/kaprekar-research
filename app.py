# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
from collections import defaultdict

# =========================================================
# MODULE 1: KAPREKAR CORE
# =========================================================

def kaprekar_step(number: int, length: int) -> int:
    digits = str(number).zfill(length)
    desc = ''.join(sorted(digits, reverse=True))
    asc = ''.join(sorted(digits))
    return int(desc) - int(asc)


def trace_full_path(seed: int, length: int, max_steps: int = 50):
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
# MODULE 2: STRUCTURAL ANALYSIS
# =========================================================

def get_symmetry_vector(number: int, length: int):
    digits = [int(d) for d in str(number).zfill(length)]
    m = length // 2
    return [digits[i] + digits[length - 1 - i] for i in range(m)]


def get_containers(number: int, length: int):
    digits = str(number).zfill(length)
    if length < 4:
        return None, None
    return digits[0:2], digits[-2:]


def compute_differences_between_iterations(path: list, length: int):
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
            'Iteration': idx,
            'From': path[idx],
            'To': path[idx + 1],
            'Digit differences': diffs,
            'Antisymmetry violation': violation,
            'Center digits equal': center_equal
        })
    return pd.DataFrame(rows)


def classify_nine_pattern(number: int, length: int) -> bool:
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
# MODULE 3: FULL RANGE SCAN (5-9 digits)
# =========================================================

def scan_full_range(length: int, sample_size: int = 20000, max_steps: int = 50,
                     progress_callback=None):
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
# MODULE 4: SEARCH BY CONTAINERS
# =========================================================

def search_by_containers(length: int, start_container: str, end_container: str,
                          sample_limit: int = 50000):
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
# MODULE 5: 3D VISUALIZATION
# =========================================================

def plot_3d_attractor_structure(df, length):
    if df.empty:
        return None

    # We use the first 3 elements of s_vector if it's long, 
    # or pad with 0 if it's short, to ensure 3D plotting works.
    def get_coords(vec):
        v = list(vec)
        while len(v) < 3:
            v.append(0)
        return v[:3]

    coords = [get_coords(row['s_vector']) for row in df['s_vector']]
    coords_array = np.array(coords)

    fig = go.Figure(data=[go.Scatter3d(
        x=coords_array[:, 0],
        y=coords_array[:, 1],
        z=coords_array[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=df['cycle_length'],
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    fig.update_layout(
        title=f"3D Structure of Attractors ({length} digits)",
        scene=dict(
            xaxis_title='Symmetry Sum 1',
            yaxis_title='Symmetry Sum 2',
            zaxis_title='Symmetry Sum 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig
  # =========================================================
# MODULE 6: STREAMLIT USER INTERFACE
# =========================================================

def main():
    st.set_page_config(page_title="Kaprekar Research Tool", layout="wide")
    st.title("🔬 Kaprekar Procedure Research Tool")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🧪 Single Number Trace", "🔍 Mass Scanning", "📦 Container Search"])

    # --- TAB 1: SINGLE NUMBER TRACE ---
    with tab1:
        st.header("Trace a single number")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            n_val = st.number_input("Enter number", min_value=0, value=495, step=1)
            l_val = st.number_input("Digit length", min_value=2, max_value=12, value=3)
            max_s = st.slider("Max steps", 10, 200, 50)
            run_trace = st.button("Run Trace")

        if run_trace:
            res = trace_full_path(n_val, l_val, max_steps=max_s)
            
            if not res['converged']:
                st.error("No convergence found within max steps.")
            else:
                st.success(f"Converged to cycle: {res['cycle']}")
                
                # Data Table
                df_trace = compute_differences_between_iterations(res['path'], l_val)
                st.subheader("Step-by-step transformation")
                st.dataframe(df_trace, use_container_width=True)

                # Symmetry Vector info
                final_val = res['cycle'][-1]
                s_vec = get_symmetry_vector(final_val, l_val)
                st.info(f"Final Attractor: {final_val} | Symmetry Vector: {s_vec}")

    # --- TAB 2: MASS SCANNING ---
    with tab2:
        st.header("Mass Scanning & 3D Analysis")
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            scan_len = st.selectbox("Digit length for scan", [3, 4, 5, 6, 7, 8, 9], index=1)
            sample_n = st.number_input("Sample size (for large n)", min_value=100, max_value=100000, value=5000)
            run_scan = st.button("Start Mass Scan")

        if run_scan:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_p(p):
                progress_bar.progress(p)
                status_text.text(f"Scanning... {int(p*100)}%")

            with st.spinner("Running scan..."):
                df_results, is_full, total = scan_full_range(scan_len, sample_size=sample_n, progress_callback=update_p)
            
            status_text.text("Scan complete!")
            
            if not df_results.empty:
                st.subheader("Scan Results Summary")
                st.write(df_results.head())

                # 3D Plot
                st.subheader("3D Attractor Structure")
                fig_3d = plot_3d_attractor_structure(df_results, scan_len)
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.warning("Not enough data for 3D plot.")
                
                # Download
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results as CSV", data=csv, file_name="kaprekar_scan.csv", mime="text/csv")
            else:
                st.warning("No attractors found in this sample.")

    # --- TAB 3: CONTAINER SEARCH ---
    with tab3:
        st.header("Search by Digit Containers")
        st.write("Find numbers that start and end with specific digit patterns.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            s_len = st.number_input("Total length", min_value=4, max_value=12, value=4)
        with c2:
            start_c = st.text_input("Start pattern (e.g. '12')", value="49")
        with c3:
            end_c = st.text_input("End pattern (e.g. '5')", value="5")
            
        run_search = st.button("Search Numbers")

        if run_search:
            if len(start_c) + len(end_c) > s_len:
                st.error("Pattern length exceeds total length!")
            else:
                found_list = search_by_containers(s_len, start_c, end_c)
                if found_list:
                    st.success(f"Found {len(found_list)} numbers.")
                    
                    # Analyze the group
                    with st.spinner("Analyzing group..."):
                        df_found = analyze_container_group(found_list, s_len)
                    
                    if not df_found.empty:
                        st.dataframe(df_found, use_container_width=True)
                        
                        # Download
                        csv_f = df_found.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Analysis", data=csv_f, file_name="container_analysis.csv", mime="text/csv")
                    else:
                        st.info("No attractors found in the searched group.")
                else:
                    st.warning("No numbers found for these constraints.")

if __name__ == "__main__":
    main()
