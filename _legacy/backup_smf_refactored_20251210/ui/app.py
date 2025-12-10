"""
SMF Dashboard (Streamlit) - v4.0 (Schema-Driven Edition)

Auto-generates UI from Config dataclass.
Sub-parameters appear dynamically based on parent selections.
Terminal output at the bottom.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import time

from smf.core.config import (
    Config, 
    ALL_METRICS, 
    UI_LABELS, 
    METRIC_DESCRIPTIONS,
)
from smf.ui.execution import ExecutionManager
from smf.ui.plots import plot_comparison
from smf.ui.ui_generator import render_config_ui

# ============================================================================
# Page Config & Styles
# ============================================================================

st.set_page_config(
    layout="wide", 
    page_title="SMF Researchstation",
    page_icon="🔬",
    initial_sidebar_state="expanded",
)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = os.path.join(os.path.dirname(__file__), 'style.css')
if os.path.exists(css_path):
    load_css(css_path)

# ============================================================================
# Session State Initialization
# ============================================================================

if "config" not in st.session_state:
    st.session_state.config = Config()

if "language" not in st.session_state:
    st.session_state.language = "en"  # Default to English

if "experiment_results" not in st.session_state:
    st.session_state.experiment_results = {}

if "exec_manager" not in st.session_state:
    st.session_state.exec_manager = ExecutionManager()

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False

cfg = st.session_state.config
lang = st.session_state.language
manager = st.session_state.exec_manager
L = UI_LABELS[lang]
M_DESC = METRIC_DESCRIPTIONS[lang]

# ============================================================================
# Sidebar: Language, Presets, Monitor
# ============================================================================

with st.sidebar:
    st.title("🔬 SMF Lab")
    
    # Language
    new_lang = st.radio(
        "Language",
        ["en", "zh"],  # English first
        index=0 if lang == "en" else 1,  # Fixed: match "en" to index 0
        format_func=lambda x: "中文" if x == "zh" else "English",
        horizontal=True,
    )
    if new_lang != lang:
        st.session_state.language = new_lang
        st.rerun()

    st.divider()

    # Presets
    st.caption("📂 " + ("Presets" if lang == "en" else "预设配置"))
    PRESETS_DIR = os.path.join(os.path.dirname(__file__), "presets")
    os.makedirs(PRESETS_DIR, exist_ok=True)
    
    preset_files = ["Custom"] + [f for f in os.listdir(PRESETS_DIR) if f.endswith(".yaml")]
    selected_preset = st.selectbox(
        "Load Preset",
        preset_files,
        label_visibility="collapsed"
    )
    
    if selected_preset != "Custom":
        path = os.path.join(PRESETS_DIR, selected_preset)
        if st.button("Load" if lang == "en" else "加载预设"):
            st.session_state.config = Config.load(path)
            st.rerun()

    st.divider()
    
    # Monitor Control
    st.subheader("📡 " + ("Monitor" if lang == "en" else "实时监控"))
    
    auto_refresh = st.checkbox(
        "Auto-Scroll Log" if lang == "en" else "自动刷新日志",
        value=st.session_state.auto_refresh,
    )
    st.session_state.auto_refresh = auto_refresh
    
    if manager.is_running():
        st.info("Running..." if lang == "en" else "运行中...")
    else:
        st.caption("Idle" if lang == "en" else "空闲")

# ============================================================================
# Main Interface: Tabs
# ============================================================================

tab_config, tab_results, tab_compare = st.tabs([
    "🛠️ " + ("Configuration" if lang == "en" else "实验配置"),
    "📊 " + ("Results" if lang == "en" else "本次结果"),
    "⚖️ " + ("Comparison" if lang == "en" else "对比实验室"),
])

# ----------------------------------------------------------------------------
# Tab 1: Configuration (Schema-Driven)
# ----------------------------------------------------------------------------
with tab_config:
    st.markdown("### " + ("Experiment Setup" if lang == "en" else "实验参数设置"))
    
    # Experiment Name & Run Button
    col_name, col_action = st.columns([3, 1])
    with col_name:
        default_name = f"Run_{len(st.session_state.experiment_results) + 1}_{cfg.algorithm.mode}"
        run_name = st.text_input("Experiment Name" if lang == "en" else "实验名称", value=default_name)
    
    with col_action:
        st.write("")
        st.write("")
        disable_run = manager.is_running()
        if st.button("🚀 " + ("RUN" if lang == "en" else "运行"), 
                     type="primary", 
                     disabled=disable_run, 
                     use_container_width=True):
            st.session_state.auto_refresh = True
            manager.start_run(cfg, run_name)
            st.rerun()

    st.divider()
    
    # =========== SCHEMA-DRIVEN UI ===========
    render_config_ui(cfg, lang)
    # ========================================

# ----------------------------------------------------------------------------
# Logic: Check for completed results
# ----------------------------------------------------------------------------
latest_result = manager.get_latest_result()
if latest_result:
    r_name = latest_result['name']
    st.session_state.experiment_results[r_name] = latest_result
    manager.current_thread = None 
    st.session_state.auto_refresh = False
    st.success(f"Run '{r_name}' Completed!")

# ----------------------------------------------------------------------------
# Tab 2: Results
# ----------------------------------------------------------------------------
with tab_results:
    if not st.session_state.experiment_results:
        st.info("No results yet." if lang == "en" else "暂无结果，请先运行实验。")
    else:
        run_names = list(st.session_state.experiment_results.keys())
        selected_run = st.selectbox("Select Run", run_names, index=len(run_names)-1)
        
        res = st.session_state.experiment_results[selected_run]
        
        # Metrics Cards
        cols = st.columns(4)
        def safe_get_last(k):
            v = res.get(k, [])
            return f"{v[-1]:.4f}" if v else "N/A"
            
        cols[0].metric("Q_Y", safe_get_last("Q_Y"))
        cols[1].metric("Q_W", safe_get_last("Q_W"))
        cols[2].metric("MSE", safe_get_last("MSE"))
        cols[3].metric("Time", f"{res.get('total_time', 0):.2f}s")
        
        # Plot Selection
        st.markdown("#### " + ("Plot" if lang == "en" else "可视化"))
        available_metrics = [m for m in ALL_METRICS if m in res and res[m]]
        metrics_to_plot = st.multiselect(
            "Select Metrics" if lang == "en" else "选择要绘制的指标",
            available_metrics,
            default=[m for m in ["Q_Y", "MSE"] if m in available_metrics]
        )
        
        if metrics_to_plot:
            fig = plot_comparison([res], metrics_to_plot, title=selected_run)
            st.pyplot(fig)

# ----------------------------------------------------------------------------
# Tab 3: Comparison
# ----------------------------------------------------------------------------
with tab_compare:
    st.markdown("### " + ("Multi-Run Comparison" if lang == "en" else "多模型对比"))
    
    if len(st.session_state.experiment_results) < 1:
        st.warning("Need at least 1 run." if lang == "en" else "至少需要1次实验。")
    else:
        all_runs = list(st.session_state.experiment_results.keys())
        compare_selection = st.multiselect(
            "Select Runs" if lang == "en" else "选择运行记录",
            all_runs,
            default=all_runs[-2:] if len(all_runs) >= 2 else all_runs
        )
        
        compare_metrics = st.multiselect(
            "Metrics" if lang == "en" else "对比指标",
            ALL_METRICS,
            default=["Q_Y", "MSE"]
        )
        
        if compare_selection and compare_metrics:
            results_to_plot = [st.session_state.experiment_results[r] for r in compare_selection]
            fig_compare = plot_comparison(results_to_plot, compare_metrics, title="Comparison")
            st.pyplot(fig_compare, use_container_width=True)
            
            with st.expander("Data Table" if lang == "en" else "数据表"):
                comp_data = []
                for r in results_to_plot:
                    row = {"Run": r['name']}
                    for m in compare_metrics:
                        vals = r.get(m, [])
                        row[m] = vals[-1] if vals else None
                    comp_data.append(row)
                st.dataframe(pd.DataFrame(comp_data))

# ============================================================================
# Progress Display
# ============================================================================
st.divider()
st.subheader("📊 " + ("Progress" if lang == "en" else "进度"))

progress_state = manager.get_progress()

# Time display  
elapsed = progress_state.elapsed
elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"

col1, col2 = st.columns([3, 1])
with col1:
    st.progress(progress_state.progress, text=progress_state.message)
with col2:
    st.metric("Time", elapsed_str)

# ============================================================================
# Terminal Output
# ============================================================================
with st.expander("🖥️ " + ("Terminal" if lang == "en" else "终端输出"), expanded=True):
    logs = manager.get_logs()
    st.code(logs if logs else "Ready...", language=None)



# ============================================================================
# Auto Refresh
# ============================================================================
if st.session_state.auto_refresh and manager.is_running():
    time.sleep(0.5)
    st.rerun()

