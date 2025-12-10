"""
SMF Dashboard (Streamlit) - v2.

Schema-driven UI that auto-generates widgets from Config dataclass.
Supports Chinese/English language switching.

Features:
1. Tabbed configuration panels
2. Multi-metric plotting (Alpha vs Metric)
3. Historical comparison
4. Config export/import
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import time
from dataclasses import asdict, fields
from typing import get_origin, get_args, Literal

from smf.core.config import (
    Config, 
    ALL_METRICS, 
    UI_LABELS, 
    METRIC_DESCRIPTIONS,
    AlgorithmMode,
    FDistribution,
    TeacherType,
    GraphType,
)
from smf.core.runner import run_experiment

# ============================================================================
# Page Config
# ============================================================================

st.set_page_config(
    layout="wide", 
    page_title="SMF Dashboard",
    page_icon="🔬",
)

# ============================================================================
# Session State Initialization
# ============================================================================

if "config" not in st.session_state:
    st.session_state.config = Config()

if "language" not in st.session_state:
    st.session_state.language = "zh"

if "results_history" not in st.session_state:
    st.session_state.results_history = []

cfg = st.session_state.config
lang = st.session_state.language
L = UI_LABELS[lang]
M_DESC = METRIC_DESCRIPTIONS[lang]

# ============================================================================
# Helper Functions
# ============================================================================

def get_literal_options(type_hint):
    """Extract options from Literal type."""
    if get_origin(type_hint) is Literal:
        return list(get_args(type_hint))
    return []

# ============================================================================
# Sidebar: Language, Presets, History
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings" if lang == "en" else "⚙️ 设置")
    
    # Language selector
    new_lang = st.radio(
        L["language"],
        ["zh", "en"],
        index=0 if lang == "zh" else 1,
        format_func=lambda x: "中文" if x == "zh" else "English",
        horizontal=True,
    )
    if new_lang != lang:
        st.session_state.language = new_lang
        st.rerun()
    
    st.divider()
    
    # Presets
    st.subheader("📂 Presets" if lang == "en" else "📂 预设配置")
    PRESETS_DIR = "smf/ui/presets"
    os.makedirs(PRESETS_DIR, exist_ok=True)
    
    preset_files = ["Custom"] + [f for f in os.listdir(PRESETS_DIR) if f.endswith(".yaml")]
    selected_preset = st.selectbox(
        "Load Preset" if lang == "en" else "加载预设",
        preset_files,
    )
    
    if selected_preset != "Custom":
        path = os.path.join(PRESETS_DIR, selected_preset)
        st.session_state.config = Config.load(path)
        cfg = st.session_state.config
        st.success(f"Loaded {selected_preset}" if lang == "en" else f"已加载 {selected_preset}")
    
    st.divider()
    
    # History
    st.subheader("📊 History" if lang == "en" else "📊 历史记录")
    if st.session_state.results_history:
        history_options = [f"Run {i+1}" for i in range(len(st.session_state.results_history))]
        selected_history = st.multiselect(
            "Compare with" if lang == "en" else "对比历史",
            history_options,
        )
    else:
        st.caption("No history yet" if lang == "en" else "暂无历史记录")

# ============================================================================
# Main Area: Title
# ============================================================================

st.title("🔬 SMF Experiment Dashboard" if lang == "en" else "🔬 SMF 实验控制台")

# ============================================================================
# Configuration Tabs
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    L["tab_physics"],
    L["tab_algorithm"],
    L["tab_model"],
    L["tab_training"],
])

# ----------------------------------------------------------------------------
# Tab 1: Physics (Matrix & Alpha Sweep)
# ----------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Matrix" if lang == "en" else "矩阵维度")
        cfg.matrix.N1 = st.number_input(L["N1"], value=cfg.matrix.N1, min_value=10, step=10)
        cfg.matrix.N2 = st.number_input(L["N2"], value=cfg.matrix.N2, min_value=10, step=10)
        cfg.matrix.M = st.number_input(L["M"], value=cfg.matrix.M, min_value=5, step=5)
    
    with col2:
        st.subheader("Alpha Sweep" if lang == "en" else "Alpha 扫描")
        cfg.alpha.start = st.number_input(L["alpha_start"], value=cfg.alpha.start, step=0.1, format="%.1f")
        cfg.alpha.stop = st.number_input(L["alpha_stop"], value=cfg.alpha.stop, step=0.1, format="%.1f")
        cfg.alpha.step = st.number_input(L["alpha_step"], value=cfg.alpha.step, step=0.1, format="%.2f", min_value=0.01)
        
        n_alphas = len(cfg.alpha_values)
        st.caption(f"Total: {n_alphas} alpha values" if lang == "en" else f"共 {n_alphas} 个 Alpha 值")

# ----------------------------------------------------------------------------
# Tab 2: Algorithm
# ----------------------------------------------------------------------------
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mode" if lang == "en" else "模式选择")
        mode_options = ["standard", "spreading", "spreading_parallel"]
        mode_idx = mode_options.index(cfg.algorithm.mode) if cfg.algorithm.mode in mode_options else 2
        cfg.algorithm.mode = st.radio(
            L["mode"],
            mode_options,
            index=mode_idx,
            format_func=lambda x: {
                "standard": "Standard BiG-AMP",
                "spreading": "Spreading (Sequential)",
                "spreading_parallel": "Spreading (Parallel) ⚡",
            }[x],
        )
        
        # Spreading-specific options
        if cfg.algorithm.mode in ["spreading", "spreading_parallel"]:
            st.divider()
            st.subheader("Spreading Options" if lang == "en" else "Spreading 选项")
            
            f_dist_options = ["gaussian", "rademacher"]
            f_dist_idx = f_dist_options.index(cfg.spreading.f_distribution)
            cfg.spreading.f_distribution = st.radio(
                L["f_distribution"],
                f_dist_options,
                index=f_dist_idx,
                format_func=lambda x: "Gaussian N(0,1)" if x == "gaussian" else "Rademacher {-1,+1}",
                horizontal=True,
            )
            
            cfg.spreading.seed = st.number_input(L["spreading_seed"], value=cfg.spreading.seed)
    
    with col2:
        st.subheader("Hyperparameters" if lang == "en" else "超参数")
        cfg.algorithm.damping = st.slider(L["damping"], 0.0, 1.0, cfg.algorithm.damping, 0.05)
        cfg.algorithm.noise_var = st.number_input(
            L["noise_var"], 
            value=cfg.algorithm.noise_var, 
            format="%.1e",
            step=1e-11,
        )
        cfg.algorithm.onsager_enabled = st.checkbox(L["onsager_enabled"], value=cfg.algorithm.onsager_enabled)
        cfg.algorithm.use_compile = st.checkbox(L["use_compile"], value=cfg.algorithm.use_compile)

# ----------------------------------------------------------------------------
# Tab 3: Model (Teacher & Graph)
# ----------------------------------------------------------------------------
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Teacher" if lang == "en" else "教师模型")
        teacher_options = ["standard", "orthogonal", "scaled_variance"]
        teacher_idx = teacher_options.index(cfg.teacher.type) if cfg.teacher.type in teacher_options else 0
        cfg.teacher.type = st.selectbox(
            L["teacher_type"],
            teacher_options,
            index=teacher_idx,
            format_func=lambda x: {
                "standard": "Standard Gaussian N(0,1)",
                "orthogonal": "Orthogonal (QR)",
                "scaled_variance": "Scaled Variance",
            }[x],
        )
        
        # Show variance_scale only for scaled_variance
        if cfg.teacher.type == "scaled_variance":
            cfg.teacher.variance_scale = st.number_input(
                L["variance_scale"], 
                value=cfg.teacher.variance_scale,
                min_value=0.1,
                step=0.1,
            )
    
    with col2:
        st.subheader("Graph" if lang == "en" else "图拓扑")
        graph_options = ["random", "uniform", "low_loop"]
        graph_idx = graph_options.index(cfg.graph.type) if cfg.graph.type in graph_options else 0
        cfg.graph.type = st.selectbox(
            L["graph_type"],
            graph_options,
            index=graph_idx,
            format_func=lambda x: {
                "random": "Random",
                "uniform": "Uniform (Bi-regular)",
                "low_loop": "Low-Loop (MCMC)",
            }[x],
        )
        
        # Show low_loop options
        if cfg.graph.type == "low_loop":
            cfg.graph.loop_order = st.number_input(L["loop_order"], value=cfg.graph.loop_order, min_value=2, max_value=4)
            cfg.graph.n_sweeps = st.number_input(L["n_sweeps"], value=cfg.graph.n_sweeps, min_value=1, max_value=20)

# ----------------------------------------------------------------------------
# Tab 4: Training & Execution
# ----------------------------------------------------------------------------
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training" if lang == "en" else "训练参数")
        cfg.training.max_steps = st.number_input(L["max_steps"], value=cfg.training.max_steps, min_value=10, step=100)
        cfg.training.samples_per_alpha = st.number_input(L["samples_per_alpha"], value=cfg.training.samples_per_alpha, min_value=1, max_value=20)
        cfg.training.seed = st.number_input(L["seed"], value=cfg.training.seed)
        
        device_options = ["cpu", "cuda"]
        device_idx = device_options.index(cfg.training.device) if cfg.training.device in device_options else 0
        cfg.training.device = st.radio(L["device"], device_options, index=device_idx, horizontal=True)
    
    with col2:
        st.subheader("Output" if lang == "en" else "输出设置")
        cfg.execution.metrics_to_compute = st.multiselect(
            L["metrics_to_compute"],
            ALL_METRICS,
            default=cfg.execution.metrics_to_compute,
            format_func=lambda x: f"{x}: {M_DESC.get(x, x)}",
        )

# ============================================================================
# Action Buttons
# ============================================================================

st.divider()

col_run, col_export = st.columns([1, 3])

with col_run:
    run_button = st.button(L["run_experiment"], type="primary", use_container_width=True)

with col_export:
    # Export config
    yaml_str = yaml.dump(cfg.to_dict(), default_flow_style=False, allow_unicode=True)
    st.download_button(
        L["download_config"],
        yaml_str,
        "config.yaml",
        mime="text/yaml",
    )

# ============================================================================
# Run Experiment
# ============================================================================

if run_button:
    with st.spinner("Running experiment..." if lang == "en" else "正在运行实验..."):
        try:
            results = run_experiment(cfg)
            st.session_state.last_results = results
            st.session_state.results_history.append(results)
            st.success("Experiment completed!" if lang == "en" else "实验完成！")
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# ============================================================================
# Results Display
# ============================================================================

if "last_results" in st.session_state:
    st.divider()
    st.header(L["results_title"])
    
    results = st.session_state.last_results
    
    # Metrics summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if "Q_Y" in results and results["Q_Y"]:
            st.metric("Q_Y (final)", f"{results['Q_Y'][-1]:.4f}")
    with col2:
        if "Q_W" in results and results["Q_W"]:
            st.metric("Q_W (final)", f"{results['Q_W'][-1]:.4f}")
    with col3:
        if "Q_X" in results and results["Q_X"]:
            st.metric("Q_X (final)", f"{results['Q_X'][-1]:.4f}")
    with col4:
        st.metric("Time", f"{results.get('total_time', 0):.2f}s")
    
    # Plot: Alpha vs Metrics
    st.subheader(L["metrics_title"])
    
    # Prepare data for plotting
    plot_data = {"Alpha": results["alpha_values"]}
    
    for metric in cfg.execution.metrics_to_compute:
        if metric in results and results[metric]:
            plot_data[metric] = results[metric]
    
    if len(plot_data) > 1:
        df = pd.DataFrame(plot_data)
        df = df.set_index("Alpha")
        st.line_chart(df)
    else:
        st.caption("No metrics to plot" if lang == "en" else "没有可绘制的指标")
