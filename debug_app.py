"""
Incremental test to find blocking component.
"""
import streamlit as st

st.set_page_config(page_title="Debug", layout="wide")
st.title("Debug Step 1: Imports")

# Step 1: Core imports
st.write("Testing imports...")

try:
    from smf.core.config import Config, ALL_METRICS, UI_LABELS, METRIC_DESCRIPTIONS
    st.success("1. Config imports OK")
except Exception as e:
    st.error(f"1. Config failed: {e}")

try:
    from smf.ui.execution import ExecutionManager
    st.success("2. ExecutionManager OK")
except Exception as e:
    st.error(f"2. ExecutionManager failed: {e}")

try:
    from smf.ui.plots import plot_comparison
    st.success("3. plots OK")
except Exception as e:
    st.error(f"3. plots failed: {e}")

try:
    from smf.ui.ui_generator import render_config_ui
    st.success("4. ui_generator OK")
except Exception as e:
    st.error(f"4. ui_generator failed: {e}")

st.divider()
st.write("Testing Config creation...")

try:
    cfg = Config()
    st.success("5. Config() creation OK")
except Exception as e:
    st.error(f"5. Config() failed: {e}")

st.divider()
st.write("Testing render_config_ui...")

try:
    render_config_ui(cfg, "en")
    st.success("6. render_config_ui OK")
except Exception as e:
    st.error(f"6. render_config_ui failed: {e}")
    import traceback
    st.code(traceback.format_exc())
