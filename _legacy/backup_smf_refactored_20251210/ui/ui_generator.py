"""
Schema-Driven UI Generator for SMF (v2 - Fixed).

Fixes:
- Unique keys for all widgets
- Better decimal formatting
- Correct conditional rendering for Spreading
"""

import streamlit as st
import dataclasses
from typing import Any, Dict, List, Optional, get_origin, get_args, Literal
from smf.core.config import Config, UI_LABELS


def infer_widget(field_type: type, field_name: str) -> str:
    """Infer appropriate Streamlit widget from field type."""
    origin = get_origin(field_type)
    
    if origin is Literal:
        return "selectbox"
    elif field_type == bool:
        return "checkbox"
    elif field_type == int:
        return "number_input_int"
    elif field_type == float:
        if "damping" in field_name:
            return "slider"
        return "number_input_float"
    elif field_type == str:
        if field_name == "device":
            return "radio"
        return "text_input"
    elif origin is list:
        return "multiselect"
    else:
        return "text_input"


def get_literal_options(field_type: type) -> List[str]:
    """Extract options from Literal type."""
    origin = get_origin(field_type)
    if origin is Literal:
        return list(get_args(field_type))
    return []


# Conditional rendering rules
CONDITIONAL_RULES = {
    # SpreadingConfig visible when algorithm.mode contains "spreading"
    "spreading": lambda cfg: "spreading" in cfg.algorithm.mode,
    # Teacher variance_scale only when teacher.type == "scaled_variance"
    "teacher.variance_scale": lambda cfg: cfg.teacher.type == "scaled_variance",
    # Graph low_loop params
    "graph.loop_order": lambda cfg: cfg.graph.type == "low_loop",
    "graph.n_sweeps": lambda cfg: cfg.graph.type == "low_loop",
    "graph.alpha_threshold": lambda cfg: cfg.graph.type == "low_loop",
}


def render_field(
    cfg_obj: Any, 
    field: dataclasses.Field, 
    lang: str,
    key_prefix: str,
) -> Any:
    """Render a single field as a Streamlit widget with unique key."""
    field_name = field.name
    field_type = field.type
    current_value = getattr(cfg_obj, field_name)
    
    # Generate unique key
    unique_key = f"{key_prefix}_{field_name}"
    
    # Get label
    L = UI_LABELS.get(lang, UI_LABELS["zh"])
    label = L.get(field_name, field_name)
    
    widget = infer_widget(field_type, field_name)
    
    if widget == "checkbox":
        return st.checkbox(label, value=current_value, key=unique_key)
    
    elif widget == "number_input_int":
        return int(st.number_input(
            label, 
            value=int(current_value), 
            step=None,  # No +/- buttons
            format="%d",
            key=unique_key
        ))
    
    elif widget == "number_input_float":
        return float(st.number_input(
            label, 
            value=float(current_value), 
            step=None,  # No +/- buttons
            format="%.2f",  # 2 decimal places
            key=unique_key
        ))
    
    elif widget == "slider":
        return st.slider(label, 0.0, 1.0, float(current_value), 0.05, key=unique_key)
    
    elif widget == "selectbox":
        options = get_literal_options(field_type)
        idx = options.index(current_value) if current_value in options else 0
        return st.selectbox(label, options, index=idx, key=unique_key)
    
    elif widget == "radio":
        if field_name == "device":
            options = ["cpu", "cuda"]
            idx = options.index(current_value) if current_value in options else 0
            return st.radio(label, options, index=idx, horizontal=True, key=unique_key)
        return st.text_input(label, value=str(current_value), key=unique_key)
    
    elif widget == "multiselect":
        return st.multiselect(label, current_value, default=current_value, key=unique_key)
    
    else:
        return st.text_input(label, value=str(current_value), key=unique_key)


def render_sub_config(
    root_cfg: Config,
    sub_cfg_name: str,
    sub_cfg: Any,
    lang: str,
    columns: int = 2,
):
    """Render all fields of a sub-config."""
    fields = dataclasses.fields(sub_cfg)
    
    cols = st.columns(columns)
    col_idx = 0
    
    for field in fields:
        # Check conditional rule for individual field
        full_key = f"{sub_cfg_name}.{field.name}"
        if full_key in CONDITIONAL_RULES:
            if not CONDITIONAL_RULES[full_key](root_cfg):
                continue
        
        with cols[col_idx % columns]:
            key_prefix = f"cfg_{sub_cfg_name}"
            new_value = render_field(sub_cfg, field, lang, key_prefix)
            setattr(sub_cfg, field.name, new_value)
        
        col_idx += 1


def render_config_ui(cfg: Config, lang: str):
    """Main entry: Render entire Config as Streamlit UI."""
    L = UI_LABELS.get(lang, UI_LABELS["zh"])
    
    GROUPS = {
        "physics": ["matrix", "alpha"],
        "algorithm": ["algorithm", "spreading"],  # spreading right after algorithm
        "model": ["teacher", "graph"],
        "training": ["training"],
        "execution": ["execution"],
    }
    
    GROUP_TITLES = {
        "physics": L.get("tab_physics", "物理参数"),
        "algorithm": L.get("tab_algorithm", "算法配置"),
        "model": L.get("tab_model", "模型配置"),
        "training": L.get("tab_training", "训练与执行"),
        "execution": "Metrics" if lang == "en" else "指标选择",
    }
    
    for group_name, sub_cfg_names in GROUPS.items():
        with st.expander(GROUP_TITLES[group_name], expanded=(group_name in ["physics", "algorithm"])):
            for sub_cfg_name in sub_cfg_names:
                # Check whole-subconfig conditional
                if sub_cfg_name in CONDITIONAL_RULES:
                    if not CONDITIONAL_RULES[sub_cfg_name](cfg):
                        continue
                
                sub_cfg = getattr(cfg, sub_cfg_name, None)
                if sub_cfg and dataclasses.is_dataclass(sub_cfg):
                    if len(sub_cfg_names) > 1:
                        st.caption(f"**{sub_cfg_name.upper()}**")
                    render_sub_config(cfg, sub_cfg_name, sub_cfg, lang)
