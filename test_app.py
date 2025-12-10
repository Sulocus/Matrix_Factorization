"""
Minimal test app to debug Streamlit rendering.
"""
import streamlit as st

st.set_page_config(page_title="Test", layout="wide")

st.title("SMF Test Page")
st.write("If you see this, basic rendering works!")

# Test 1: Basic widgets
val = st.number_input("Number", value=10)
st.write(f"Value: {val}")

# Test 2: Sidebar
with st.sidebar:
    st.write("Sidebar content")
    
# Test 3: Expander
with st.expander("Test Expander"):
    st.write("Expander content")

st.success("All tests passed!")
