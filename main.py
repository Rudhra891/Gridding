import streamlit as st
from apps import line_app, grid_app

# -----------------------------
# Custom CSS for Styling
# -----------------------------
def local_css():
    st.markdown("""
        <style>
        /* Background color */
        .stApp {
            background: background: #00F260;  /* fallback for old browsers */
background: -webkit-linear-gradient(to right, #0575E6, #00F260);  /* Chrome 10-25, Safari 5.1-6 */
background: linear-gradient(to right, #0575E6, #00F260); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */
;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: background: #4DA0B0;  /* fallback for old browsers */
background: -webkit-linear-gradient(to right, #D39D38, #4DA0B0);  /* Chrome 10-25, Safari 5.1-6 */
background: linear-gradient(to right, #D39D38, #4DA0B0); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */


;
            padding: 20px;
        }

        /* Titles */
        h1, h2, h3 {
            color: #2c3e50;
            font-family: 'Helvetica Neue', sans-serif;
        }

        /* Buttons */
        button[kind="primary"] {
            background-color: #2d6cdf !important;
            color: white !important;
            border-radius: 8px;
            font-size: 16px !important;
        }

        /* Download buttons */
        .stDownloadButton button {
            background-color: #38b000 !important;
            color: white !important;
            border-radius: 8px;
            font-size: 14px !important;
        }

        /* Sidebar radio buttons */
        .stRadio label {
            font-size: 15px !important;
            color: #2c3e50;
        }
        </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Main App Entry
# -----------------------------
def main():
    local_css()  # apply custom styling

    st.sidebar.title("Navigation")
    app_choice = st.sidebar.radio("****Choose-View****", ["Line View", "Grid View"])

    if app_choice == "Line View":
        line_app.run()
    elif app_choice == "Grid View":
        grid_app.run()

if __name__ == "__main__":
    main()
