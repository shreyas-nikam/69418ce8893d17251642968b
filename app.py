
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the current directory to sys.path to allow imports from application_pages
# This addresses ModuleNotFoundError if the current working directory isn't automatically in sys.path
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.append(script_dir)

st.set_page_config(page_title="PE-AI Readiness Simulator", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("PE-AI Readiness Simulator: VentureBridge Capital")
st.divider()

st.markdown("""
In this lab, you assume the role of **Alex, a Quantitative Analyst at VentureBridge Capital**. Your mission is to rigorously evaluate **'InnovateCo'**, a potential acquisition target in the Manufacturing sector, for its **AI readiness**. This application will guide you through VentureBridge Capital's proprietary, story-driven workflow to:

1.  **Assess InnovateCo's internal AI capabilities** (Idiosyncratic Readiness) across key dimensions.
2.  **Quantify its overall AI maturity** using the `PE Org-AI-R Score`, considering both internal strengths and external market opportunities.
3.  **Identify critical AI readiness gaps** against industry benchmarks, pinpointing areas for strategic investment.
4.  **Model future AI readiness scenarios** and perform sensitivity analysis to identify high-impact improvement areas.
5.  **Evaluate InnovateCo's attractiveness to future buyers** from an AI perspective, yielding an `Exit-AI-R Score`.

Each step mirrors a real-world task in private equity due diligence, helping you provide data-driven insights to inform multi-million dollar investment decisions.
""")

# Page Navigation
page = st.sidebar.selectbox(
    label="Navigation",
    options=[
        "1. Welcome & Company Selection",
        "2. Define Static Framework Parameters",
        "3. Collect Raw Idiosyncratic Readiness Ratings",
        "4. Calculate Normalized Idiosyncratic Readiness",
        "5. Compute the Overall PE Org-AI-R Score",
        "6. Perform Gap Analysis Against Industry Benchmarks",
        "7. Conduct Scenario Analysis for Strategic Planning",
        "8. Perform Sensitivity Analysis of Key Dimensions",
        "9. Evaluate Exit-Readiness"
    ]
)

# Load and run the selected page
if page == "1. Welcome & Company Selection":
    from application_pages.page_1_welcome import main
    main()
elif page == "2. Define Static Framework Parameters":
    from application_pages.page_2_static_framework import main
    main()
elif page == "3. Collect Raw Idiosyncratic Readiness Ratings":
    from application_pages.page_3_raw_idiosyncratic_ratings import main
    main()
elif page == "4. Calculate Normalized Idiosyncratic Readiness":
    from application_pages.page_4_normalized_idiosyncratic_readiness import main
    main()
elif page == "5. Compute the Overall PE Org-AI-R Score":
    from application_pages.page_5_pe_org_ai_r_score import main
    main()
elif page == "6. Perform Gap Analysis Against Industry Benchmarks":
    from application_pages.page_6_gap_analysis import main
    main()
elif page == "7. Conduct Scenario Analysis for Strategic Planning":
    from application_pages.page_7_scenario_analysis import main
    main()
elif page == "8. Perform Sensitivity Analysis of Key Dimensions":
    from application_pages.page_8_sensitivity_analysis import main
    main()
elif page == "9. Evaluate Exit-Readiness":
    from application_pages.page_9_exit_readiness import main
    main()
