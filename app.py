
import streamlit as st
import pandas as pd
import sys
import os

# Add the application_pages directory to the Python path
# This allows importing pages directly as modules within the application_pages directory
# without requiring an __init__.py if the execution context doesn't treat it as a package.
sys.path.append(os.path.join(os.path.dirname(__file__), "application_pages"))

# Ensure all pages are imported for navigation
# Changed import statements to directly import module names from the path
from welcome_company_selection import main as welcome_company_selection_main
from define_framework_parameters import main as define_framework_parameters_main
from collect_raw_readiness_ratings import main as collect_raw_readiness_ratings_main
from calculate_idiosyncratic_readiness import main as calculate_idiosyncratic_readiness_main
from compute_overall_pe_org_ai_r_score import main as compute_overall_pe_org_ai_r_score_main
from perform_gap_analysis import main as perform_gap_analysis_main
from conduct_scenario_analysis import main as conduct_scenario_analysis_main
from perform_sensitivity_analysis import main as perform_sensitivity_analysis_main
from evaluate_exit_readiness import main as evaluate_exit_readiness_main

st.set_page_config(page_title="PE-AI Readiness Simulator - QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("PE-AI Readiness Simulator")
st.divider()

st.markdown("""
In this lab, you step into the shoes of **Alex, a Quantitative Analyst at VentureBridge Capital**.
Your critical mission is to evaluate **'InnovateCo'**, a potential acquisition target in the Manufacturing sector,
through the lens of its AI readiness.

**The Challenge:** Private Equity firms need a robust, quantitative framework to assess a target company's
true AI potential – going beyond superficial claims to understand internal capabilities, external market
opportunity, and their synergy. Your task is to provide this structured assessment, quantifying AI maturity,
identifying specific investment opportunities and risks, and ultimately informing multi-million dollar investment decisions.

This application guides you through an **end-to-end narrative**:
1.  **Introduce the Target**: Identify InnovateCo and its industry.
2.  **Review Framework**: Understand the foundational data and models.
3.  **Assess Internal Capabilities**: Input raw data from due diligence.
4.  **Calculate Idiosyncratic Readiness**: Quantify InnovateCo's internal AI maturity.
5.  **Compute Overall PE Org-AI-R Score**: Combine internal and external factors for a holistic score.
6.  **Analyze Gaps**: Benchmark InnovateCo against its industry peers.
7.  **Plan Scenarios**: Model future AI readiness under various investment strategies.
8.  **Identify Priorities**: Determine which AI dimensions drive the most impact.
9.  **Evaluate Exit Strategy**: Assess InnovateCo's AI appeal to future buyers.

Each step allows you to interact with data, models, and UI components to solve realistic problems relevant to Alex's role,
culminating in actionable insights for VentureBridge Capital's Portfolio Managers.
""")

st.divider()

# Navigation
page_options = [
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

selected_page = st.sidebar.selectbox(label="Navigation", options=page_options)

# Initialize session state for all relevant variables if not already present
if "company_name" not in st.session_state:
    st.session_state["company_name"] = "InnovateCo"
if "company_industry" not in st.session_state:
    st.session_state["company_industry"] = "Manufacturing"
if "raw_dimension_ratings" not in st.session_state:
    st.session_state["raw_dimension_ratings"] = {
        'Data Infrastructure': 2, 'AI Governance': 1, 'Technology Stack': 3,
        'Talent': 2, 'Leadership': 3, 'Use Case Portfolio': 1, 'Culture': 2
    }
if "idiosyncratic_readiness" not in st.session_state:
    st.session_state["idiosyncratic_readiness"] = 0.0
if "innovateco_dimension_scores" not in st.session_state:
    st.session_state["innovateco_dimension_scores"] = pd.DataFrame()
if "dimension_weights_df" not in st.session_state:
    st.session_state["dimension_weights_df"] = pd.DataFrame()
if "systematic_opportunity" not in st.session_state:
    st.session_state["systematic_opportunity"] = 0 # Will be populated in page 2/5
if "alpha_param" not in st.session_state:
    st.session_state["alpha_param"] = 0.6
if "beta_param" not in st.session_state:
    st.session_state["beta_param"] = 0.15
if "synergy_score" not in st.session_state:
    st.session_state["synergy_score"] = 50
if "pe_org_ai_r_score" not in st.session_state:
    st.session_state["pe_org_ai_r_score"] = 0.0
if "gap_analysis_df" not in st.session_state:
    st.session_state["gap_analysis_df"] = pd.DataFrame()
if "scenario_results_df" not in st.session_state:
    st.session_state["scenario_results_df"] = pd.DataFrame()
if "sensitivity_change_delta" not in st.session_state:
    st.session_state["sensitivity_change_delta"] = 1
if "sensitivity_df" not in st.session_state:
    st.session_state["sensitivity_df"] = pd.DataFrame()
if "exit_visible_score" not in st.session_state:
    st.session_state["exit_visible_score"] = 30
if "exit_documented_score" not in st.session_state:
    st.session_state["exit_documented_score"] = 20
if "exit_sustainable_score" not in st.session_state:
    st.session_state["exit_sustainable_score"] = 25
if "exit_ai_r_score" not in st.session_state:
    st.session_state["exit_ai_r_score"] = 0.0

# Route to the selected page
if selected_page == "1. Welcome & Company Selection":
    welcome_company_selection_main()
elif selected_page == "2. Define Static Framework Parameters":
    define_framework_parameters_main()
elif selected_page == "3. Collect Raw Idiosyncratic Readiness Ratings":
    collect_raw_readiness_ratings_main()
elif selected_page == "4. Calculate Normalized Idiosyncratic Readiness":
    calculate_idiosyncratic_readiness_main()
elif selected_page == "5. Compute the Overall PE Org-AI-R Score":
    compute_overall_pe_org_ai_r_score_main()
elif selected_page == "6. Perform Gap Analysis Against Industry Benchmarks":
    perform_gap_analysis_main()
elif selected_page == "7. Conduct Scenario Analysis for Strategic Planning":
    conduct_scenario_analysis_main()
elif selected_page == "8. Perform Sensitivity Analysis of Key Dimensions":
    perform_sensitivity_analysis_main()
elif selected_page == "9. Evaluate Exit-Readiness":
    evaluate_exit_readiness_main()


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
