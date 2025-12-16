
import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest
from utils import (
    SYSTEMATIC_OPPORTUNITY_SCORES, DEFAULT_DIMENSION_WEIGHTS, INDUSTRY_BENCHMARKS,
    EXIT_AI_R_WEIGHTS, calculate_idiosyncratic_readiness, calculate_pe_org_ai_r,
    calculate_exit_ai_r, perform_gap_analysis, run_scenario_analysis, perform_sensitivity_analysis
)

# Helper function to navigate between pages using the sidebar selectbox
def navigate_to_page(at: AppTest, page_name: str) -> AppTest:
    """Navigates the AppTest instance to the specified page using the sidebar selectbox."""
    # The navigation selectbox is the only st.sidebar.selectbox in the main app.py file
    # Get its current options and find the index of the target page
    options = at.selectbox[0].options # Assuming the sidebar selectbox is at index 0
    page_index = options.index(page_name)
    at.selectbox[0].set_index(page_index).run()
    return at


def test_full_application_flow_and_calculations():
    """
    Tests the full end-to-end flow of the Streamlit application,
    including navigation, widget interactions, session state updates,
    and verification of calculations and displayed data.
    """
    at = AppTest.from_file("app.py").run()

    # --- Initial State & Page 1: Welcome & Company Selection ---
    # The app starts on page 1 by default, as it's the first option in the sidebar selectbox.
    assert at.title[0].value == "PE-AI Readiness Simulator"
    assert "In this lab, you step into the shoes of **Alex" in at.markdown[0].value

    # Check initial values of widgets on Page 1 (these are rendered by welcome_company_selection.py)
    # text_input for company name (first text_input on the page)
    assert at.text_input[0].value == "InnovateCo"
    # selectbox for industry (first selectbox rendered by the page content)
    assert at.selectbox[1].value == "Manufacturing" # selectbox[0] is the sidebar navigation

    # Change company name and industry
    at.text_input[0].set_value("TestCorp").run()
    at.selectbox[1].set_value("Healthcare").run()
    
    # Verify session state updates
    assert at.session_state["company_name"] == "TestCorp"
    assert at.session_state["company_industry"] == "Healthcare"
    # Verify displayed text reflects changes
    assert f"**TestCorp** in the **Healthcare** sector." in at.markdown[-1].value
    assert "Remember to adjust subsequent input parameters" in at.info[0].value

    # --- Page 2: Define Static Framework Parameters ---
    at = navigate_to_page(at, "2. Define Static Framework Parameters")
    assert at.header[0].value == "2. Define Static Framework Parameters"

    # Verify displayed dataframes (using pd.testing.assert_frame_equal for robustness)
    pd.testing.assert_frame_equal(
        at.dataframe[0].value,
        pd.DataFrame(SYSTEMATIC_OPPORTUNITY_SCORES.items(), columns=['Industry', 'Systematic Opportunity Score (0-100)'])
    )
    pd.testing.assert_frame_equal(
        at.dataframe[1].value,
        pd.DataFrame(DEFAULT_DIMENSION_WEIGHTS.items(), columns=['Dimension', 'Weight'])
    )
    pd.testing.assert_frame_equal(
        at.dataframe[2].value, # INDUSTRY_BENCHMARKS transposed
        pd.DataFrame(INDUSTRY_BENCHMARKS).T
    )
    pd.testing.assert_frame_equal(
        at.dataframe[3].value,
        pd.DataFrame(EXIT_AI_R_WEIGHTS.items(), columns=['Component', 'Weight'])
    )

    # Verify systematic_opportunity in session state (should reflect "Healthcare" chosen earlier)
    expected_systematic_opportunity = SYSTEMATIC_OPPORTUNITY_SCORES["Healthcare"]
    assert at.session_state["systematic_opportunity"] == expected_systematic_opportunity
    assert f"Systematic Opportunity Score is **{expected_systematic_opportunity}**." in at.markdown[6].value

    # --- Page 3: Collect Raw Idiosyncratic Readiness Ratings ---
    at = navigate_to_page(at, "3. Collect Raw Idiosyncratic Readiness Ratings")
    assert at.header[0].value == "3. Collect Raw Idiosyncratic Readiness Ratings"

    # Check default slider values (from app.py initialization)
    initial_raw_ratings = {
        'Data Infrastructure': 2, 'AI Governance': 1, 'Technology Stack': 3,
        'Talent': 2, 'Leadership': 3, 'Use Case Portfolio': 1, 'Culture': 2
    }
    slider_keys = list(initial_raw_ratings.keys())
    for i, dim in enumerate(slider_keys):
        expected_default_val = initial_raw_ratings[dim]
        # Sliders are ordered based on DEFAULT_DIMENSION_WEIGHTS.keys()
        assert at.slider[i].value == expected_default_val
        # Also verify session state for each raw_rating_... key
        assert at.session_state[f"raw_rating_{dim.lower().replace(' ', '_')}"] == expected_default_val
        assert f"Alex's current assessment for **{dim}**: {expected_default_val}/5" in at.info[i].value
    
    # Change some slider values and verify session state updates
    at.slider[0].set_value(4).run() # Data Infrastructure from 2 to 4
    at.slider[1].set_value(3).run() # AI Governance from 1 to 3
    
    # Update expected_raw_ratings to reflect changes
    expected_raw_ratings_for_calc = initial_raw_ratings.copy()
    expected_raw_ratings_for_calc['Data Infrastructure'] = 4
    expected_raw_ratings_for_calc['AI Governance'] = 3
    
    assert at.session_state["raw_dimension_ratings"] == expected_raw_ratings_for_calc
    assert at.slider[0].value == 4
    assert at.slider[1].value == 3
    assert "Alex's current assessment for **Data Infrastructure**: 4/5" in at.info[0].value
    assert "Alex's current assessment for **AI Governance**: 3/5" in at.info[1].value


    # --- Page 4: Calculate Normalized Idiosyncratic Readiness ---
    at = navigate_to_page(at, "4. Calculate Normalized Idiosyncratic Readiness")
    assert at.header[0].value == "4. Calculate Normalized Idiosyncratic Readiness"
    assert "idiosyncratic_readiness" in at.session_state
    
    # Check warning if raw_dimension_ratings not present (shouldn't happen in this flow)
    # assert at.warning[0].value == "Please go back to 'Collect Raw Idiosyncratic Readiness Ratings' to input the data."

    # Recalculate expected scores using utils functions for verification
    ir_score, ir_dim_df, ir_weights_df = calculate_idiosyncratic_readiness(
        at.session_state["raw_dimension_ratings"], DEFAULT_DIMENSION_WEIGHTS
    )
    
    # Verify overall score and session state
    assert abs(at.session_state["idiosyncratic_readiness"] - ir_score) < 0.01
    assert at.metric[0].value == f"{ir_score:.1f}"
    
    # Verify dimension scores and weights dataframes
    pd.testing.assert_frame_equal(at.session_state["innovateco_dimension_scores"], ir_dim_df)
    pd.testing.assert_frame_equal(at.session_state["dimension_weights_df"], ir_weights_df)
    # Check displayed dataframes (set_index because that's how they are displayed in app)
    pd.testing.assert_frame_equal(at.dataframe[0].value, ir_dim_df.set_index('Dimension'))
    pd.testing.assert_frame_equal(at.dataframe[1].value, ir_weights_df.set_index('Dimension'))
    assert at.pyplot[0].figure is not None # Check if plot is generated

    # --- Page 5: Compute the Overall PE Org-AI-R Score ---
    at = navigate_to_page(at, "5. Compute the Overall PE Org-AI-R Score")
    assert at.header[0].value == "5. Compute the Overall PE Org-AI-R Score"
    assert "pe_org_ai_r_score" in at.session_state
    
    # Check warning if idiosyncratic_readiness not present (shouldn't happen)
    # assert at.warning[0].value == "Please complete previous steps to calculate Idiosyncratic Readiness."

    # Check default slider values (from app.py initialization)
    assert at.slider[0].value == 0.6 # alpha_param
    assert at.slider[1].value == 0.15 # beta_param
    assert at.slider[2].value == 50 # synergy_score

    # Recalculate expected PE Org-AI-R score
    expected_pe_org_ai_r = calculate_pe_org_ai_r(
        at.session_state["idiosyncratic_readiness"],
        at.session_state["systematic_opportunity"],
        at.session_state["alpha_param"],
        at.session_state["beta_param"],
        at.session_state["synergy_score"]
    )
    # Verify score and metric
    assert abs(at.session_state["pe_org_ai_r_score"] - expected_pe_org_ai_r) < 0.01
    assert at.metric[0].value == f"{expected_pe_org_ai_r:.1f}"

    # Change alpha and synergy scores and re-verify calculation
    at.slider[0].set_value(0.7).run() # Change alpha from 0.6 to 0.7
    at.slider[2].set_value(60).run() # Change synergy from 50 to 60
    
    assert at.session_state["alpha_param"] == 0.7
    assert at.session_state["synergy_score"] == 60

    updated_pe_org_ai_r = calculate_pe_org_ai_r(
        at.session_state["idiosyncratic_readiness"],
        at.session_state["systematic_opportunity"],
        0.7, # Updated alpha
        at.session_state["beta_param"],
        60 # Updated synergy
    )
    assert abs(at.session_state["pe_org_ai_r_score"] - updated_pe_org_ai_r) < 0.01
    assert at.metric[0].value == f"{updated_pe_org_ai_r:.1f}"

    # --- Page 6: Perform Gap Analysis Against Industry Benchmarks ---
    at = navigate_to_page(at, "6. Perform Gap Analysis Against Industry Benchmarks")
    assert at.header[0].value == "6. Perform Gap Analysis Against Industry Benchmarks"
    assert "gap_analysis_df" in at.session_state

    # Recalculate expected gap analysis
    expected_gap_analysis_df = perform_gap_analysis(
        at.session_state["innovateco_dimension_scores"],
        at.session_state["company_industry"]
    )
    # Verify dataframe content (scores are floats, so use rtol/atol)
    pd.testing.assert_frame_equal(at.session_state["gap_analysis_df"], expected_gap_analysis_df, rtol=1e-2, atol=1e-2)
    # Check displayed dataframe (set_index because that's how it's displayed in app)
    pd.testing.assert_frame_equal(at.dataframe[0].value, expected_gap_analysis_df.set_index('Dimension'), rtol=1e-2, atol=1e-2)
    
    assert at.pyplot[0].figure is not None # Company vs Industry plot
    assert at.pyplot[1].figure is not None # Gaps plot

    # --- Page 7: Conduct Scenario Analysis for Strategic Planning ---
    at = navigate_to_page(at, "7. Conduct Scenario Analysis for Strategic Planning")
    assert at.header[0].value == "7. Conduct Scenario Analysis for Strategic Planning"
    assert "scenario_results_df" in at.session_state

    # Define scenario definitions from the app's `conduct_scenario_analysis.py`
    scenario_definitions = {
        'Optimistic': {
            'Data Infrastructure': 2, 'AI Governance': 2, 'Technology Stack': 1,
            'Talent': 2, 'Leadership': 1, 'Use Case Portfolio': 2, 'Culture': 2
        },
        'Moderate': {
            'Data Infrastructure': 1, 'AI Governance': 1, 'Technology Stack': 0,
            'Talent': 1, 'Leadership': 0, 'Use Case Portfolio': 1, 'Culture': 1
        },
        'Pessimistic': {
            'Data Infrastructure': 0, 'AI Governance': 0, 'Technology Stack': 0,
            'Talent': 0, 'Leadership': 0, 'Use Case Portfolio': 0, 'Culture': 0
        }
    }
    
    expected_scenario_results_df = run_scenario_analysis(
        at.session_state["raw_dimension_ratings"],
        at.session_state["company_industry"],
        at.session_state["alpha_param"],
        at.session_state["beta_param"],
        at.session_state["synergy_score"],
        scenario_definitions
    )
    # Verify displayed scenario definitions table
    display_scenario_df = pd.DataFrame(scenario_definitions).T.fillna(0).astype(int)
    pd.testing.assert_frame_equal(at.dataframe[0].value, display_scenario_df)

    # Verify scenario_results_df content (scores are floats)
    pd.testing.assert_frame_equal(at.session_state["scenario_results_df"], expected_scenario_results_df, rtol=1e-2, atol=1e-2)
    # Check displayed dataframe (set_index because that's how it's displayed in app)
    pd.testing.assert_frame_equal(at.dataframe[1].value, expected_scenario_results_df.set_index('Scenario'), rtol=1e-2, atol=1e-2)
    assert at.pyplot[0].figure is not None # Check if plot is generated

    # --- Page 8: Perform Sensitivity Analysis of Key Dimensions ---
    at = navigate_to_page(at, "8. Perform Sensitivity Analysis of Key Dimensions")
    assert at.header[0].value == "8. Perform Sensitivity Analysis of Key Dimensions"
    assert "sensitivity_df" in at.session_state

    # Check default slider value
    assert at.slider[0].value == 1 # sensitivity_change_delta

    expected_sensitivity_df = perform_sensitivity_analysis(
        at.session_state["raw_dimension_ratings"],
        at.session_state["company_industry"],
        at.session_state["alpha_param"],
        at.session_state["beta_param"],
        at.session_state["synergy_score"],
        at.session_state["sensitivity_change_delta"]
    )
    # Verify sensitivity_df content (scores are floats)
    pd.testing.assert_frame_equal(at.session_state["sensitivity_df"], expected_sensitivity_df, rtol=1e-2, atol=1e-2)
    pd.testing.assert_frame_equal(at.dataframe[0].value, expected_sensitivity_df, rtol=1e-2, atol=1e-2)
    assert at.pyplot[0].figure is not None # Check if plot is generated

    # Change sensitivity delta and re-verify
    at.slider[0].set_value(2).run()
    assert at.session_state["sensitivity_change_delta"] == 2
    
    updated_sensitivity_df = perform_sensitivity_analysis(
        at.session_state["raw_dimension_ratings"],
        at.session_state["company_industry"],
        at.session_state["alpha_param"],
        at.session_state["beta_param"],
        at.session_state["synergy_score"],
        2 # Updated delta
    )
    pd.testing.assert_frame_equal(at.session_state["sensitivity_df"], updated_sensitivity_df, rtol=1e-2, atol=1e-2)
    pd.testing.assert_frame_equal(at.dataframe[0].value, updated_sensitivity_df, rtol=1e-2, atol=1e-2)

    # --- Page 9: Evaluate Exit-Readiness ---
    at = navigate_to_page(at, "9. Evaluate Exit-Readiness")
    assert at.header[0].value == "9. Evaluate Exit-Readiness"
    assert "exit_ai_r_score" in at.session_state

    # Check default slider values
    assert at.slider[0].value == 30 # exit_visible_score
    assert at.slider[1].value == 20 # exit_documented_score
    assert at.slider[2].value == 25 # exit_sustainable_score

    # Recalculate expected Exit-AI-R score
    expected_exit_ai_r = calculate_exit_ai_r(
        at.session_state["exit_visible_score"],
        at.session_state["exit_documented_score"],
        at.session_state["exit_sustainable_score"],
        EXIT_AI_R_WEIGHTS
    )
    # Verify score and metric
    assert abs(at.session_state["exit_ai_r_score"] - expected_exit_ai_r) < 0.01
    assert at.metric[0].value == f"{expected_exit_ai_r:.1f}"

    # Change a slider value and re-verify
    at.slider[0].set_value(50).run() # Change visible score
    assert at.session_state["exit_visible_score"] == 50

    updated_exit_ai_r = calculate_exit_ai_r(
        50, # Updated visible score
        at.session_state["exit_documented_score"],
        at.session_state["exit_sustainable_score"],
        EXIT_AI_R_WEIGHTS
    )
    assert abs(at.session_state["exit_ai_r_score"] - updated_exit_ai_r) < 0.01
    assert at.metric[0].value == f"{updated_exit_ai_r:.1f}"
    assert at.pyplot[0].figure is not None # Check if plot is generated

