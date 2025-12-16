
from streamlit.testing.v1 import AppTest
import pandas as pd
import numpy as np

# Path to the Streamlit app file
APP_FILE = "app.py"

def test_initial_state_and_defaults():
    """
    Test that the app loads with expected default values and initial components.
    Verifies default session state, widget values, metrics, and dataframe structures.
    """
    at = AppTest.from_file(APP_FILE).run()

    # --- Verify Sidebar and Title ---
    assert at.sidebar.image[0].src == "https://www.quantuniversity.com/assets/img/logo5.jpg"
    assert at.title[0].value == "QuLab: PE-AI readiness simulator"

    # --- Section 1: Welcome & Company Selection ---
    assert at.text_input[0].label == "Target Company Name"
    assert at.text_input[0].value == "InnovateCo"
    assert at.selectbox[0].label == "Company Industry Sector"
    assert at.selectbox[0].value == "Manufacturing"

    # Verify session state defaults
    assert at.session_state["company_name"] == "InnovateCo"
    assert at.session_state["company_industry"] == "Manufacturing"
    assert at.session_state["raw_rating_data_infra"] == 2
    assert at.session_state["raw_rating_ai_governance"] == 1
    assert at.session_state["alpha_param"] == 0.6
    assert at.session_state["beta_param"] == 0.15
    assert at.session_state["synergy_score"] == 50
    assert at.session_state["sensitivity_change_delta"] == 1
    assert at.session_state["exit_visible_score"] == 30
    assert at.session_state["exit_documented_score"] == 20
    assert at.session_state["exit_sustainable_score"] == 25

    # --- Section 2: Define Static Framework Parameters (Expander) ---
    at.expander[0].open().run() # Open expander to access its contents
    assert at.dataframe[0].to_dict('records') # Systematic Opportunity Scores
    assert at.dataframe[1].to_dict('records') # Dimension Weights
    assert at.dataframe[2].to_dict('records') # Industry Benchmarks
    at.expander[0].close().run() # Close it again if needed, though not strictly necessary

    # --- Section 3: Collect Raw Idiosyncratic Readiness Ratings ---
    # Check a few sliders for default values
    assert at.slider[0].label == "Data Infrastructure Rating (1-5)"
    assert at.slider[0].value == 2
    assert at.slider[1].label == "AI Governance Rating (1-5)"
    assert at.slider[1].value == 1
    assert at.slider[6].label == "Culture Rating (1-5)"
    assert at.slider[6].value == 2

    # --- Section 4: Calculate Normalized Idiosyncratic Readiness ---
    assert at.metric[0].label == "InnovateCo's Idiosyncratic Readiness Score"
    assert round(float(at.metric[0].value), 2) == 46.00 # Calculated manually for defaults
    assert len(at.dataframe) >= 3 # innovateco_dimension_scores_df
    assert len(at.pyplot) >= 1 # Idiosyncratic Readiness Dimension Scores plot

    # --- Section 5: Compute the Overall PE Org-AI-R Score ---
    assert at.slider[7].label == "Weight on Organizational Factors ($\\alpha$)"
    assert at.slider[7].value == 0.6
    assert at.slider[8].label == "Synergy Coefficient ($\\beta$)"
    assert at.slider[8].value == 0.15
    assert at.slider[9].label == "Synergy Score (0-100)"
    assert at.slider[9].value == 50
    assert at.metric[1].label == "Overall PE Org-AI-R Score for InnovateCo"
    assert round(float(at.metric[1].value), 2) == 65.10 # Calculated manually for defaults

    # --- Section 6: Perform Gap Analysis Against Industry Benchmarks ---
    assert len(at.dataframe) >= 4 # Gap analysis dataframe
    assert len(at.pyplot) >= 3 # Two more plots for gap analysis

    # --- Section 7: Conduct Scenario Analysis for Strategic Planning ---
    assert len(at.dataframe) >= 5 # Scenario results dataframe
    assert len(at.pyplot) >= 4 # One more plot for scenario analysis

    # --- Section 8: Perform Sensitivity Analysis of Key Dimensions ---
    assert at.slider[10].label == "Raw Rating Change for Sensitivity Analysis ($\\pm$ points)"
    assert at.slider[10].value == 1
    assert len(at.dataframe) >= 6 # Sensitivity dataframe
    assert len(at.pyplot) >= 5 # One more plot for sensitivity analysis

    # --- Section 9: Evaluate Exit-Readiness ---
    assert at.slider[11].label == "Visible Score (0-100)"
    assert at.slider[11].value == 30
    assert at.slider[12].label == "Documented Score (0-100)"
    assert at.slider[12].value == 20
    assert at.slider[13].label == "Sustainable Score (0-100)"
    assert at.slider[13].value == 25
    assert at.metric[2].label == "Overall Exit-AI-R Score for InnovateCo"
    assert round(float(at.metric[2].value), 2) == 25.25 # Calculated manually for defaults
    assert len(at.pyplot) >= 6 # One more plot for exit readiness


def test_company_info_changes_update_framework():
    """
    Test changing company name and industry affects relevant displays and framework parameters.
    """
    at = AppTest.from_file(APP_FILE).run()

    # Change company name
    at.text_input[0].set_value("QuantumCorp").run()
    assert at.session_state["company_name"] == "QuantumCorp"
    assert "QuantumCorp" in at.title[0].value
    assert "QuantumCorp" in at.metric[0].label # Idiosyncratic readiness label update

    # Change industry to 'Healthcare'
    at.selectbox[0].set_value("Healthcare").run()
    assert at.session_state["company_industry"] == "Healthcare"
    assert "Healthcare" in at.markdown[4].value # "AI Readiness Dimension Weights for Healthcare"

    # Verify updated weights in the dataframe (inside expander)
    at.expander[0].open().run() # Open the expander to make dataframe accessible
    weights_df = at.dataframe[1].to_dict('records')
    # For Healthcare, 'AI Governance' weight should be 0.20 (vs. 0.10 for Manufacturing)
    ai_governance_weight = next((item for item in weights_df if item["Dimension"] == "AI Governance"), {}).get("Weight")
    assert ai_governance_weight == 0.20

    # Verify Systematic Opportunity Score update (Healthcare is 80, Manufacturing was 75)
    # The PE Org-AI-R score will change due to this, check its label mentions new industry
    assert "Healthcare" in at.metric[1].label

    # Ensure the Idiosyncratic Readiness score recalculates with new weights
    # Default raw ratings: Data Infrastructure=2, AI Governance=1, Tech Stack=3, Talent=2, Leadership=3, Use Case Portfolio=1, Culture=2
    # Normalized scores: 40, 20, 60, 40, 60, 20, 40
    # Healthcare weights: DI=0.15, AG=0.20, TS=0.15, Talent=0.15, Leadership=0.10, UCP=0.15, Culture=0.10
    # Idiosyncratic Readiness (Healthcare, default raw):
    # (40*0.15) + (20*0.20) + (60*0.15) + (40*0.15) + (60*0.10) + (20*0.15) + (40*0.10)
    # = 6.0 + 4.0 + 9.0 + 6.0 + 6.0 + 3.0 + 4.0 = 38.0
    assert round(float(at.metric[0].value), 2) == 38.00


def test_idiosyncratic_readiness_calculation():
    """
    Test that changing raw ratings correctly updates the Idiosyncratic Readiness Score and its detailed dataframe.
    """
    at = AppTest.from_file(APP_FILE).run()

    # Capture initial score for 'Manufacturing' industry
    initial_idio_score = float(at.metric[0].value) # Should be 46.0

    # Change 'Data Infrastructure' raw rating from 2 to 5 (slider index 0)
    at.slider[0].set_value(5).run()
    assert at.session_state["raw_rating_data_infra"] == 5

    # Manual calculation for expected score with 'Data Infrastructure': 5
    # (5/5)*100 = 100
    # Manufacturing weights: DI=0.20, AG=0.10, TS=0.15, Talent=0.15, Leadership=0.15, UCP=0.15, Culture=0.10
    # Raw ratings (updated DI, others default): 5, 1, 3, 2, 3, 1, 2
    # Normalized scores: 100, 20, 60, 40, 60, 20, 40
    # Weighted sum: (100*0.20) + (20*0.10) + (60*0.15) + (40*0.15) + (60*0.15) + (20*0.15) + (40*0.10)
    # = 20 + 2 + 9 + 6 + 9 + 3 + 4 = 53.0
    updated_idio_score = float(at.metric[0].value)
    assert updated_idio_score == 53.0

    # Verify the detailed dimension scores dataframe also reflects this change
    dimension_scores_df = at.dataframe[3].to_dict('records') # Index 3 after expander frames
    data_infra_score = next((item for item in dimension_scores_df if item["Dimension"] == "Data Infrastructure"), {}).get("Normalized Score (0-100)")
    assert data_infra_score == 100.0

    # Change another rating: 'Use Case Portfolio' from 1 to 3 (slider index 5)
    at.slider[5].set_value(3).run()
    assert at.session_state["raw_rating_use_case_portfolio"] == 3
    # New normalized for Use Case: (3/5)*100 = 60
    # Old normalized Use Case: 20
    # Change in Idio score: (60-20) * Weight_UCP = 40 * 0.15 = 6.0
    # New Idio score: 53.0 + 6.0 = 59.0
    assert float(at.metric[0].value) == 59.0


def test_pe_org_ai_r_score_calculation():
    """
    Test that changing alpha, beta, and synergy parameters correctly updates the PE Org-AI-R Score.
    Uses default Idiosyncratic Readiness (46.0) and Systematic Opportunity (75 for Manufacturing).
    """
    at = AppTest.from_file(APP_FILE).run()

    # Initial PE Org-AI-R with defaults:
    # Idiosyncratic Readiness (Manufacturing, default raw ratings): 46.0
    # Systematic Opportunity (Manufacturing): 75
    # alpha: 0.6, beta: 0.15, Synergy Score: 50
    # PE Org-AI-R = (0.6 * 46.0) + ((1 - 0.6) * 75) + (0.15 * 50)
    #             = 27.6 + 30.0 + 7.5 = 65.1
    initial_pe_org_ai_r = float(at.metric[1].value)
    assert round(initial_pe_org_ai_r, 2) == 65.10

    # Change alpha to 0.8 (slider index 7)
    at.slider[7].set_value(0.8).run()
    assert at.session_state["alpha_param"] == 0.8
    # PE Org-AI-R = (0.8 * 46.0) + (0.2 * 75) + (0.15 * 50)
    #             = 36.8 + 15.0 + 7.5 = 59.3
    assert round(float(at.metric[1].value), 2) == 59.30

    # Change beta to 0.20 (slider index 8)
    at.slider[8].set_value(0.20).run()
    assert at.session_state["beta_param"] == 0.20
    # PE Org-AI-R = (0.8 * 46.0) + (0.2 * 75) + (0.20 * 50)
    #             = 36.8 + 15.0 + 10.0 = 61.8
    assert round(float(at.metric[1].value), 2) == 61.80

    # Change Synergy Score to 70 (slider index 9)
    at.slider[9].set_value(70).run()
    assert at.session_state["synergy_score"] == 70
    # PE Org-AI-R = (0.8 * 46.0) + (0.2 * 75) + (0.20 * 70)
    #             = 36.8 + 15.0 + 14.0 = 65.8
    assert round(float(at.metric[1].value), 2) == 65.80


def test_gap_analysis():
    """
    Test that the gap analysis dataframe is correctly populated with calculated gaps and priorities.
    Uses default settings for 'InnovateCo' in 'Manufacturing'.
    """
    at = AppTest.from_file(APP_FILE).run()

    # Default Idiosyncratic Readiness normalized scores for Manufacturing:
    # Data Infrastructure: 40, AI Governance: 20, Technology Stack: 60, Talent: 40, Leadership: 60, Use Case Portfolio: 20, Culture: 40
    # Industry Benchmarks for Manufacturing:
    # Data Infrastructure: 70, AI Governance: 55, Technology Stack: 65, Talent: 60, Leadership: 70, Use Case Portfolio: 60, Culture: 50

    # Expected Gaps (Benchmark - Company Score) and Priorities:
    # DI: 70 - 40 = 30 (High)
    # AG: 55 - 20 = 35 (High)
    # TS: 65 - 60 = 5 (Low)
    # Talent: 60 - 40 = 20 (Medium)
    # Leadership: 70 - 60 = 10 (Low)
    # UCP: 60 - 20 = 40 (High)
    # Culture: 50 - 40 = 10 (Low)

    gap_df_raw = at.dataframe[4].to_dict('records') # Index 4 after expander frames and dimension scores df
    gap_df = pd.DataFrame(gap_df_raw)

    # Verify specific gaps and priorities
    assert gap_df[gap_df['Dimension'] == 'AI Governance']['Gap (Benchmark - Company)'].iloc[0] == 35
    assert gap_df[gap_df['Dimension'] == 'AI Governance']['Priority'].iloc[0] == "High"

    assert gap_df[gap_df['Dimension'] == 'Use Case Portfolio']['Gap (Benchmark - Company)'].iloc[0] == 40
    assert gap_df[gap_df['Dimension'] == 'Use Case Portfolio']['Priority'].iloc[0] == "High"

    assert gap_df[gap_df['Dimension'] == 'Talent']['Gap (Benchmark - Company)'].iloc[0] == 20
    assert gap_df[gap_df['Dimension'] == 'Talent']['Priority'].iloc[0] == "Medium"

    assert gap_df[gap_df['Dimension'] == 'Technology Stack']['Gap (Benchmark - Company)'].iloc[0] == 5
    assert gap_df[gap_df['Dimension'] == 'Technology Stack']['Priority'].iloc[0] == "Low"


def test_scenario_analysis():
    """
    Test that scenario analysis correctly calculates PE Org-AI-R scores for predefined scenarios.
    """
    at = AppTest.from_file(APP_FILE).run()

    # Initial PE Org-AI-R (Base Case)
    base_org_ai_r = round(float(at.metric[1].value), 2) # Should be 65.10

    scenario_results_df_raw = at.dataframe[5].to_dict('records') # Index 5
    scenario_df = pd.DataFrame(scenario_results_df_raw)

    assert "Base Case" in scenario_df['Scenario'].values
    assert round(scenario_df[scenario_df['Scenario'] == 'Base Case']['PE Org-AI-R Score'].iloc[0], 2) == base_org_ai_r

    # Manual calculation for 'Optimistic Scenario' (see app.py for changes)
    # Initial: Idio=46.0, SysOpp=75, alpha=0.6, beta=0.15, Synergy=50
    # Raw ratings changes: Data Infra (+2, so 2->4), Tech Stack (+1, so 3->4), Use Case Portfolio (+2, so 1->3)
    # Synergy Score change: +20 (so 50->70)
    # New Raw Ratings: DI=4, AG=1, TS=4, Talent=2, Leadership=3, UCP=3, Culture=2
    # Normalized Scores: DI=80, AG=20, TS=80, Talent=40, Leadership=60, UCP=60, Culture=40
    # Manufacturing weights: DI=0.20, AG=0.10, TS=0.15, Talent=0.15, Leadership=0.15, UCP=0.15, Culture=0.10
    # New Idio Readiness: (80*0.20) + (20*0.10) + (80*0.15) + (40*0.15) + (60*0.15) + (60*0.15) + (40*0.10)
    #                   = 16.0 + 2.0 + 12.0 + 6.0 + 9.0 + 9.0 + 4.0 = 58.0
    # New PE Org-AI-R = (0.6 * 58.0) + (0.4 * 75) + (0.15 * 70)
    #                 = 34.8 + 30.0 + 10.5 = 75.3
    optimistic_score = scenario_df[scenario_df['Scenario'] == 'Optimistic Scenario']['PE Org-AI-R Score'].iloc[0]
    assert round(optimistic_score, 2) == 75.30

    # Manual calculation for 'Pessimistic Scenario'
    # Raw ratings changes: AI Governance (-1, so 1->1), Culture (-1, so 2->1) - Note AI Governance can't go below 1
    # Synergy Score change: -10 (so 50->40)
    # New Raw Ratings: DI=2, AG=1, TS=3, Talent=2, Leadership=3, UCP=1, Culture=1
    # Normalized Scores: DI=40, AG=20, TS=60, Talent=40, Leadership=60, UCP=20, Culture=20
    # New Idio Readiness: (40*0.20) + (20*0.10) + (60*0.15) + (40*0.15) + (60*0.15) + (20*0.15) + (20*0.10)
    #                   = 8.0 + 2.0 + 9.0 + 6.0 + 9.0 + 3.0 + 2.0 = 39.0
    # New PE Org-AI-R = (0.6 * 39.0) + (0.4 * 75) + (0.15 * 40)
    #                 = 23.4 + 30.0 + 6.0 = 59.4
    pessimistic_score = scenario_df[scenario_df['Scenario'] == 'Pessimistic Scenario']['PE Org-AI-R Score'].iloc[0]
    assert round(pessimistic_score, 2) == 59.40


def test_sensitivity_analysis():
    """
    Test that sensitivity analysis correctly calculates the impact of dimension changes on PE Org-AI-R.
    Also verifies the sorting of the results.
    """
    at = AppTest.from_file(APP_FILE).run()

    # Initial sensitivity_change_delta is 1 (slider index 10)
    assert at.slider[10].value == 1

    sensitivity_df_raw = at.dataframe[6].to_dict('records') # Index 6
    sensitivity_df = pd.DataFrame(sensitivity_df_raw)

    # Check sorting: 'Impact on PE Org-AI-R' should be monotonically increasing
    assert sensitivity_df['Impact on PE Org-AI-R'].is_monotonic_increasing

    # Base PE Org-AI-R score = 65.1
    # Test 'Data Infrastructure (+1 Raw Rating)'
    # Base raw rating for DI is 2. Change to 2+1 = 3.
    # Normalized score changes from (2/5)*100 = 40 to (3/5)*100 = 60. Change = +20.
    # Weight for DI (Manufacturing) = 0.20.
    # Impact on Idiosyncratic Readiness = +20 * 0.20 = +4.0
    # Impact on PE Org-AI-R = alpha * Impact_Idio = 0.6 * 4.0 = +2.4
    data_infra_impact = sensitivity_df[sensitivity_df['Dimension Change'] == 'Data Infrastructure (+1 Raw Rating)']['Impact on PE Org-AI-R'].iloc[0]
    assert round(data_infra_impact, 2) == 2.40

    # Test 'Synergy Score (+10)' (change_delta * 10 = 1 * 10 = 10)
    # Base Synergy Score = 50. Change to 50+10 = 60. Change = +10.
    # Impact on PE Org-AI-R = beta * Impact_Synergy = 0.15 * 10 = +1.5
    synergy_impact = sensitivity_df[sensitivity_df['Dimension Change'] == 'Synergy Score (+10)']['Impact on PE Org-AI-R'].iloc[0]
    assert round(synergy_impact, 2) == 1.50

    # Change sensitivity delta to 2 and re-run
    at.slider[10].set_value(2).run()
    assert at.session_state["sensitivity_change_delta"] == 2

    sensitivity_df_updated_raw = at.dataframe[6].to_dict('records')
    sensitivity_df_updated = pd.DataFrame(sensitivity_df_updated_raw)

    # Check 'Data Infrastructure (+2 Raw Rating)' with new delta
    # Base raw rating for DI is 2. Change to 2+2 = 4.
    # Normalized score changes from 40 to (4/5)*100 = 80. Change = +40.
    # Impact on Idiosyncratic Readiness = +40 * 0.20 = +8.0
    # Impact on PE Org-AI-R = alpha * Impact_Idio = 0.6 * 8.0 = +4.8
    data_infra_impact_delta2 = sensitivity_df_updated[sensitivity_df_updated['Dimension Change'] == 'Data Infrastructure (+2 Raw Rating)']['Impact on PE Org-AI-R'].iloc[0]
    assert round(data_infra_impact_delta2, 2) == 4.80


def test_exit_ai_r_score_calculation():
    """
    Test that changing Visible, Documented, and Sustainable scores correctly updates the Exit-AI-R Score.
    """
    at = AppTest.from_file(APP_FILE).run()

    # Initial state:
    # Visible: 30, Documented: 20, Sustainable: 25
    # Weights: Visible: 0.4, Documented: 0.35, Sustainable: 0.25
    # Expected Exit-AI-R = (30 * 0.4) + (20 * 0.35) + (25 * 0.25)
    #                   = 12.0 + 7.0 + 6.25 = 25.25
    initial_exit_ai_r = float(at.metric[2].value)
    assert round(initial_exit_ai_r, 2) == 25.25

    # Change Visible score to 60 (slider index 11)
    at.slider[11].set_value(60).run()
    assert at.session_state["exit_visible_score"] == 60
    # Expected Exit-AI-R with Visible=60:
    # (60 * 0.4) + (20 * 0.35) + (25 * 0.25)
    # = 24.0 + 7.0 + 6.25 = 37.25
    assert round(float(at.metric[2].value), 2) == 37.25

    # Change Documented score to 50 (slider index 12)
    at.slider[12].set_value(50).run()
    assert at.session_state["exit_documented_score"] == 50
    # Expected Exit-AI-R with Visible=60, Documented=50:
    # (60 * 0.4) + (50 * 0.35) + (25 * 0.25)
    # = 24.0 + 17.5 + 6.25 = 47.75
    assert round(float(at.metric[2].value), 2) == 47.75

    # Change Sustainable score to 75 (slider index 13)
    at.slider[13].set_value(75).run()
    assert at.session_state["exit_sustainable_score"] == 75
    # Expected Exit-AI-R with Visible=60, Documented=50, Sustainable=75:
    # (60 * 0.4) + (50 * 0.35) + (75 * 0.25)
    # = 24.0 + 17.5 + 18.75 = 60.25
    assert round(float(at.metric[2].value), 2) == 60.25
