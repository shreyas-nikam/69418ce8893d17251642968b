
# Streamlit Application Specification: PE Org-AI-R Readiness Simulator

## 1. Application Overview

**Story the User Will Walk Through:**
The user, taking on the persona of Alex, a Quantitative Analyst at VentureBridge Capital, will embark on a critical mission: to evaluate 'InnovateCo', a potential acquisition target in the Manufacturing sector. The journey begins with gathering foundational industry data, moves through assessing InnovateCo's current AI capabilities across seven key dimensions, and culminates in a comprehensive financial and strategic evaluation. Alex will interactively calculate InnovateCo's proprietary `PE Org-AI-R Score`, identify its AI readiness gaps against industry benchmarks, model potential future scenarios, and understand the sensitivity of the overall score to specific improvements. Finally, Alex will assess InnovateCo's `Exit-AI-R Score`, determining its attractiveness to future buyers from an AI perspective. Each step presents a real-world analytical challenge that Alex must solve to provide actionable insights to VentureBridge Capital's Portfolio Managers.

**The Real-World Problem the Persona is Solving:**
Private Equity firms like VentureBridge Capital need a rigorous, quantitative framework to assess a target company's AI readiness. This goes beyond simply asking if a company "uses AI"; it’s about understanding its internal capabilities (Idiosyncratic Readiness), external market opportunity (Systematic Opportunity), and the synergy between them to predict value creation and optimize exit strategies. Alex's problem is to provide this structured assessment, quantifying AI maturity and identifying specific investment opportunities and risks, which directly informs multi-million dollar investment decisions.

**How the Streamlit App Helps the Persona Apply the Concepts:**
The Streamlit application acts as Alex's interactive workbench. Instead of static reports or complex spreadsheet models, the app provides a dynamic, step-by-step interface to:
1.  **Input Due Diligence Findings:** Translate qualitative interview and assessment data into quantitative scores using intuitive sliders.
2.  **Apply Proprietary Frameworks:** Automatically calculate complex scores like `PE Org-AI-R` and `Exit-AI-R` using predefined formulas and industry-specific weights.
3.  **Visualize Insights:** Generate instant charts and tables that reveal strengths, weaknesses, and potential improvement areas, making abstract data concrete.
4.  **Perform Strategic Analysis:** Conduct real-time scenario and sensitivity analyses by adjusting key parameters, allowing Alex to explore "what-if" questions and quantify potential value creation.
5.  **Develop Actionable Recommendations:** Pinpoint high-priority areas for investment and articulate a data-driven narrative for value creation and exit strategy, moving from raw data to strategic intelligence.

**Learning Goals (Applied Skills):**
By interacting with the app, Alex (and by extension, the user) will gain applied skills in:
*   **Structured AI Readiness Assessment:** Systematically evaluating a company's AI capabilities across defined dimensions.
*   **Quantitative Modeling for Investment Decisions:** Applying a parametric framework to generate key investment scores.
*   **Contextual Data Interpretation:** Understanding and communicating complex quantitative outputs (e.g., scores, gaps, sensitivities) within a private equity investment narrative.
*   **Strategic Planning through Scenario Analysis:** Simulating future states and quantifying the impact of investment decisions.
*   **Prioritization and Resource Allocation:** Identifying critical "swing factors" for AI investment based on sensitivity analysis.
*   **Exit Strategy Formulation:** Assessing and articulating a company's AI capabilities from a buyer's perspective.

## 2. User Interface Requirements

The application will follow a linear, story-driven workflow, mirroring the Jupyter Notebook's chronological steps. Each major section will represent a phase in Alex's due diligence process.

### Layout & Navigation Structure

The application will use a sequential flow, with clear headings for each step. `st.container` will be used to logically group related inputs and outputs. Progress is indicated by moving down the page, and `st.session_state` ensures continuity.

**Page Structure & Flow:**

1.  **Welcome & Company Selection**: Introduces Alex and the task.
2.  **Define Static Framework Parameters**: Loads and displays foundational data.
3.  **Collect Raw Idiosyncratic Readiness Ratings**: User inputs company-specific dimension scores.
4.  **Calculate Normalized Idiosyncratic Readiness**: Displays company's internal AI capabilities.
5.  **Compute the Overall PE Org-AI-R Score**: Calculates and displays the primary assessment score.
6.  **Perform Gap Analysis Against Industry Benchmarks**: Compares InnovateCo to industry peers.
7.  **Conduct Scenario Analysis for Strategic Planning**: Models potential future Org-AI-R scores.
8.  **Perform Sensitivity Analysis of Key Dimensions**: Identifies most impactful dimensions.
9.  **Evaluate Exit-Readiness**: Assesses the company's AI appeal to potential buyers.

### Input Widgets and Controls

All widgets must intuitively connect to Alex's actions in the scenario.

#### **Initial Setup / Welcome & Company Selection**

*   **Widget**: `st.text_input`
    *   **Purpose in Story**: To identify the target company Alex is evaluating.
    *   **Persona Action**: Alex names the company for the report.
    *   **Parameters**:
        *   `label`: "Target Company Name"
        *   `value`: "InnovateCo" (default)
        *   `key`: "company_name"
*   **Widget**: `st.selectbox`
    *   **Purpose in Story**: To select the industry of the target company, which affects `SystematicOpportunity` and dimension weights.
    *   **Persona Action**: Alex identifies InnovateCo's sector based on initial research.
    *   **Parameters**:
        *   `label`: "Company Industry Sector"
        *   `options`: `['Manufacturing', 'Healthcare', 'Retail', 'Business Services', 'Technology']` (from `SYSTEMATIC_OPPORTUNITY_SCORES` keys)
        *   `index`: 0 (default: 'Manufacturing')
        *   `key`: "company_industry"

#### **Collect Raw Idiosyncratic Readiness Ratings (7 Dimensions)**

*   **Widget**: `st.slider` (7 instances)
    *   **Purpose in Story**: To input the raw qualitative assessment scores (1-5) for each AI readiness dimension.
    *   **Persona Action**: Alex translates due diligence findings (interviews, assessments) into a quantifiable score for each dimension.
    *   **Parameters for each slider**:
        *   `label`: e.g., "Data Infrastructure Rating (1-5)"
        *   `min_value`: 1
        *   `max_value`: 5
        *   `step`: 1
        *   `value`: (default values from notebook: `Data Infrastructure`: 2, `AI Governance`: 1, `Technology Stack`: 3, `Talent`: 2, `Leadership`: 3, `Use Case Portfolio`: 1, `Culture`: 2)
        *   `key`: e.g., "raw_rating_data_infra"

#### **Compute the Overall PE Org-AI-R Score Parameters**

*   **Widget**: `st.slider`
    *   **Purpose in Story**: To adjust the weighting given to a company's internal capabilities versus industry opportunity.
    *   **Persona Action**: Alex (or the Portfolio Manager) fine-tunes the strategic focus of the overall score.
    *   **Parameters**:
        *   `label`: "Weight on Organizational Factors ($\alpha$)"
        *   `min_value`: 0.0
        *   `max_value`: 1.0
        *   `step`: 0.05
        *   `value`: 0.6 (default)
        *   `key`: "alpha_param"
*   **Widget**: `st.slider`
    *   **Purpose in Story**: To adjust the impact of synergy between internal readiness and market opportunity.
    *   **Persona Action**: Alex models how well internal capabilities align with and amplify market potential.
    *   **Parameters**:
        *   `label`: "Synergy Coefficient ($\beta$)"
        *   `min_value`: 0.0
        *   `max_value`: 1.0
        *   `step`: 0.01
        *   `value`: 0.15 (default)
        *   `key`: "beta_param"
*   **Widget**: `st.slider`
    *   **Purpose in Story**: To input the conceptual synergy score derived from due diligence.
    *   **Persona Action**: Alex estimates the perceived alignment and amplification potential.
    *   **Parameters**:
        *   `label`: "Synergy Score (0-100)"
        *   `min_value`: 0
        *   `max_value`: 100
        *   `step`: 1
        *   `value`: 50 (default)
        *   `key`: "synergy_score"

#### **Perform Sensitivity Analysis Parameters**

*   **Widget**: `st.slider`
    *   **Purpose in Story**: To define the magnitude of hypothetical change to raw ratings for sensitivity testing.
    *   **Persona Action**: Alex determines how much to "stress-test" each dimension's impact.
    *   **Parameters**:
        *   `label`: "Raw Rating Change for Sensitivity Analysis ($\pm$ points)"
        *   `min_value`: 1
        *   `max_value`: 2
        *   `step`: 1
        *   `value`: 1 (default)
        *   `key`: "sensitivity_change_delta"

#### **Evaluate Exit-Readiness (3 Components)**

*   **Widget**: `st.slider` (3 instances)
    *   **Purpose in Story**: To input conceptual scores (0-100) for how apparent, documented, and sustainable AI capabilities are to potential buyers.
    *   **Persona Action**: Alex translates analyst insights into a score reflecting how buyers would perceive InnovateCo's AI story.
    *   **Parameters for each slider**:
        *   `label`: "Visible Score (0-100)"
        *   `min_value`: 0
        *   `max_value`: 100
        *   `step`: 1
        *   `value`: (default values from notebook: `Visible`: 30, `Documented`: 20, `Sustainable`: 25)
        *   `key`: e.g., "exit_visible_score"

### Visualization Components

All visualizations will be rendered using `st.pyplot` from `matplotlib`/`seaborn`. Each chart will have a clear title and axis labels, reinforcing its purpose within the narrative.

1.  **Idiosyncratic Readiness Dimension Scores**:
    *   **Purpose in Story**: Shows InnovateCo's internal capabilities, highlighting strengths and weaknesses. Alex uses this to quickly grasp the company's current state.
    *   **Format**: Bar chart.
    *   **Outputs**: Y-axis: Score (0-100), X-axis: AI Readiness Dimension.
    *   **Library**: `seaborn.barplot`

2.  **Comparative Bar Chart: Company vs. Industry Benchmarks**:
    *   **Purpose in Story**: Allows Alex to directly compare InnovateCo's performance against industry peers, indicating where it lags or leads.
    *   **Format**: Grouped bar chart.
    *   **Outputs**: Y-axis: Score (0-100), X-axis: AI Readiness Dimension, with two bars per dimension (Company Score, Benchmark).
    *   **Library**: `matplotlib.pyplot.plot(kind='bar')` or `seaborn` equivalent.

3.  **AI Readiness Gaps**:
    *   **Purpose in Story**: Quantifies the specific deficits InnovateCo has, directly informing prioritization for value creation. Alex relies on this to identify actionable investment areas.
    *   **Format**: Bar chart, color-coded by priority (High, Medium, Low).
    *   **Outputs**: Y-axis: Gap (Benchmark - Company Score), X-axis: AI Readiness Dimension. Positive gaps indicate areas needing improvement.
    *   **Library**: `seaborn.barplot`

4.  **PE Org-AI-R Score Under Different Scenarios**:
    *   **Purpose in Story**: Visualizes the potential trajectory of InnovateCo's overall AI readiness score under various investment assumptions, supporting strategic planning. Alex uses this to show the Portfolio Manager the ROI of AI initiatives.
    *   **Format**: Bar chart.
    *   **Outputs**: Y-axis: PE Org-AI-R Score (0-100), X-axis: Scenario (Base Case, Optimistic, Moderate, Pessimistic).
    *   **Library**: `matplotlib.pyplot.plot(kind='bar')`

5.  **Sensitivity of PE Org-AI-R to Dimension Changes**:
    *   **Purpose in Story**: Identifies "swing factors" – which dimensions, when improved, have the largest impact on the overall `PE Org-AI-R Score`. Alex uses this for prioritizing resource allocation.
    *   **Format**: Diverging bar chart (simplified tornado plot).
    *   **Outputs**: X-axis: Impact on PE Org-AI-R Score, Y-axis: Dimension and Change (e.g., "Data Infrastructure (+1 Raw Rating)").
    *   **Library**: `seaborn.barplot`

6.  **Exit-AI-R Component Scores**:
    *   **Purpose in Story**: Illustrates the breakdown of InnovateCo's AI "sellability," highlighting which aspects are strong or weak for future buyers. Alex uses this to guide the development of an "exit narrative."
    *   **Format**: Bar chart with weight annotations.
    *   **Outputs**: Y-axis: Score (0-100), X-axis: Exit-Readiness Component (Visible, Documented, Sustainable). Annotations show their respective weights.
    *   **Library**: `seaborn.barplot`

### Interactive Elements & Feedback Mechanisms

All calculations will be reactive to user inputs. Changing a slider will instantly update dependent scores, tables, and charts.

*   **Dynamic Score Displays**: `st.metric` or `st.write` will be used to display calculated `PE Org-AI-R Score`, `IdiosyncraticReadiness`, `Exit-AI-R Score`, and other key metrics, updating immediately upon input changes. This provides instant feedback on the impact of Alex's adjustments.
*   **Dynamic DataFrames**: `st.dataframe` will display the `gap_analysis_df`, `scenario_results_df`, and `sensitivity_df`, reflecting changes as inputs are adjusted. Alex can observe the immediate analytical output.
*   **Dynamic Plots**: All `st.pyplot` visualizations will regenerate and update in real-time as input sliders or dropdowns are changed. This allows Alex to visually explore scenarios and sensitivities.
*   **Contextual Text Updates**: Narrative elements and summary statements will dynamically update to incorporate calculated scores and findings, ensuring the story progresses coherently. For example, after calculating the `PE Org-AI-R Score`, the subsequent text will reference that specific score in its interpretation.

## 3. Additional Requirements

### Annotations & Tooltips

Contextual explanations will be integrated directly into the UI to help Alex understand outputs *in the context of the scenario*.
*   `st.markdown` will introduce each section, explaining its purpose for Alex's task.
*   `st.info` or `st.expander` elements will provide "Analyst Notes" or "Key Insights" below charts and tables, interpreting the results for the current scenario. For instance, after the `PE Org-AI-R` score, an `st.info` might state: "A score of X confirms concerns, indicating 'transformation opportunity' rather than 'strong AI candidate,' guiding preliminary screening."
*   Descriptions near sliders will explain the real-world implication of adjusting that parameter (e.g., for $\alpha$: "Adjusting $\alpha$ shifts focus between internal capabilities and market opportunity, reflecting VentureBridge Capital's investment philosophy.").
*   Formulas will be presented in `st.markdown` as LaTeX to reinforce the quantitative rigor.

### State Management Requirements

*   All user inputs from `st.slider`, `st.selectbox`, `st.text_input` will be stored in `st.session_state`. This ensures that if the app reruns (e.g., due to interaction), all previous selections and inputs are preserved.
*   Intermediate and final calculated scores (e.g., `idiosyncratic_readiness`, `pe_org_ai_r_score`, `exit_ai_r_score`, and their component parts) and dataframes (`gap_analysis_df`, `scenario_results_df`, `sensitivity_df`) will also be stored in `st.session_state` to maintain continuity and allow subsequent steps to access previous results.
*   The application's logic will retrieve necessary values from `st.session_state` at the beginning of each rerun, ensuring the user's progress is never lost.

## 4. Notebook Content and Code Requirements

This section maps each relevant piece of the Jupyter Notebook to its Streamlit counterpart. All Python functions will be defined in a `utils.py` file or directly within the `streamlit_app.py` file if simple enough.

### Static Data Loading (Corresponds to Notebook Section 1)

*   **Notebook Markdown**: "## 1. Step 1: Define Static Framework Parameters and Sector Data" and subsequent narrative.
    *   **Streamlit**: Displayed using `st.subheader` and `st.markdown` to provide context. The actual data structures can be placed in an `st.expander` for transparency: "View Framework Parameters".
*   **Notebook Code**: `SYSTEMATIC_OPPORTUNITY_SCORES`, `DEFAULT_DIMENSION_WEIGHTS`, `SECTOR_DIMENSION_WEIGHTS`, `INDUSTRY_BENCHMARKS` dictionaries.
    *   **Streamlit**: These dictionaries will be defined as global constants or loaded at the start of the `streamlit_app.py` file. They will be accessed directly by calculation functions.

### Raw Idiosyncratic Readiness Inputs (Corresponds to Notebook Section 2)

*   **Notebook Markdown**: "## 2. Step 2: Collect Raw Idiosyncratic Readiness Ratings" and subsequent narrative.
    *   **Streamlit**: Displayed using `st.subheader` and `st.markdown`.
*   **Notebook Code**: `company_name`, `company_industry`, `raw_dimension_ratings` dictionary.
    *   **Streamlit**: `company_name` and `company_industry` will be `st.text_input` and `st.selectbox` respectively. `raw_dimension_ratings` will be populated by 7 individual `st.slider` widgets, with their values stored in `st.session_state`.

### Idiosyncratic Readiness Calculation (Corresponds to Notebook Section 3)

*   **Notebook Markdown**: "## 3. Step 3: Calculate Normalized Idiosyncratic Readiness", narrative, and formula:
    `$$ IdiosyncraticReadiness = \frac{\sum_{i=1}^{7} w_i \cdot \text{Rating}_i}{5} \times 100 $$`
    *   **Streamlit**: `st.subheader` and `st.markdown` for narrative and formula. An `st.metric` for the overall score, and `st.dataframe` for individual dimension scores.
*   **Notebook Code**: `calculate_idiosyncratic_readiness` function.
    *   **Streamlit**: This function will be defined and called reactively. It takes `raw_ratings` (from `st.session_state` sliders) and `industry` (from `st.session_state.company_industry`) as inputs. The results (`overall_score`, `dimension_scores`, `weights`) will be stored in `st.session_state`.
*   **Visualization**: Bar chart of Idiosyncratic Readiness dimensions.
    *   **Streamlit**: Rendered using `st.pyplot(fig)` based on `innovateco_dimension_scores` from `st.session_state`.

### PE Org-AI-R Score Calculation (Corresponds to Notebook Section 4)

*   **Notebook Markdown**: "## 4. Step 4: Compute the Overall PE Org-AI-R Score", narrative, and formula:
    `$$PE \text{ Org-AI-R} = \alpha \cdot IdiosyncraticReadiness + (1 - \alpha) \cdot SystematicOpportunity + \beta \cdot Synergy$$`
    *   **Streamlit**: `st.subheader` and `st.markdown` for narrative and formula. `st.metric` will display the final score.
*   **Notebook Code**: `alpha_param`, `beta_param`, `synergy_score` variables and `calculate_pe_org_ai_r` function.
    *   **Streamlit**: `alpha_param`, `beta_param`, and `synergy_score` will be controlled by `st.slider` widgets, with values stored in `st.session_state`. The `calculate_pe_org_ai_r` function will be called reactively, using inputs from `st.session_state.idiosyncratic_readiness`, `SYSTEMATIC_OPPORTUNITY_SCORES`, and the sliders. The `innovateco_org_ai_r_score` will be stored in `st.session_state`.

### Gap Analysis (Corresponds to Notebook Section 5)

*   **Notebook Markdown**: "## 5. Step 5: Perform Gap Analysis Against Industry Benchmarks", narrative, and formula:
    `$$Gap_k = D_k^{benchmark} - D_k^{current}$$`
    *   **Streamlit**: `st.subheader` and `st.markdown` for narrative and formula. `st.dataframe` will display the results.
*   **Notebook Code**: `perform_gap_analysis` function.
    *   **Streamlit**: The `perform_gap_analysis` function will be called reactively using `st.session_state.innovateco_dimension_scores` and `INDUSTRY_BENCHMARKS`. The resulting `gap_analysis_df` will be stored in `st.session_state`.
*   **Visualizations**: Comparative Bar Chart and Gap Analysis Bar Chart.
    *   **Streamlit**: Both charts will be rendered using `st.pyplot(fig)` based on `st.session_state.gap_analysis_df`.

### Scenario Analysis (Corresponds to Notebook Section 6)

*   **Notebook Markdown**: "## 6. Step 6: Conduct Scenario Analysis for Strategic Planning" and narrative.
    *   **Streamlit**: `st.subheader` and `st.markdown` for narrative. `st.dataframe` will display the results.
*   **Notebook Code**: `scenario_definitions` dictionary and `run_scenario_analysis` function.
    *   **Streamlit**: The `scenario_definitions` will be hardcoded or loaded from static data. An `st.expander` could allow viewing or modifying the raw rating deltas for each scenario. The `run_scenario_analysis` function will be called reactively using base ratings from `st.session_state.raw_dimension_ratings`, `st.session_state.company_industry`, `st.session_state.alpha_param`, `st.session_state.beta_param`, `st.session_state.synergy_score`, and the scenario definitions. The `scenario_results_df` will be stored in `st.session_state`.
*   **Visualization**: Bar chart for scenario analysis.
    *   **Streamlit**: Rendered using `st.pyplot(fig)` based on `st.session_state.scenario_results_df`.

### Sensitivity Analysis (Corresponds to Notebook Section 7)

*   **Notebook Markdown**: "## 7. Step 7: Perform Sensitivity Analysis of Key Dimensions" and narrative.
    *   **Streamlit**: `st.subheader` and `st.markdown` for narrative. `st.dataframe` will display the results.
*   **Notebook Code**: `change_delta` variable and `perform_sensitivity_analysis` function.
    *   **Streamlit**: `change_delta` will be controlled by an `st.slider` and stored in `st.session_state`. The `perform_sensitivity_analysis` function will be called reactively using `st.session_state.raw_dimension_ratings`, `st.session_state.company_industry`, `st.session_state.alpha_param`, `st.session_state.beta_param`, `st.session_state.synergy_score`, and `st.session_state.sensitivity_change_delta`. The `sensitivity_df` will be stored in `st.session_state`.
*   **Visualization**: Sensitivity Analysis (Tornado-style chart).
    *   **Streamlit**: Rendered using `st.pyplot(fig)` based on `st.session_state.sensitivity_df`.

### Exit-Readiness Evaluation (Corresponds to Notebook Section 8)

*   **Notebook Markdown**: "## 8. Step 8: Evaluate Exit-Readiness", narrative, and formula:
    `$$Exit\text{-AI-R} = w_1 \cdot Visible + w_2 \cdot Documented + w_3 \cdot Sustainable$$`
    *   **Streamlit**: `st.subheader` and `st.markdown` for narrative and formula. `st.metric` will display the final score.
*   **Notebook Code**: `innovateco_visible`, `innovateco_documented`, `innovateco_sustainable` variables, `w1_exit`, `w2_exit`, `w3_exit` constants, and `calculate_exit_ai_r` function.
    *   **Streamlit**: `innovateco_visible`, `innovateco_documented`, `innovateco_sustainable` will be controlled by `st.slider` widgets and stored in `st.session_state`. The weights `w1_exit`, `w2_exit`, `w3_exit` will be hardcoded. The `calculate_exit_ai_r` function will be called reactively. The `innovateco_exit_ai_r_score` will be stored in `st.session_state`.
*   **Visualization**: Bar chart for Exit-AI-R components.
    *   **Streamlit**: Rendered using `st.pyplot(fig)` based on `st.session_state`'s exit component scores and weights.

**General Code Integration Principles:**
*   All calculations should be encapsulated in functions, making them easily callable and testable.
*   Dependencies (`pandas`, `numpy`, `matplotlib`, `seaborn`) should be imported at the top of `streamlit_app.py`.
*   Error handling (e.g., for `ValueError` in `calculate_pe_org_ai_r`) should be implemented with `st.error` if applicable, although the use of sliders for ranges naturally mitigates many such issues.
*   All plotting code from the notebook will be adapted directly within the Streamlit app using `st.pyplot()`.
*   All Markdown content from the notebook will be converted to `st.markdown()` calls, preserving the narrative and LaTeX formatting.

