
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="QuLab: PE-AI readiness simulator", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
# FIX: Made the main title dynamic to include company name, as expected by tests
st.title(f"QuLab: PE-AI readiness simulator - {st.session_state.get('company_name', 'InnovateCo')}")
st.divider()

st.markdown("""
### The VentureBridge Capital AI Readiness Journey: InnovateCo Evaluation

Welcome to the PE Org-AI-R Readiness Simulator, your interactive workbench for assessing target companies from an AI perspective.
You are Alex, a Quantitative Analyst at VentureBridge Capital. Your mission is to rigorously evaluate 'InnovateCo', a promising
acquisition target in the Manufacturing sector.

This simulator will guide you through a structured due diligence process, allowing you to:
1.  **Gather Foundational Data:** Define industry context and critical AI readiness parameters.
2.  **Assess Internal Capabilities:** Quantify InnovateCo's current AI maturity across seven key dimensions.
3.  **Calculate Proprietary Scores:** Compute the `PE Org-AI-R Score` to understand its overall AI readiness and potential for value creation.
4.  **Identify Gaps & Opportunities:** Compare InnovateCo's standing against industry benchmarks to pinpoint areas for strategic investment.
5.  **Model Future Scenarios:** Simulate how investments in AI could transform InnovateCo's readiness and financial attractiveness.
6.  **Prioritize Initiatives:** Determine which AI dimensions offer the greatest leverage for improvement using sensitivity analysis.
7.  **Evaluate Exit Potential:** Assess InnovateCo's `Exit-AI-R Score`, a crucial metric for future buyers.

Each step provides real-time insights, empowering VentureBridge Capital to make data-driven investment decisions, optimize portfolio
companies for AI-driven growth, and craft compelling exit narratives. Let's begin InnovateCo's AI readiness assessment!
""")

# --- Constants & Static Data ---
SYSTEMATIC_OPPORTUNITY_SCORES = {
    'Manufacturing': 75,
    'Healthcare': 80,
    'Retail': 65,
    'Business Services': 70,
    'Technology': 90
}

DEFAULT_DIMENSION_WEIGHTS = {
    'Data Infrastructure': 0.15,
    'AI Governance': 0.10,
    'Technology Stack': 0.15,
    'Talent': 0.20,
    'Leadership': 0.15,
    'Use Case Portfolio': 0.15,
    'Culture': 0.10
}

SECTOR_DIMENSION_WEIGHTS = {
    'Manufacturing': {
        'Data Infrastructure': 0.20,
        'AI Governance': 0.10,
        'Technology Stack': 0.15,
        'Talent': 0.15,
        'Leadership': 0.15,
        'Use Case Portfolio': 0.15,
        'Culture': 0.10
    },
    'Healthcare': {
        'Data Infrastructure': 0.15,
        'AI Governance': 0.20,
        'Technology Stack': 0.15,
        'Talent': 0.15,
        'Leadership': 0.10,
        'Use Case Portfolio': 0.15,
        'Culture': 0.10
    },
    'Retail': {
        'Data Infrastructure': 0.15,
        'AI Governance': 0.10,
        'Technology Stack': 0.15,
        'Talent': 0.15,
        'Leadership': 0.15,
        'Use Case Portfolio': 0.20,
        'Culture': 0.10
    },
    'Business Services': {
        'Data Infrastructure': 0.15,
        'AI Governance': 0.10,
        'Technology Stack': 0.15,
        'Talent': 0.20,
        'Leadership': 0.15,
        'Use Case Portfolio': 0.15,
        'Culture': 0.10
    },
    'Technology': {
        'Data Infrastructure': 0.15,
        'AI Governance': 0.10,
        'Technology Stack': 0.20,
        'Talent': 0.20,
        'Leadership': 0.15,
        'Use Case Portfolio': 0.10,
        'Culture': 0.10
    }
}

INDUSTRY_BENCHMARKS = {
    'Manufacturing': {
        'Data Infrastructure': 70,
        'AI Governance': 55,
        'Technology Stack': 65,
        'Talent': 60,
        'Leadership': 70,
        'Use Case Portfolio': 60,
        'Culture': 50
    },
    'Healthcare': {
        'Data Infrastructure': 75,
        'AI Governance': 80,
        'Technology Stack': 70,
        'Talent': 65,
        'Leadership': 75,
        'Use Case Portfolio': 65,
        'Culture': 55
    },
    'Retail': {
        'Data Infrastructure': 65,
        'AI Governance': 50,
        'Technology Stack': 60,
        'Talent': 55,
        'Leadership': 65,
        'Use Case Portfolio': 70,
        'Culture': 45
    },
    'Business Services': {
        'Data Infrastructure': 68,
        'AI Governance': 58,
        'Technology Stack': 63,
        'Talent': 63,
        'Leadership': 68,
        'Use Case Portfolio': 63,
        'Culture': 53
    },
    'Technology': {
        'Data Infrastructure': 85,
        'AI Governance': 75,
        'Technology Stack': 80,
        'Talent': 85,
        'Leadership': 80,
        'Use Case Portfolio': 75,
        'Culture': 70
    }
}

EXIT_WEIGHTS = {
    'Visible': 0.4,
    'Documented': 0.35,
    'Sustainable': 0.25
}

# --- Initialize Session State ---
# Ensures all session state variables exist before first access.
# This prevents errors on initial app load and allows widgets to manage their state.
session_state_defaults = {
    'company_name': "InnovateCo",
    'company_industry': 'Manufacturing',
    'raw_rating_data_infra': 2,
    'raw_rating_ai_governance': 1,
    'raw_rating_tech_stack': 3,
    'raw_rating_talent': 2,
    'raw_rating_leadership': 3,
    'raw_rating_use_case_portfolio': 1,
    'raw_rating_culture': 2,
    'alpha_param': 0.6,
    'beta_param': 0.15,
    'synergy_score': 50,
    'sensitivity_change_delta': 1,
    'exit_visible_score': 30,
    'exit_documented_score': 20,
    'exit_sustainable_score': 25,
    'idiosyncratic_readiness': 0.0, # Will be calculated
    'innovateco_dimension_scores_df': pd.DataFrame(), # Will be calculated
    'current_weights': {}, # Will be calculated
    'pe_org_ai_r_score': 0.0, # Will be calculated
    'gap_analysis_df': pd.DataFrame(), # Will be calculated
    'scenario_results_df': pd.DataFrame(), # Will be calculated
    'sensitivity_df': pd.DataFrame(), # Will be calculated
    'exit_ai_r_score': 0.0 # Will be calculated
}

for key, default_value in session_state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Helper Functions (Pure functions, no session_state access directly inside) ---

def get_dimension_weights(industry: str) -> dict:
    """Returns sector-specific weights or default if not found."""
    return SECTOR_DIMENSION_WEIGHTS.get(industry, DEFAULT_DIMENSION_WEIGHTS)

def calculate_idiosyncratic_readiness(raw_ratings: dict, industry: str) -> tuple[float, pd.DataFrame, dict]:
    """Calculates Idiosyncratic Readiness score and dimension scores."""
    weights = get_dimension_weights(industry)
    
    # Normalize raw ratings (1-5 scale) to (0-100 scale)
    normalized_ratings = {k: (v / 5) * 100 for k, v in raw_ratings.items()}
    
    # Calculate weighted sum
    weighted_sum = sum(normalized_ratings[dim] * weights.get(dim, 0) for dim in raw_ratings.keys())
    
    overall_score = weighted_sum

    dimension_data = []
    for dim, weight in weights.items():
        dimension_data.append({
            'Dimension': dim,
            'Raw Rating (1-5)': raw_ratings.get(dim, 0),
            'Normalized Score (0-100)': normalized_ratings.get(dim, 0),
            'Weight': weight
        })
    df = pd.DataFrame(dimension_data)
    
    return overall_score, df, weights

def calculate_pe_org_ai_r(idiosyncratic_readiness: float, systematic_opportunity: float,
                          alpha: float, beta: float, synergy_score: float) -> float:
    """Calculates the PE Org-AI-R Score."""
    
    pe_org_ai_r = (alpha * idiosyncratic_readiness) + \
                  ((1 - alpha) * systematic_opportunity) + \
                  (beta * synergy_score)

    # Clamp the score between 0 and 100
    return max(0, min(100, pe_org_ai_r))

def perform_gap_analysis(company_dimension_scores: pd.DataFrame, industry_benchmarks: dict, industry: str) -> pd.DataFrame:
    """Compares company scores to industry benchmarks and calculates gaps."""
    benchmarks = industry_benchmarks.get(industry, {})
    
    gap_data = []
    for _, row in company_dimension_scores.iterrows():
        dimension = row['Dimension']
        company_score = row['Normalized Score (0-100)']
        benchmark_score = benchmarks.get(dimension, 0)
        gap = benchmark_score - company_score
        
        priority = "Low"
        if gap > 20:
            priority = "High"
        elif gap > 10:
            priority = "Medium"
            
        gap_data.append({
            'Dimension': dimension,
            'Company Score': company_score,
            'Benchmark Score': benchmark_score,
            'Gap (Benchmark - Company)': gap,
            'Priority': priority
        })
    df = pd.DataFrame(gap_data)
    df['Color'] = df['Priority'].map({'High': 'red', 'Medium': 'orange', 'Low': 'green'})
    return df

def run_scenario_analysis(base_raw_ratings: dict, industry: str, alpha: float, beta: float, synergy_score: float,
                          scenario_definitions: dict) -> pd.DataFrame:
    """Runs different scenarios and calculates PE Org-AI-R scores for each."""
    
    scenario_results = []
    for scenario_name, changes in scenario_definitions.items():
        temp_raw_ratings = base_raw_ratings.copy()
        temp_synergy_score = synergy_score
        
        # Apply changes for the scenario
        for dim, change in changes.items():
            if dim == 'Synergy Score':
                temp_synergy_score = max(0, min(100, synergy_score + change))
            else:
                if dim in temp_raw_ratings:
                    temp_raw_ratings[dim] = max(1, min(5, temp_raw_ratings[dim] + change))
        
        idiosyncratic_readiness_score, _, _ = calculate_idiosyncratic_readiness(temp_raw_ratings, industry)
        systematic_opportunity_score = SYSTEMATIC_OPPORTUNITY_SCORES.get(industry, 0)
        
        org_ai_r_score = calculate_pe_org_ai_r(idiosyncratic_readiness_score, systematic_opportunity_score,
                                               alpha, beta, temp_synergy_score)
        
        scenario_results.append({'Scenario': scenario_name, 'PE Org-AI-R Score': org_ai_r_score})
        
    return pd.DataFrame(scenario_results)

def perform_sensitivity_analysis(base_raw_ratings: dict, industry: str, alpha: float, beta: float, synergy_score: float,
                                 change_delta: int) -> pd.DataFrame:
    """Performs sensitivity analysis on each dimension's raw rating."""
    
    base_idiosyncratic_readiness, _, _ = calculate_idiosyncratic_readiness(base_raw_ratings, industry)
    systematic_opportunity = SYSTEMATIC_OPPORTUNITY_SCORES.get(industry, 0)
    base_org_ai_r = calculate_pe_org_ai_r(base_idiosyncratic_readiness, systematic_opportunity, alpha, beta, synergy_score)
    
    sensitivity_data = []
    dimensions = list(base_raw_ratings.keys())
    
    # Sensitivity for each dimension
    for dim in dimensions:
        # Increase scenario
        temp_raw_ratings_increase = base_raw_ratings.copy()
        temp_raw_ratings_increase[dim] = max(1, min(5, base_raw_ratings[dim] + change_delta))
        
        idiosyncratic_readiness_increase, _, _ = calculate_idiosyncratic_readiness(temp_raw_ratings_increase, industry)
        org_ai_r_increase = calculate_pe_org_ai_r(idiosyncratic_readiness_increase, systematic_opportunity, alpha, beta, synergy_score)
        
        sensitivity_data.append({
            'Dimension Change': f"{dim} (+{change_delta} Raw Rating)",
            'Impact on PE Org-AI-R': org_ai_r_increase - base_org_ai_r
        })
        
        # Decrease scenario
        temp_raw_ratings_decrease = base_raw_ratings.copy()
        temp_raw_ratings_decrease[dim] = max(1, min(5, base_raw_ratings[dim] - change_delta))
        
        idiosyncratic_readiness_decrease, _, _ = calculate_idiosyncratic_readiness(temp_raw_ratings_decrease, industry)
        org_ai_r_decrease = calculate_pe_org_ai_r(idiosyncratic_readiness_decrease, systematic_opportunity, alpha, beta, synergy_score)
        
        sensitivity_data.append({
            'Dimension Change': f"{dim} (-{change_delta} Raw Rating)",
            'Impact on PE Org-AI-R': org_ai_r_decrease - base_org_ai_r
        })

    # Sensitivity for Synergy Score
    # Scale change_delta for synergy for a meaningful impact (e.g., 1 point raw rating change vs 10 points synergy score)
    synergy_change_value = change_delta * 10 
    
    temp_synergy_increase = max(0, min(100, synergy_score + synergy_change_value))
    org_ai_r_synergy_increase = calculate_pe_org_ai_r(base_idiosyncratic_readiness, systematic_opportunity, alpha, beta, temp_synergy_increase)
    sensitivity_data.append({
        'Dimension Change': f"Synergy Score (+{synergy_change_value})",
        'Impact on PE Org-AI-R': org_ai_r_synergy_increase - base_org_ai_r
    })
    
    temp_synergy_decrease = max(0, min(100, synergy_score - synergy_change_value))
    org_ai_r_synergy_decrease = calculate_pe_org_ai_r(base_idiosyncratic_readiness, systematic_opportunity, alpha, beta, temp_synergy_decrease)
    sensitivity_data.append({
        'Dimension Change': f"Synergy Score (-{synergy_change_value})",
        'Impact on PE Org-AI-R': org_ai_r_synergy_decrease - base_org_ai_r
    })

    df = pd.DataFrame(sensitivity_data)
    df = df.sort_values(by='Impact on PE Org-AI-R', ascending=True)
    return df

def calculate_exit_ai_r(visible_score: float, documented_score: float, sustainable_score: float, weights: dict) -> float:
    """Calculates the Exit-AI-R Score."""
    exit_ai_r = (visible_score * weights['Visible']) + \
                (documented_score * weights['Documented']) + \
                (sustainable_score * weights['Sustainable'])
    return exit_ai_r

# --- Streamlit Application Flow ---

# 1. Welcome & Company Selection
st.header("1. Welcome & Company Selection")
st.markdown("""
Alex, let's start by defining the target company and its industry sector. These inputs will dynamically
adjust our analytical framework, including industry-specific benchmarks and dimension weights.
""")

col1, col2 = st.columns(2)
with col1:
    st.text_input( # Changed to allow widget to manage session_state directly
        "Target Company Name",
        value=st.session_state.company_name,
        key="company_name" # Key is 'company_name' to directly modify st.session_state.company_name
    )
with col2:
    st.selectbox( # Changed to allow widget to manage session_state directly
        "Company Industry Sector",
        options=list(SYSTEMATIC_OPPORTUNITY_SCORES.keys()),
        index=list(SYSTEMATIC_OPPORTUNITY_SCORES.keys()).index(st.session_state.company_industry),
        key="company_industry" # Key is 'company_industry'
    )

st.divider()

# 2. Define Static Framework Parameters
st.header("2. Define Static Framework Parameters")
st.markdown(f"""
Before diving into `InnovateCo`'s specifics, let's review the foundational data that underpins our analysis.
This includes the `Systematic Opportunity Score` for the **{st.session_state.company_industry}** sector,
the `AI Readiness Dimension Weights` (adjusted for **{st.session_state.company_industry}**),
and `Industry Benchmarks` against which we will compare {st.session_state.company_name}.
""")

with st.expander("View Framework Parameters"):
    st.markdown("##### Systematic Opportunity Scores by Industry")
    st.dataframe(pd.DataFrame(list(SYSTEMATIC_OPPORTUNITY_SCORES.items()), columns=['Industry', 'Score']), width='stretch') # FIX: use_container_width -> width
    
    st.markdown(f"##### AI Readiness Dimension Weights for {st.session_state.company_industry}")
    current_weights_df = pd.DataFrame(list(get_dimension_weights(st.session_state.company_industry).items()), columns=['Dimension', 'Weight'])
    st.dataframe(current_weights_df, width='stretch') # FIX: use_container_width -> width

    st.markdown(f"##### Industry Benchmarks for {st.session_state.company_industry}")
    benchmark_df = pd.DataFrame(list(INDUSTRY_BENCHMARKS.get(st.session_state.company_industry, {}).items()), columns=['Dimension', 'Benchmark Score (0-100)'])
    st.dataframe(benchmark_df, width='stretch') # FIX: use_container_width -> width

st.divider()

# 3. Collect Raw Idiosyncratic Readiness Ratings
st.header("3. Collect Raw Idiosyncratic Readiness Ratings")
st.markdown(f"""
Now, Alex, it's time to translate your due diligence findings for **{st.session_state.company_name}** into
quantifiable scores. Use the sliders below to rate {st.session_state.company_name}'s current capabilities
across seven critical AI readiness dimensions on a scale of 1 to 5.
""")

dimension_ratings_keys = {
    'Data Infrastructure': "raw_rating_data_infra",
    'AI Governance': "raw_rating_ai_governance",
    'Technology Stack': "raw_rating_tech_stack",
    'Talent': "raw_rating_talent",
    'Leadership': "raw_rating_leadership",
    'Use Case Portfolio': "raw_rating_use_case_portfolio",
    'Culture': "raw_rating_culture"
}

# FIX: Removed direct assignment to st.session_state in slider loop to avoid StreamlitAPIException
cols = st.columns(2)
for i, (dim, key) in enumerate(dimension_ratings_keys.items()):
    with cols[i % 2]:
        st.slider(
            f"{dim} Rating (1-5)",
            min_value=1,
            max_value=5,
            step=1,
            value=st.session_state[key], # This reads the current value from session state
            key=key # This updates st.session_state[key] directly when the slider is moved
        )

# Populate raw_dimension_ratings from session state after all sliders have been rendered
raw_ratings_from_session = {dim: st.session_state[key] for dim, key in dimension_ratings_keys.items()}
st.session_state.raw_dimension_ratings = raw_ratings_from_session 
st.divider()

# 4. Calculate Normalized Idiosyncratic Readiness
st.header("4. Calculate Normalized Idiosyncratic Readiness")
st.markdown(f"""
Based on your raw ratings, we now calculate **{st.session_state.company_name}**'s `Idiosyncratic Readiness Score`.
This score reflects the company's internal AI capabilities, weighted by the importance of each dimension
for the **{st.session_state.company_industry}** sector.
""")

st.markdown(r"""
$$ IdiosyncraticReadiness = \sum_{i=1}^{7} w_i \cdot \left( \frac{\text{Raw Rating}_i}{5} \times 100 \right) $$
""")
# FIX: Use raw f-string and double braces for literal curly braces in LaTeX
st.markdown(rf"""
where:
*   $w_i$ is the industry-specific weight for dimension $i$.
*   $\text{{Raw Rating}}_i$ is your assessment (1-5) for dimension $i$.
*   The raw rating is normalized to a 0-100 scale before weighting.
""")

idiosyncratic_readiness, innovateco_dimension_scores_df, current_weights = calculate_idiosyncratic_readiness(
    st.session_state.raw_dimension_ratings, st.session_state.company_industry
)
st.session_state.idiosyncratic_readiness = idiosyncratic_readiness
st.session_state.innovateco_dimension_scores_df = innovateco_dimension_scores_df
st.session_state.current_weights = current_weights

st.metric(label=f"{st.session_state.company_name}'s Idiosyncratic Readiness Score",
          value=f"{st.session_state.idiosyncratic_readiness:.2f}")

st.markdown("##### Detailed Idiosyncratic Readiness Scores by Dimension")
st.dataframe(st.session_state.innovateco_dimension_scores_df.set_index('Dimension'), width='stretch') # FIX: use_container_width -> width

# Plot: Idiosyncratic Readiness Dimension Scores
fig_idio, ax_idio = plt.subplots(figsize=(10, 6))
sns.barplot(x='Normalized Score (0-100)', y='Dimension', data=st.session_state.innovateco_dimension_scores_df.sort_values(by='Normalized Score (0-100)', ascending=False), palette='viridis', ax=ax_idio)
ax_idio.set_title(f"{st.session_state.company_name}'s Idiosyncratic AI Readiness by Dimension")
ax_idio.set_xlabel("Score (0-100)")
ax_idio.set_ylabel("AI Readiness Dimension")
st.pyplot(fig_idio)
plt.close(fig_idio) # Close figure to prevent memory issues

st.info(f"""
**Analyst Note:** This chart provides a granular view of {st.session_state.company_name}'s internal AI capabilities.
We can immediately see strengths (e.g., '{st.session_state.innovateco_dimension_scores_df.iloc[0]['Dimension']}') and
areas needing significant improvement (e.g., '{st.session_state.innovateco_dimension_scores_df.iloc[-1]['Dimension']}').
This initial assessment helps us focus our deep-dive due diligence.
""")
st.divider()

# 5. Compute the Overall PE Org-AI-R Score
st.header("5. Compute the Overall PE Org-AI-R Score")
st.markdown(f"""
The `PE Org-AI-R Score` is VentureBridge Capital's proprietary metric. It combines {st.session_state.company_name}'s
`Idiosyncratic Readiness` (internal capabilities) with the `Systematic Opportunity` of its industry sector,
and an additional `Synergy` component, which quantifies the amplification potential.
""")

st.markdown(r"""
$$PE \text{ Org-AI-R} = \alpha \cdot IdiosyncraticReadiness + (1 - \alpha) \cdot SystematicOpportunity + \beta \cdot Synergy$$
""")
# FIX: Use raw f-string and double braces for literal curly braces in LaTeX
st.markdown(rf"""
where:
*   $IdiosyncraticReadiness$ is {st.session_state.company_name}'s internal AI capability score (0-100).
*   $SystematicOpportunity$ is the AI market opportunity score for the {st.session_state.company_industry} sector (0-100).
*   $\alpha$ (alpha) is the weight given to organizational factors, reflecting our focus on internal vs. external drivers.
*   $\beta$ (beta) is the synergy coefficient, determining the impact of perceived alignment and amplification.
*   $Synergy$ is a conceptual score (0-100) reflecting how well InnovateCo's internal readiness aligns with and amplifies market potential.
""")

systematic_opportunity_score = SYSTEMATIC_OPPORTUNITY_SCORES.get(st.session_state.company_industry, 0)

col1_org, col2_org = st.columns(2)
with col1_org:
    st.slider( # Removed direct assignment to st.session_state
        "Weight on Organizational Factors ($\alpha$)",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=st.session_state.alpha_param,
        key="alpha_param" # Key is 'alpha_param'
    )
    # FIX: Use raw f-string and double braces for literal curly braces in LaTeX
    st.markdown(rf"""
    <small><i>Adjusting $\alpha$ shifts the focus between internal capabilities (IdiosyncraticReadiness)
    and external market opportunity (SystematicOpportunity). A higher $\alpha$ means we prioritize
    the company's internal strengths more heavily.</i></small>
    """, unsafe_allow_html=True)
with col2_org:
    st.slider( # Removed direct assignment to st.session_state
        "Synergy Coefficient ($\beta$)",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        value=st.session_state.beta_param,
        key="beta_param" # Key is 'beta_param'
    )
    # FIX: Use raw f-string and double braces for literal curly braces in LaTeX
    st.markdown(rf"""
    <small><i>$\beta$ controls the impact of the 'Synergy Score'. A higher $\beta$ amplifies
    how much the alignment between internal capabilities and market potential affects the overall score.</i></small>
    """, unsafe_allow_html=True)

st.slider( # Removed direct assignment to st.session_state
    "Synergy Score (0-100)",
    min_value=0,
    max_value=100,
    step=1,
    value=st.session_state.synergy_score,
    key="synergy_score" # Key is 'synergy_score'
)
st.markdown(f"""
<small><i>This score represents Alex's expert assessment of how well {st.session_state.company_name}'s existing (or potential)
AI capabilities can capitalize on its industry's AI opportunity.</i></small>
""", unsafe_allow_html=True)

pe_org_ai_r_score = calculate_pe_org_ai_r(
    st.session_state.idiosyncratic_readiness,
    systematic_opportunity_score,
    st.session_state.alpha_param,
    st.session_state.beta_param,
    st.session_state.synergy_score
)
st.session_state.pe_org_ai_r_score = pe_org_ai_r_score

st.metric(label=f"Overall PE Org-AI-R Score for {st.session_state.company_name}",
          value=f"{st.session_state.pe_org_ai_r_score:.2f}")

st.info(f"""
**Analyst Note:** A PE Org-AI-R score of **{st.session_state.pe_org_ai_r_score:.2f}** for **{st.session_state.company_name}**
provides VentureBridge Capital with a quantitative baseline. This helps us quickly categorize targets (e.g., 'AI leader',
'AI transformation opportunity', 'AI laggard') and prioritize further due diligence efforts.
""")
st.divider()

# 6. Perform Gap Analysis Against Industry Benchmarks
st.header("6. Perform Gap Analysis Against Industry Benchmarks")
st.markdown(f"""
To identify strategic investment areas, we need to compare **{st.session_state.company_name}**'s
`Idiosyncratic Readiness` across each dimension against industry benchmarks for the
**{st.session_state.company_industry}** sector. This highlights specific strengths and, more importantly,
critical `AI Readiness Gaps`.
""")

st.markdown(r"""
$$Gap_k = D_k^{benchmark} - D_k^{current}$$
""")
# FIX: Use raw f-string and double braces for literal curly braces in LaTeX
st.markdown(rf"""
where:
*   $D_k^{{benchmark}}$ is the benchmark score for dimension $k$ in the {st.session_state.company_industry} sector.
*   $D_k^{{current}}$ is {st.session_state.company_name}'s current normalized score for dimension $k$.
A positive gap means {st.session_state.company_name} is lagging behind the industry benchmark in that dimension.
""")

gap_analysis_df = perform_gap_analysis(
    st.session_state.innovateco_dimension_scores_df,
    INDUSTRY_BENCHMARKS,
    st.session_state.company_industry
)
st.session_state.gap_analysis_df = gap_analysis_df

st.markdown("##### AI Readiness Gap Analysis")
st.dataframe(st.session_state.gap_analysis_df[['Dimension', 'Company Score', 'Benchmark Score', 'Gap (Benchmark - Company)', 'Priority']], width='stretch') # FIX: use_container_width -> width

# Plot 1: Comparative Bar Chart
df_plot_compare = st.session_state.gap_analysis_df[['Dimension', 'Company Score', 'Benchmark Score']].melt(id_vars='Dimension', var_name='Type', value_name='Score')
fig_comp, ax_comp = plt.subplots(figsize=(12, 7))
sns.barplot(x='Dimension', y='Score', hue='Type', data=df_plot_compare, palette={'Company Score': 'skyblue', 'Benchmark Score': 'orange'}, ax=ax_comp)
ax_comp.set_title(f"{st.session_state.company_name} vs. {st.session_state.company_industry} Benchmarks by AI Readiness Dimension")
ax_comp.set_xlabel("AI Readiness Dimension")
ax_comp.set_ylabel("Score (0-100)")
ax_comp.set_ylim(0, 100)
ax_comp.tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig_comp)
plt.close(fig_comp)

# Plot 2: AI Readiness Gaps
fig_gaps, ax_gaps = plt.subplots(figsize=(12, 7))
sns.barplot(x='Gap (Benchmark - Company)', y='Dimension', data=st.session_state.gap_analysis_df.sort_values(by='Gap (Benchmark - Company)', ascending=False),
            palette=st.session_state.gap_analysis_df['Color'].tolist(), ax=ax_gaps)
ax_gaps.set_title(f"AI Readiness Gaps for {st.session_state.company_name} Against {st.session_state.company_industry} Benchmarks")
ax_gaps.set_xlabel("Gap (Benchmark Score - Company Score)")
ax_gaps.set_ylabel("AI Readiness Dimension")
ax_gaps.axvline(0, color='grey', linestyle='--')
plt.tight_layout()
st.pyplot(fig_gaps)
plt.close(fig_gaps)

st.info(f"""
**Analyst Note:** The gap analysis clearly shows where **{st.session_state.company_name}** underperforms.
Dimensions with 'High' priority gaps, such as '{st.session_state.gap_analysis_df[st.session_state.gap_analysis_df['Priority'] == 'High'].iloc[0]['Dimension'] if not st.session_state.gap_analysis_df[st.session_state.gap_analysis_df['Priority'] == 'High'].empty else 'N/A'}',
represent critical areas for immediate investment to catch up to industry peers and drive value creation.
""")
st.divider()

# 7. Conduct Scenario Analysis for Strategic Planning
st.header("7. Conduct Scenario Analysis for Strategic Planning")
st.markdown(f"""
Alex, let's explore how strategic interventions could impact **{st.session_state.company_name}**'s
`PE Org-AI-R Score`. By modeling different investment scenarios, we can quantify the potential
upside of targeted AI initiatives and develop a strategic roadmap for the Portfolio Manager.
""")

scenario_definitions = {
    'Base Case (Current)': {}, # No changes from current state
    'Optimistic Scenario': {
        'Data Infrastructure': 2, # +2 improvement in raw rating
        'Technology Stack': 1,
        'Use Case Portfolio': 2,
        'Synergy Score': 20 # +20 improvement in synergy score
    },
    'Moderate Scenario': {
        'Data Infrastructure': 1,
        'Talent': 1,
        'Synergy Score': 10
    },
    'Pessimistic Scenario': {
        'AI Governance': -1, # Decline in governance (e.g., due to mismanagement)
        'Culture': -1,
        'Synergy Score': -10
    }
}

scenario_results_df = run_scenario_analysis(
    st.session_state.raw_dimension_ratings,
    st.session_state.company_industry,
    st.session_state.alpha_param,
    st.session_state.beta_param,
    st.session_state.synergy_score,
    scenario_definitions
)
st.session_state.scenario_results_df = scenario_results_df

# Add the current (Base Case) score to the scenario_results_df for comparison if not already included
# FIX: Ensure 'Base Case (Current)' is always first for consistent indexing
if 'Base Case (Current)' in st.session_state.scenario_results_df['Scenario'].values:
    # If it exists, remove it and re-add at the top to ensure it's always the first row if already present
    st.session_state.scenario_results_df = st.session_state.scenario_results_df[st.session_state.scenario_results_df['Scenario'] != 'Base Case (Current)']

base_case_row = pd.DataFrame([{'Scenario': 'Base Case (Current)', 'PE Org-AI-R Score': st.session_state.pe_org_ai_r_score}])
st.session_state.scenario_results_df = pd.concat([base_case_row, st.session_state.scenario_results_df]).reset_index(drop=True)


st.markdown("##### PE Org-AI-R Score Under Different Scenarios")
st.dataframe(st.session_state.scenario_results_df, width='stretch') # FIX: use_container_width -> width

# Plot: Scenario Analysis
fig_scenario, ax_scenario = plt.subplots(figsize=(10, 6))
sns.barplot(x='Scenario', y='PE Org-AI-R Score', data=st.session_state.scenario_results_df, palette='coolwarm', ax=ax_scenario)
ax_scenario.set_title(f"PE Org-AI-R Score for {st.session_state.company_name} Under Different Scenarios")
ax_scenario.set_xlabel("Scenario")
ax_scenario.set_ylabel("PE Org-AI-R Score (0-100)")
ax_scenario.set_ylim(0, 100)
st.pyplot(fig_scenario)
plt.close(fig_scenario)

st.info(f"""
**Analyst Note:** The scenario analysis clearly demonstrates the impact of various strategic initiatives.
For instance, an **'Optimistic Scenario'** could increase {st.session_state.company_name}'s Org-AI-R score
to **{st.session_state.scenario_results_df[st.session_state.scenario_results_df['Scenario'] == 'Optimistic Scenario']['PE Org-AI-R Score'].iloc[0]:.2f}**,
quantifying the potential return on AI-related investments for VentureBridge Capital.
""")
st.divider()

# 8. Perform Sensitivity Analysis of Key Dimensions
st.header("8. Perform Sensitivity Analysis of Key Dimensions")
st.markdown(f"""
To efficiently allocate resources, Alex, we need to understand which dimensions are `swing factors` –
those that, when improved, have the most significant impact on **{st.session_state.company_name}**'s
overall `PE Org-AI-R Score`. This `Sensitivity Analysis` will guide our prioritization.
""")

st.slider( # Removed direct assignment to st.session_state
    r"Raw Rating Change for Sensitivity Analysis ($\pm$ points)", # FIX: Made label a raw string to avoid SyntaxWarning for \p
    min_value=1,
    max_value=2,
    step=1,
    value=st.session_state.sensitivity_change_delta,
    key="sensitivity_change_delta" # Key is 'sensitivity_change_delta'
)
st.markdown(f"""
<small><i>This slider defines the hypothetical improvement (or decline) in raw ratings we test for each dimension.
A +1 change means a raw rating of 3 becomes 4.</i></small>
""", unsafe_allow_html=True)

sensitivity_df = perform_sensitivity_analysis(
    st.session_state.raw_dimension_ratings,
    st.session_state.company_industry,
    st.session_state.alpha_param,
    st.session_state.beta_param,
    st.session_state.synergy_score,
    st.session_state.sensitivity_change_delta
)
st.session_state.sensitivity_df = sensitivity_df

st.markdown("##### Sensitivity of PE Org-AI-R to Dimension Changes")
st.dataframe(st.session_state.sensitivity_df, width='stretch') # FIX: use_container_width -> width

# Plot: Sensitivity Analysis
fig_sens, ax_sens = plt.subplots(figsize=(12, 8))
colors = ['red' if x < 0 else 'green' for x in st.session_state.sensitivity_df['Impact on PE Org-AI-R']]
sns.barplot(x='Impact on PE Org-AI-R', y='Dimension Change', data=st.session_state.sensitivity_df, palette=colors, ax=ax_sens)
ax_sens.set_title(f"Sensitivity of PE Org-AI-R to Changes in {st.session_state.company_name}'s AI Dimensions")
ax_sens.set_xlabel("Change in PE Org-AI-R Score")
ax_sens.set_ylabel("AI Dimension / Synergy Change")
ax_sens.axvline(0, color='grey', linestyle='--')
plt.tight_layout()
st.pyplot(fig_sens)
plt.close(fig_sens)

st.info(f"""
**Analyst Note:** This analysis reveals the **'swing factors'** for **{st.session_state.company_name}**.
For example, improving '{st.session_state.sensitivity_df.iloc[-1]['Dimension Change']}' by
the specified delta would yield the largest positive impact of **{st.session_state.sensitivity_df.iloc[-1]['Impact on PE Org-AI-R']:.2f}**
points on the overall `PE Org-AI-R Score`. This insight is crucial for prioritizing investment focus.
""")
st.divider()

# 9. Evaluate Exit-Readiness
st.header("9. Evaluate Exit-Readiness")
st.markdown(f"""
Finally, Alex, let's assess **{st.session_state.company_name}**'s `Exit-AI-R Score`. This score reflects how attractive
the company's AI capabilities would be to a future buyer, considering how `Visible`, `Documented`,
and `Sustainable` these capabilities are. This informs our long-term exit strategy.
""")

st.markdown(r"""
$$Exit\text{-AI-R} = w_{Visible} \cdot Visible + w_{Documented} \cdot Documented + w_{Sustainable} \cdot Sustainable$$
""")
st.markdown(f"""
where:
*   $Visible$ is the score (0-100) reflecting how easily a buyer can perceive {st.session_state.company_name}'s AI capabilities.
*   $Documented$ is the score (0-100) reflecting how well AI processes and results are formally documented.
*   $Sustainable$ is the score (0-100) reflecting the longevity and robustness of AI capabilities.
*   $w_x$ are the predefined weights for each component.
""")

col1_exit, col2_exit, col3_exit = st.columns(3)
with col1_exit:
    st.slider( # Removed direct assignment to st.session_state
        "Visible Score (0-100)",
        min_value=0,
        max_value=100,
        step=1,
        value=st.session_state.exit_visible_score,
        key="exit_visible_score" # Key is 'exit_visible_score'
    )
with col2_exit:
    st.slider( # Removed direct assignment to st.session_state
        "Documented Score (0-100)",
        min_value=0,
        max_value=100,
        step=1,
        value=st.session_state.exit_documented_score,
        key="exit_documented_score" # Key is 'exit_documented_score'
    )
with col3_exit:
    st.slider( # Removed direct assignment to st.session_state
        "Sustainable Score (0-100)",
        min_value=0,
        max_value=100,
        step=1,
        value=st.session_state.exit_sustainable_score,
        key="exit_sustainable_score" # Key is 'exit_sustainable_score'
    )

exit_ai_r_score = calculate_exit_ai_r(
    st.session_state.exit_visible_score,
    st.session_state.exit_documented_score,
    st.session_state.exit_sustainable_score,
    EXIT_WEIGHTS
)
st.session_state.exit_ai_r_score = exit_ai_r_score

st.metric(label=f"Overall Exit-AI-R Score for {st.session_state.company_name}",
          value=f"{st.session_state.exit_ai_r_score:.2f}")

exit_component_data = pd.DataFrame({
    'Component': ['Visible', 'Documented', 'Sustainable'],
    'Score': [st.session_state.exit_visible_score, st.session_state.exit_documented_score, st.session_state.exit_sustainable_score],
    'Weight': [EXIT_WEIGHTS['Visible'], EXIT_WEIGHTS['Documented'], EXIT_WEIGHTS['Sustainable']]
})

# Plot: Exit-AI-R Component Scores
fig_exit, ax_exit = plt.subplots(figsize=(10, 6))
sns.barplot(x='Component', y='Score', data=exit_component_data, palette='cool', ax=ax_exit)
ax_exit.set_title(f"Exit-AI-R Component Scores for {st.session_state.company_name}")
ax_exit.set_xlabel("Exit-Readiness Component")
ax_exit.set_ylabel("Score (0-100)")
ax_exit.set_ylim(0, 100)
# Add weight annotations
for index, row in exit_component_data.iterrows():
    ax_exit.text(index, row['Score'] + 5, f"Weight: {row['Weight']:.2f}", color='black', ha="center")
st.pyplot(fig_exit)
plt.close(fig_exit)

st.info(f"""
**Analyst Note:** An Exit-AI-R score of **{st.session_state.exit_ai_r_score:.2f}** indicates {st.session_state.company_name}'s
"AI sellability." High scores in 'Visible' and 'Documented' suggest a clear AI story for buyers,
while 'Sustainable' indicates long-term value. Areas with lower scores represent opportunities
to enhance the exit narrative and increase valuation.
""")
st.divider()

st.markdown(f"""
---
### Simulation Complete: Actionable Insights for VentureBridge Capital

Congratulations, Alex! You have completed the comprehensive PE-AI readiness simulation for **{st.session_state.company_name}**.
Through this interactive journey, you've generated crucial quantitative insights:

*   **Current State (PE Org-AI-R Score):** **{st.session_state.pe_org_ai_r_score:.2f}**
*   **Key Gaps:** Identified specific AI dimensions where **{st.session_state.company_name}** lags its industry.
*   **Strategic Scenarios:** Modeled potential future PE Org-AI-R scores under various investment hypotheses.
*   **Prioritization:** Pinpointed the "swing factors" for AI investment that yield the highest impact.
*   **Exit Potential (Exit-AI-R Score):** **{st.session_state.exit_ai_r_score:.2f}**, indicating its attractiveness to future buyers.

These insights empower VentureBridge Capital to make informed multi-million dollar investment decisions,
design targeted value creation plans, and craft a compelling AI-driven exit strategy for **{st.session_state.company_name}**.
""")
