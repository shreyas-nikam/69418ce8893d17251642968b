
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock

# --- File Setup (would typically be in a conftest.py or setup script) ---
# This part is for ensuring the app files exist for AppTest to find them.
# In a real testing environment, these files would already exist in your project structure.

def write_file_to_github(filepath, content):
    """Helper to create files for testing."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)

# Define file contents as provided in the problem description
requirements_txt_contents = """
streamlit==1.33.0
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.4
seaborn==0.13.2
"""

utils_py_contents = """
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Global Constants (Static Framework Data) ---
# Systematic Opportunity Scores by Industry
SYSTEMATIC_OPPORTUNITY_SCORES = {
    'Manufacturing': 80,
    'Healthcare': 70,
    'Retail': 65,
    'Business Services': 75,
    'Technology': 90
}

# Default weights for dimensions (used if no specific sector weights)
DEFAULT_DIMENSION_WEIGHTS = {
    'Data Infrastructure': 0.15,
    'AI Governance': 0.10,
    'Technology Stack': 0.20,
    'Talent': 0.15,
    'Leadership': 0.10,
    'Use Case Portfolio': 0.20,
    'Culture': 0.10
}

# Sector-specific adjustments for dimension weights
SECTOR_DIMENSION_WEIGHTS = {
    'Manufacturing': {
        'Data Infrastructure': 0.20, # Higher importance for OT data, supply chain
        'AI Governance': 0.10,
        'Technology Stack': 0.15,
        'Talent': 0.15,
        'Leadership': 0.10,
        'Use Case Portfolio': 0.20, # Predictive maintenance, process optimization
        'Culture': 0.10
    },
    'Healthcare': {
        'Data Infrastructure': 0.15,
        'AI Governance': 0.20, # Critical for regulatory compliance, patient privacy
        'Technology Stack': 0.15,
        'Talent': 0.15,
        'Leadership': 0.10,
        'Use Case Portfolio': 0.15,
        'Culture': 0.10
    },
    'Retail': {
        'Data Infrastructure': 0.15,
        'AI Governance': 0.10,
        'Technology Stack': 0.20,
        'Talent': 0.10,
        'Leadership': 0.10,
        'Use Case Portfolio': 0.25, # Personalization, inventory optimization
        'Culture': 0.10
    },
    'Business Services': {
        'Data Infrastructure': 0.10,
        'AI Governance': 0.10,
        'Technology Stack': 0.20,
        'Talent': 0.20, # High importance for service delivery
        'Leadership': 0.15,
        'Use Case Portfolio': 0.15,
        'Culture': 0.10
    },
    'Technology': {
        'Data Infrastructure': 0.15,
        'AI Governance': 0.10,
        'Technology Stack': 0.25, # Core to the business
        'Talent': 0.20,
        'Leadership': 0.15,
        'Use Case Portfolio': 0.10,
        'Culture': 0.05
    }
}

# Industry Benchmarks for each dimension (0-100 scale)
INDUSTRY_BENCHMARKS = {
    'Data Infrastructure': 65,
    'AI Governance': 50,
    'Technology Stack': 70,
    'Talent': 60,
    'Leadership': 75,
    'Use Case Portfolio': 55,
    'Culture': 50
}

# Exit-AI-R Component Weights
EXIT_READINESS_WEIGHTS = {
    'Visible': 0.4,
    'Documented': 0.35,
    'Sustainable': 0.25
}

# List of dimensions for consistent ordering
DIMENSIONS = list(DEFAULT_DIMENSION_WEIGHTS.keys())


# --- Calculation Functions ---

def calculate_idiosyncratic_readiness(raw_ratings: dict, industry: str) -> (float, pd.DataFrame, dict):
    """
    Calculates the normalized Idiosyncratic Readiness score and dimension-specific scores.
    Args:
        raw_ratings (dict): Dictionary of raw ratings (1-5) for each dimension.
        industry (str): The industry sector of the company.
    Returns:
        tuple: (overall_score, dimension_scores_df, applied_weights_dict)
    """
    applied_weights = SECTOR_DIMENSION_WEIGHTS.get(industry, DEFAULT_DIMENSION_WEIGHTS)

    # Ensure weights sum to 1 for normalization
    weight_sum = sum(applied_weights.values())
    if abs(weight_sum - 1.0) > 1e-6: # Check if sum is close to 1
        applied_weights = {dim: w / weight_sum for dim, w in applied_weights.items()}

    dimension_scores = []
    for dim in DIMENSIONS:
        raw_rating = raw_ratings.get(dim, 1) # Default to 1 if not found
        weight = applied_weights.get(dim, 0)
        # Scale raw rating (1-5) to (0-100) and apply weight
        scaled_score = ((raw_rating - 1) / 4) * 100 # Scale 1-5 to 0-100
        dimension_scores.append({
            'Dimension': dim,
            'Raw Rating (1-5)': raw_rating,
            'Scaled Score (0-100)': scaled_score,
            'Weight': weight,
            'Weighted Score': scaled_score * weight
        })
    
    dimension_scores_df = pd.DataFrame(dimension_scores)
    
    # The overall score is the sum of weighted scaled scores.
    # It's already implicitly scaled to 0-100 by the 'Scaled Score (0-100)' calculation.
    overall_score = dimension_scores_df['Weighted Score'].sum()
    
    return overall_score, dimension_scores_df, applied_weights


def calculate_pe_org_ai_r(
    idiosyncratic_readiness_score: float,
    systematic_opportunity_score: float,
    alpha: float,
    beta: float,
    synergy_score: float
) -> float:
    """
    Calculates the PE Org-AI-R Score.
    Args:
        idiosyncratic_readiness_score (float): The calculated Idiosyncratic Readiness score (0-100).
        systematic_opportunity_score (float): The industry-specific Systematic Opportunity score (0-100).
        alpha (float): Weight on Organizational Factors (0.0-1.0).
        beta (float): Synergy Coefficient (0.0-1.0).
        synergy_score (float): Conceptual Synergy Score (0-100).
    Returns:
        float: The calculated PE Org-AI-R Score.
    """
    org_ai_r = (
        alpha * idiosyncratic_readiness_score +
        (1 - alpha) * systematic_opportunity_score +
        beta * (synergy_score * (idiosyncratic_readiness_score / 100) * (systematic_opportunity_score / 100)) # Synergy scaled by readiness and opportunity
    )
    # Cap the score at 100
    return min(org_ai_r, 100.0)


def perform_gap_analysis(innovateco_dimension_scores_df: pd.DataFrame, industry_benchmarks: dict) -> pd.DataFrame:
    """
    Performs gap analysis comparing InnovateCo's scores to industry benchmarks.
    Args:
        innovateco_dimension_scores_df (pd.DataFrame): DataFrame with InnovateCo's scaled dimension scores.
        industry_benchmarks (dict): Dictionary of industry benchmark scores for each dimension.
    Returns:
        pd.DataFrame: DataFrame with gap analysis results.
    """
    gap_data = []
    for index, row in innovateco_dimension_scores_df.iterrows():
        dimension = row['Dimension']
        company_score = row['Scaled Score (0-100)']
        benchmark_score = industry_benchmarks.get(dimension, 0)
        gap = benchmark_score - company_score
        
        priority = 'Low'
        if gap > 20:
            priority = 'High'
        elif gap > 10:
            priority = 'Medium'

        gap_data.append({
            'Dimension': dimension,
            'InnovateCo Score': company_score,
            'Benchmark Score': benchmark_score,
            'Gap': gap,
            'Priority': priority
        })
    return pd.DataFrame(gap_data)


def run_scenario_analysis(
    base_raw_ratings: dict,
    industry: str,
    alpha: float,
    beta: float,
    base_synergy_score: float,
    scenario_definitions: dict
) -> pd.DataFrame:
    """
    Runs scenario analysis for the PE Org-AI-R score.
    Args:
        base_raw_ratings (dict): InnovateCo's current raw dimension ratings.
        industry (str): The industry sector.
        alpha (float): Current alpha parameter.
        beta (float): Current beta parameter.
        base_synergy_score (float): Current synergy score.
        scenario_definitions (dict): Dictionary defining rating deltas and synergy adjustments for each scenario.
    Returns:
        pd.DataFrame: DataFrame with scenario results.
    """
    scenario_results = []
    base_idiosyncratic_score, _, _ = calculate_idiosyncratic_readiness(base_raw_ratings, industry)
    base_systematic_opportunity = SYSTEMATIC_OPPORTUNITY_SCORES[industry]
    base_org_ai_r = calculate_pe_org_ai_r(base_idiosyncratic_score, base_systematic_opportunity, alpha, beta, base_synergy_score)

    scenario_results.append({
        'Scenario': 'Base Case',
        'PE Org-AI-R Score': base_org_ai_r
    })

    for scenario_name, params in scenario_definitions.items():
        adjusted_raw_ratings = base_raw_ratings.copy()
        for dim, delta in params.get('rating_deltas', {}).items():
            current_rating = adjusted_raw_ratings.get(dim, 1)
            adjusted_raw_ratings[dim] = min(max(current_rating + delta, 1), 5) # Ratings clamped between 1 and 5
        
        adjusted_synergy_score = base_synergy_score + params.get('synergy_delta', 0)
        adjusted_synergy_score = min(max(adjusted_synergy_score, 0), 100) # Synergy clamped between 0 and 100

        idiosyncratic_score, _, _ = calculate_idiosyncratic_readiness(adjusted_raw_ratings, industry)
        systematic_opportunity = SYSTEMATIC_OPPORTUNITY_SCORES[industry]
        org_ai_r = calculate_pe_org_ai_r(idiosyncratic_score, systematic_opportunity, alpha, beta, adjusted_synergy_score)
        
        scenario_results.append({
            'Scenario': scenario_name,
            'PE Org-AI-R Score': org_ai_r
        })
    
    return pd.DataFrame(scenario_results)


def perform_sensitivity_analysis(
    base_raw_ratings: dict,
    industry: str,
    alpha: float,
    beta: float,
    synergy_score: float,
    change_delta: int
) -> pd.DataFrame:
    """
    Performs sensitivity analysis on each dimension's impact on the PE Org-AI-R Score.
    Args:
        base_raw_ratings (dict): InnovateCo's current raw dimension ratings.
        industry (str): The industry sector.
        alpha (float): Current alpha parameter.
        beta (float): Current beta parameter.
        synergy_score (float): Current synergy score.
        change_delta (int): Points to change raw rating by (e.g., +/- 1 or 2).
    Returns:
        pd.DataFrame: DataFrame with sensitivity analysis results.
    """
    sensitivity_results = []
    base_idiosyncratic_score, _, _ = calculate_idiosyncratic_readiness(base_raw_ratings, industry)
    base_systematic_opportunity = SYSTEMATIC_OPPORTUNITY_SCORES[industry]
    base_org_ai_r = calculate_pe_org_ai_r(base_idiosyncratic_score, base_systematic_opportunity, alpha, beta, synergy_score)

    # Test each dimension
    for dim in DIMENSIONS:
        original_rating = base_raw_ratings.get(dim, 1)

        # Increase scenario
        increased_ratings = base_raw_ratings.copy()
        increased_ratings[dim] = min(original_rating + change_delta, 5)
        increased_idiosyncratic_score, _, _ = calculate_idiosyncratic_readiness(increased_ratings, industry)
        increased_org_ai_r = calculate_pe_org_ai_r(increased_idiosyncratic_score, base_systematic_opportunity, alpha, beta, synergy_score)
        impact_increase = increased_org_ai_r - base_org_ai_r
        
        sensitivity_results.append({
            'Dimension': dim,
            'Change': f'{dim} (+{change_delta} Raw Rating)',
            'Impact on PE Org-AI-R': impact_increase
        })

        # Decrease scenario
        decreased_ratings = base_raw_ratings.copy()
        decreased_ratings[dim] = max(original_rating - change_delta, 1)
        decreased_idiosyncratic_score, _, _ = calculate_idiosyncratic_readiness(decreased_ratings, industry)
        decreased_org_ai_r = calculate_pe_org_ai_r(decreased_idiosyncratic_score, base_systematic_opportunity, alpha, beta, synergy_score)
        impact_decrease = decreased_org_ai_r - base_org_ai_r

        sensitivity_results.append({
            'Dimension': dim,
            'Change': f'{dim} (-{change_delta} Raw Rating)',
            'Impact on PE Org-AI-R': impact_decrease
        })
    
    return pd.DataFrame(sensitivity_results)


def calculate_exit_ai_r(
    visible_score: int,
    documented_score: int,
    sustainable_score: int,
    weights: dict = EXIT_READINESS_WEIGHTS
) -> float:
    """
    Calculates the Exit-AI-R Score.
    Args:
        visible_score (int): Score for how apparent AI capabilities are (0-100).
        documented_score (int): Score for how well AI capabilities are documented (0-100).
        sustainable_score (int): Score for how sustainable AI capabilities are (0-100).
        weights (dict): Weights for Visible, Documented, Sustainable components.
    Returns:
        float: The calculated Exit-AI-R Score.
    """
    exit_ai_r = (
        weights['Visible'] * visible_score +
        weights['Documented'] * documented_score +
        weights['Sustainable'] * sustainable_score
    )
    return exit_ai_r

# --- Plotting Functions ---

def plot_idiosyncratic_readiness(dimension_scores_df: pd.DataFrame) -> plt.Figure:
    """Plots Idiosyncratic Readiness dimension scores."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Scaled Score (0-100)', y='Dimension', data=dimension_scores_df.sort_values('Scaled Score (0-100)', ascending=False), ax=ax, palette='viridis')
    ax.set_title('InnovateCo Idiosyncratic Readiness Dimension Scores', fontsize=16)
    ax.set_xlabel('Scaled Score (0-100)', fontsize=12)
    ax.set_ylabel('AI Readiness Dimension', fontsize=12)
    plt.tight_layout()
    return fig

def plot_comparative_benchmark(gap_analysis_df: pd.DataFrame) -> plt.Figure:
    """Plots comparative bar chart of company vs. industry benchmarks."""
    df_melted = gap_analysis_df[['Dimension', 'InnovateCo Score', 'Benchmark Score']].melt(
        id_vars='Dimension', var_name='Type', value_name='Score'
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x='Score', y='Dimension', hue='Type', data=df_melted.sort_values('Dimension'), ax=ax, palette='Paired')
    ax.set_title('InnovateCo vs. Industry Benchmarks', fontsize=16)
    ax.set_xlabel('Score (0-100)', fontsize=12)
    ax.set_ylabel('AI Readiness Dimension', fontsize=12)
    ax.legend(title='Score Type')
    plt.tight_layout()
    return fig

def plot_gap_analysis(gap_analysis_df: pd.DataFrame) -> plt.Figure:
    """Plots AI Readiness Gaps."""
    fig, ax = plt.subplots(figsize=(12, 7))
    # Sort by gap for better visualization of priorities
    sorted_df = gap_analysis_df.sort_values('Gap', ascending=False)
    
    # Define a color palette for priorities
    palette = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    sns.barplot(x='Gap', y='Dimension', hue='Priority', data=sorted_df, ax=ax, palette=palette, dodge=False)
    
    ax.set_title('AI Readiness Gaps: Benchmark - InnovateCo Score', fontsize=16)
    ax.set_xlabel('Gap (Points)', fontsize=12)
    ax.set_ylabel('AI Readiness Dimension', fontsize=12)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.legend(title='Priority')
    plt.tight_layout()
    return fig

def plot_scenario_analysis(scenario_results_df: pd.DataFrame) -> plt.Figure:
    """Plots PE Org-AI-R Score under different scenarios."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Scenario', y='PE Org-AI-R Score', data=scenario_results_df, ax=ax, palette='coolwarm')
    ax.set_title('PE Org-AI-R Score Under Different Scenarios', fontsize=16)
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('PE Org-AI-R Score (0-100)', fontsize=12)
    ax.set_ylim(0, 100) # Ensure y-axis always goes to 100
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_sensitivity_analysis(sensitivity_df: pd.DataFrame, change_delta: int) -> plt.Figure:
    """Plots Sensitivity of PE Org-AI-R to Dimension Changes (diverging bar chart)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by absolute impact for clearer visualization of most sensitive dimensions
    sorted_df = sensitivity_df.reindex(sensitivity_df['Impact on PE Org-AI-R'].abs().sort_values(ascending=False).index)
    
    colors = ['firebrick' if x < 0 else 'mediumseagreen' for x in sorted_df['Impact on PE Org-AI-R']]
    
    ax.barh(sorted_df['Change'], sorted_df['Impact on PE Org-AI-R'], color=colors)
    ax.set_title(f'Sensitivity of PE Org-AI-R to +/- {change_delta} Raw Rating Change', fontsize=16)
    ax.set_xlabel('Impact on PE Org-AI-R Score', fontsize=12)
    ax.set_ylabel('Dimension and Change', fontsize=12)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    return fig

def plot_exit_ai_r_components(visible_score, documented_score, sustainable_score, weights) -> plt.Figure:
    """Plots Exit-AI-R Component Scores with weights."""
    scores_data = {
        'Component': ['Visible', 'Documented', 'Sustainable'],
        'Score': [visible_score, documented_score, sustainable_score],
        'Weight': [weights['Visible'], weights['Documented'], weights['Sustainable']]
    }
    df = pd.DataFrame(scores_data)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Component', y='Score', data=df, ax=ax, palette='pastel')

    # Add weight annotations
    for index, row in df.iterrows():
        ax.text(index, row['Score'] + 2, f'Weight: {row["Weight"]:.2f}', color='black', ha='center', va='bottom', fontsize=10)

    ax.set_title('Exit-AI-R Component Scores and Weights', fontsize=16)
    ax.set_xlabel('Exit-Readiness Component', fontsize=12)
    ax.set_ylabel('Score (0-100)', fontsize=12)
    ax.set_ylim(0, 110) # Adjust y-lim to make space for annotations
    plt.tight_layout()
    return fig
"""

page_1_welcome_contents = """
import streamlit as st
import pandas as pd
from utils import SYSTEMATIC_OPPORTUNITY_SCORES, DIMENSIONS

def main():
    st.subheader("Mission Briefing: InnovateCo Acquisition Target")

    st.markdown(
        \"\"\"
        Welcome, Alex. As a Quantitative Analyst at VentureBridge Capital, your expertise is crucial for our latest potential acquisition: **InnovateCo**, a leading innovator in the Manufacturing sector.
        
        Our goal is to thoroughly evaluate InnovateCo's AI readiness and potential for value creation through AI integration. This isn't just about identifying if they use AI, but understanding their deep-seated capabilities, market opportunities, and ultimately, how attractive they will be to future buyers from an AI perspective.

        Your task is to navigate this simulator, inputting findings from our due diligence, applying VentureBridge Capital's proprietary frameworks, and generating actionable insights for our Portfolio Managers.

        Let's start by defining the target company and its industry.
        \"\"\"
    )

    # Initialize session state for company details if not present
    if "company_name" not in st.session_state:
        st.session_state["company_name"] = "InnovateCo"
    if "company_industry" not in st.session_state:
        st.session_state["company_industry"] = 'Manufacturing'
    
    # Initial setup for raw ratings, will be updated on the next page
    if "raw_dimension_ratings" not in st.session_state:
        st.session_state["raw_dimension_ratings"] = {
            'Data Infrastructure': 2,
            'AI Governance': 1,
            'Technology Stack': 3,
            'Talent': 2,
            'Leadership': 3,
            'Use Case Portfolio': 1,
            'Culture': 2
        }

    st.markdown("---")
    st.markdown("### Identify the Target Company")

    # Widget for company name
    st.session_state["company_name"] = st.text_input(
        label="Target Company Name",
        value=st.session_state["company_name"],
        key="company_name_input" # Use a distinct key to avoid conflicts with session_state key
    )
    st.session_state["company_name"] = st.session_state["company_name_input"]


    # Widget for company industry sector
    industry_options = list(SYSTEMATIC_OPPORTUNITY_SCORES.keys())
    # Ensure the default value is in the options list for index lookup
    default_industry_index = industry_options.index(st.session_state["company_industry"]) if st.session_state["company_industry"] in industry_options else 0

    st.session_state["company_industry"] = st.selectbox(
        label="Company Industry Sector",
        options=industry_options,
        index=default_industry_index,
        key="company_industry_select" # Use a distinct key
    )
    st.session_state["company_industry"] = st.session_state["company_industry_select"]


    st.info(
        f"""
        **Analyst Note:** Alex, selecting the correct industry sector for **{st.session_state['company_name']}**
        is vital. It automatically loads sector-specific AI opportunity scores and adjusts
        the weighting of AI readiness dimensions, reflecting industry nuances and
        VentureBridge Capital's strategic focus. This ensures our assessment is
        contextually relevant.
        """
    )
"""

page_2_static_framework_contents = """
import streamlit as st
import pandas as pd
from utils import SYSTEMATIC_OPPORTUNITY_SCORES, SECTOR_DIMENSION_WEIGHTS, INDUSTRY_BENCHMARKS, DIMENSIONS

def main():
    st.subheader("Reviewing VentureBridge Capital's AI Readiness Framework")

    st.markdown(
        f\"\"\"
        Alex, before we dive into InnovateCo's specific assessment, it's crucial to understand the foundational data and frameworks VentureBridge Capital employs.
        This step allows you to review the **static parameters** that will anchor our calculations, ensuring consistency and rigor in our due diligence process.

        As you analyze **{st.session_state.get('company_name', 'InnovateCo')}** in the **{st.session_state.get('company_industry', 'Manufacturing')}** sector, these parameters will be automatically applied to contextualize its AI readiness.
        \"\"\"
    )

    st.markdown("---")
    
    st.markdown("### Industry-Specific Systematic Opportunity Scores")
    st.markdown(
        \"\"\"
        The `SystematicOpportunity` score represents the inherent potential for AI in a given industry.
        It reflects external market factors, technological maturity in the sector, and competitive landscape.
        \"\"\"
    )
    st.dataframe(pd.DataFrame(SYSTEMATIC_OPPORTUNITY_SCORES.items(), columns=['Industry', 'Systematic Opportunity Score (0-100)']))
    st.info(
        \"\"\"
        **Analyst Note:** For **InnovateCo** in the **Manufacturing** sector, the `SystematicOpportunity` is currently set at **80**. This acts as a baseline for the market potential for AI in its operating environment.
        \"\"\"
    )

    st.markdown("---")

    st.markdown("### AI Readiness Dimension Weights by Sector")
    st.markdown(
        \"\"\"
        Different industries prioritize AI dimensions differently. These weights reflect VentureBridge Capital's view on what matters most for AI success in various sectors.
        For example, 'AI Governance' might carry more weight in Healthcare due to regulatory demands, while 'Technology Stack' is paramount in Technology firms.
        \"\"\"
    )
    with st.expander("View All Sector-Specific Dimension Weights"):
        # Convert dictionary of dictionaries to a more readable DataFrame
        weights_df = pd.DataFrame(SECTOR_DIMENSION_WEIGHTS).T
        weights_df.index.name = 'Industry Sector'
        st.dataframe(weights_df.style.format("{:.2f}"))

    current_industry = st.session_state.get('company_industry', 'Manufacturing')
    st.markdown(f"**InnovateCo's ({current_industry}) Applied Dimension Weights:**")
    current_weights_df = pd.DataFrame([SECTOR_DIMENSION_WEIGHTS.get(current_industry, {})]).T.reset_index()
    current_weights_df.columns = ['Dimension', 'Weight']
    st.dataframe(current_weights_df.style.format("{:.2f}"))
    st.info(
        \"\"\"
        **Analyst Note:** These are the specific weights applied to InnovateCo's internal capabilities given its manufacturing context.
        They ensure that our assessment emphasizes the dimensions most critical for AI value creation in that sector.
        \"\"\"
    )

    st.markdown("---")

    st.markdown("### Industry AI Readiness Benchmarks")
    st.markdown(
        \"\"\"
        These benchmarks represent the average or target AI readiness scores for companies within a typical industry.
        We'll use these to perform a gap analysis later, identifying where InnovateCo stands relative to its peers.
        \"\"\"
    )
    benchmarks_df = pd.DataFrame(INDUSTRY_BENCHMARKS.items(), columns=['Dimension', 'Benchmark Score (0-100)'])
    st.dataframe(benchmarks_df)
    st.info(
        \"\"\"
        **Analyst Note:** Comparing InnovateCo against these benchmarks will highlight areas of strength and, more importantly,
        identify critical areas for improvement where strategic investment can yield significant returns.
        \"\"\"
    )
"""

page_3_raw_idiosyncratic_ratings_contents = """
import streamlit as st
from utils import DIMENSIONS

def main():
    st.subheader("Collecting Raw Idiosyncratic Readiness Ratings for InnovateCo")

    st.markdown(
        f\"\"\"
        Alex, this is where your deep dive into **{st.session_state.get('company_name', 'InnovateCo')}** comes into play.
        Based on your due diligence – interviews with management, technical assessments, and operational reviews – you need to quantify
        InnovateCo's current state across seven key AI readiness dimensions.

        Each dimension is rated on a scale of 1 to 5, where 1 indicates a nascent or non-existent capability and 5 signifies
        a highly mature, optimized capability. Your accuracy here directly impacts the precision of our overall assessment.
        \"\"\"
    )

    st.markdown("---")
    st.markdown("### InnovateCo's Raw Dimension Ratings")

    # Initialize raw_dimension_ratings in session state if not present or ensure it's a dict
    if "raw_dimension_ratings" not in st.session_state or not isinstance(st.session_state["raw_dimension_ratings"], dict):
        st.session_state["raw_dimension_ratings"] = {
            'Data Infrastructure': 2,
            'AI Governance': 1,
            'Technology Stack': 3,
            'Talent': 2,
            'Leadership': 3,
            'Use Case Portfolio': 1,
            'Culture': 2
        }

    st.markdown(
        \"\"\"
        *Move the sliders to reflect InnovateCo's current state for each dimension.*
        \"\"\"
    )

    col1, col2 = st.columns(2)
    # Create 7 sliders for the raw dimension ratings
    for i, dimension in enumerate(DIMENSIONS):
        with (col1 if i < len(DIMENSIONS)/2 else col2):
            st.session_state["raw_dimension_ratings"][dimension] = st.slider(
                label=f"{dimension} Rating (1-5)",
                min_value=1,
                max_value=5,
                step=1,
                value=st.session_state["raw_dimension_ratings"].get(dimension, 1), # Use get with default to prevent KeyError
                key=f"raw_rating_{dimension.replace(' ', '_').lower()}"
            )

    st.info(
        \"\"\"
        **Analyst Note:** Each adjustment here represents a critical finding from your due diligence.
        For example, a low score in 'AI Governance' might signal regulatory risk or a lack of ethical guidelines,
        while a high 'Technology Stack' score suggests a solid foundation for AI deployment.
        These raw inputs are the bedrock of our quantitative model.
        \"\"\"
    )
"""

page_4_normalized_idiosyncratic_readiness_contents = """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import calculate_idiosyncratic_readiness, plot_idiosyncratic_readiness, DIMENSIONS

def main():
    st.subheader("InnovateCo's Idiosyncratic Readiness: Unpacking Internal AI Capabilities")

    st.markdown(
        f\"\"\"
        Alex, having captured the raw ratings from your due diligence, we now translate these into a **normalized Idiosyncratic Readiness score**.
        This score represents **{st.session_state.get('company_name', 'InnovateCo')}'s** internal AI capabilities, considering the unique
        weighting of each dimension based on its **{st.session_state.get('company_industry', 'Manufacturing')}** industry context.

        This step reveals where InnovateCo truly stands in its internal AI journey, highlighting areas of strength and immediate focus.
        \"\"\"
    )

    # Ensure raw_dimension_ratings is initialized
    if "raw_dimension_ratings" not in st.session_state:
        st.session_state["raw_dimension_ratings"] = {dim: 1 for dim in DIMENSIONS} # Default to 1 if not set

    # Calculate Idiosyncratic Readiness
    idiosyncratic_readiness_score, dimension_scores_df, applied_weights = calculate_idiosyncratic_readiness(
        st.session_state["raw_dimension_ratings"],
        st.session_state.get("company_industry", "Manufacturing")
    )

    # Store results in session state
    st.session_state["idiosyncratic_readiness_score"] = idiosyncratic_readiness_score
    st.session_state["innovateco_dimension_scores_df"] = dimension_scores_df
    st.session_state["applied_weights"] = applied_weights

    st.markdown("---")

    st.markdown("### Overall Idiosyncratic Readiness Score")
    st.markdown(
        r\"\"\"
        The `IdiosyncraticReadiness` score aggregates the weighted and scaled raw ratings across all dimensions.
        The formula is:
        $$ IdiosyncraticReadiness = \\frac{\\sum_{i=1}^{7} w_i \\cdot \\text{ScaledRating}_i}{100} \\times 100 $$
        where $\\text{ScaledRating}_i$ is the raw rating (1-5) scaled to a 0-100 range, and $w_i$ is the industry-specific weight for dimension $i$.
        \"\"\"
    )
    
    st.metric(
        label=f"InnovateCo's Overall Idiosyncratic Readiness Score",
        value=f"{st.session_state['idiosyncratic_readiness_score']:.2f} / 100"
    )

    st.info(
        f"""
        **Analyst Note:** A score of **{st.session_state['idiosyncratic_readiness_score']:.2f}** for **{st.session_state.get('company_name', 'InnovateCo')}**
        quantifies its current internal AI maturity. This single figure encapsulates its strengths and weaknesses
        across all dimensions, weighted by their strategic importance in the **{st.session_state.get('company_industry', 'Manufacturing')}** sector.
        """
    )

    st.markdown("---")

    st.markdown("### Detailed Dimension Scores")
    st.markdown(
        \"\"\"
        Below are the individual scores for each AI readiness dimension, providing a granular view of InnovateCo's capabilities.
        \"\"\"
    )
    st.dataframe(st.session_state["innovateco_dimension_scores_df"].style.format({
        'Raw Rating (1-5)': '{:.0f}',
        'Scaled Score (0-100)': '{:.2f}',
        'Weight': '{:.2f}',
        'Weighted Score': '{:.2f}'
    }))

    st.markdown("---")

    st.markdown("### Visualizing InnovateCo's Internal Strengths and Weaknesses")
    st.markdown(
        \"\"\"
        This bar chart visually represents InnovateCo's performance across each AI readiness dimension.
        It quickly highlights areas where the company has a solid foundation versus areas requiring significant attention and investment.
        \"\"\"
    )
    # Plotting the dimension scores
    fig = plot_idiosyncratic_readiness(st.session_state["innovateco_dimension_scores_df"])
    st.pyplot(fig)
    plt.close(fig) # Close the figure to prevent display issues

    st.info(
        """
        **Analyst Note:** From this chart, you can clearly see InnovateCo's strongest and weakest AI dimensions.
        For example, a low 'Use Case Portfolio' score suggests they haven't effectively identified or implemented AI applications,
        while a decent 'Technology Stack' indicates foundational infrastructure. These insights will guide our value creation strategy.
        """
    )
"""

page_5_pe_org_ai_r_score_contents = """
import streamlit as st
from utils import calculate_pe_org_ai_r, SYSTEMATIC_OPPORTUNITY_SCORES, DIMENSIONS

def main():
    st.subheader("Calculating the Overall PE Org-AI-R Score")

    st.markdown(
        f\"\"\"
        Alex, this is the core of our investment assessment: calculating the **PE Org-AI-R Score**.
        This score provides a holistic view of **{st.session_state.get('company_name', 'InnovateCo')}'s** AI readiness by
        integrating its internal capabilities (`IdiosyncraticReadiness`), the external market opportunity
        (`SystematicOpportunity`), and the synergy between the two.

        Your adjustments to the weighting parameters ($\alpha$ and $\beta$) allow VentureBridge Capital to
        fine-tune the strategic focus, reflecting our specific investment thesis and risk appetite.
        \"\"\"
    )

    # Initialize session state for parameters if not present
    if "alpha_param" not in st.session_state:
        st.session_state["alpha_param"] = 0.6
    if "beta_param" not in st.session_state:
        st.session_state["beta_param"] = 0.15
    if "synergy_score" not in st.session_state:
        st.session_state["synergy_score"] = 50

    # Retrieve necessary scores from session state (should be calculated by previous page)
    idiosyncratic_readiness_score = st.session_state.get("idiosyncratic_readiness_score", 0.0)
    company_industry = st.session_state.get("company_industry", "Manufacturing")
    systematic_opportunity_score = SYSTEMATIC_OPPORTUNITY_SCORES.get(company_industry, 0)

    st.markdown("---")

    st.markdown("### Adjust PE Org-AI-R Score Parameters")

    # Alpha parameter slider
    st.session_state["alpha_param"] = st.slider(
        label=r"Weight on Organizational Factors ($\alpha$)",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=st.session_state["alpha_param"],
        key="alpha_param_slider"
    )
    st.info(
        r"""
        **Analyst Note:** Adjusting $\alpha$ shifts the balance of the overall score.
        A higher $\alpha$ means we place more emphasis on **InnovateCo's** internal AI capabilities
        (`IdiosyncraticReadiness`), while a lower $\alpha$ prioritizes the broader market potential
        (`SystematicOpportunity`) of the **Manufacturing** sector. This reflects VentureBridge Capital's
        investment philosophy for this deal.
        """
    )

    # Beta parameter slider
    st.session_state["beta_param"] = st.slider(
        label=r"Synergy Coefficient ($\beta$)",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        value=st.session_state["beta_param"],
        key="beta_param_slider"
    )
    st.info(
        r"""
        **Analyst Note:** The $\beta$ coefficient determines the weight given to the synergy component.
        A higher $\beta$ suggests that we believe InnovateCo's internal AI readiness can significantly
        amplify its ability to capitalize on market opportunities. It's about how well its capabilities
        align with and leverage external potential.
        """
    )

    # Synergy Score slider
    st.session_state["synergy_score"] = st.slider(
        label="Synergy Score (0-100)",
        min_value=0,
        max_value=100,
        step=1,
        value=st.session_state["synergy_score"],
        key="synergy_score_slider"
    )
    st.info(
        """
        **Analyst Note:** This Synergy Score is your qualitative assessment, Alex, of how well
        InnovateCo's internal AI efforts actually align with and amplify its market opportunities.
        A high score here means excellent strategic fit and execution potential.
        """
    )

    st.markdown("---")

    st.markdown("### The PE Org-AI-R Score Formula")
    st.markdown(
        r\"\"\"
        The `PE Org-AI-R Score` is calculated using the following proprietary formula:
        $$PE \\text{ Org-AI-R} = \\alpha \\cdot IdiosyncraticReadiness + (1 - \\alpha) \\cdot SystematicOpportunity + \\beta \\cdot Synergy$$
        where `Synergy` in the formula is represented by `Synergy Score` $\\times$ (`IdiosyncraticReadiness`/100) $\\times$ (`SystematicOpportunity`/100)
        to reflect the multiplicative effect of actual capabilities and market potential.
        \"\"\"
    )

    # Calculate PE Org-AI-R Score
    pe_org_ai_r_score = calculate_pe_org_ai_r(
        idiosyncratic_readiness_score,
        systematic_opportunity_score,
        st.session_state["alpha_param"],
        st.session_state["beta_param"],
        st.session_state["synergy_score"]
    )
    st.session_state["pe_org_ai_r_score"] = pe_org_ai_r_score

    st.metric(
        label=f"InnovateCo's Overall PE Org-AI-R Score",
        value=f"{st.session_state['pe_org_ai_r_score']:.2f} / 100",
        delta_color="off"
    )

    st.info(
        f"""
        **Analyst Note:** A score of **{st.session_state['pe_org_ai_r_score']:.2f}** for **{st.session_state.get('company_name', 'InnovateCo')}**
        provides VentureBridge Capital with a quantitative anchor for its investment decision.
        It suggests a company with potential to leverage AI, but the specific breakdown
        in `IdiosyncraticReadiness` and the `Synergy` component will be critical
        for designing our value creation plan. This confirms preliminary screening or
        highlights areas for deeper due diligence.
        """
    )
"""

page_6_gap_analysis_contents = """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import perform_gap_analysis, plot_comparative_benchmark, plot_gap_analysis, INDUSTRY_BENCHMARKS, DIMENSIONS

def main():
    st.subheader("InnovateCo's AI Readiness Gap Analysis")

    st.markdown(
        f\"\"\"
        Alex, with **{st.session_state.get('company_name', 'InnovateCo')}'s** Idiosyncratic Readiness score established,
        it's time to put it into perspective. This step involves performing a **Gap Analysis** by comparing
        InnovateCo's current AI capabilities against established industry benchmarks.

        This comparison is critical for VentureBridge Capital to identify specific areas where InnovateCo
        lags its peers, signaling clear opportunities for targeted investment and operational improvements
        post-acquisition. It directly informs our value creation strategy.
        \"\"\"
    )

    # Ensure dimension scores are available from previous page
    if "innovateco_dimension_scores_df" not in st.session_state:
        st.error("Please navigate through 'Normalized Idiosyncratic Readiness' first to calculate dimension scores.")
        return

    # Perform Gap Analysis
    gap_analysis_df = perform_gap_analysis(
        st.session_state["innovateco_dimension_scores_df"],
        INDUSTRY_BENCHMARKS
    )
    st.session_state["gap_analysis_df"] = gap_analysis_df

    st.markdown("---")

    st.markdown("### Comparative Analysis: InnovateCo vs. Industry Benchmarks")
    st.markdown(
        \"\"\"
        This visualization provides a direct, side-by-side comparison of InnovateCo's scaled scores for each
        AI readiness dimension against the average industry benchmark. This helps to quickly identify
        where InnovateCo is competitive and where it is falling behind.
        \"\"\"
    )

    # Plot comparative benchmark
    fig_comp = plot_comparative_benchmark(st.session_state["gap_analysis_df"])
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    st.info(
        """
        **Analyst Note:** A quick glance here shows us where InnovateCo is on par or even
        exceeding industry standards, but more importantly, where significant gaps exist.
        These gaps represent opportunities for VentureBridge Capital to inject capital and expertise,
        driving post-acquisition growth and competitive advantage.
        """
    )

    st.markdown("---")

    st.markdown("### Quantifying the AI Readiness Gaps")
    st.markdown(
        r\"\"\"
        The gap for each dimension ($Gap_k$) is calculated as the difference between the industry benchmark score
        ($D_k^{benchmark}$) and InnovateCo's current scaled score ($D_k^{current}$):
        $$Gap_k = D_k^{benchmark} - D_k^{current}$$
        A positive gap indicates an area where InnovateCo needs to improve to reach industry standards.
        \"\"\"
    )
    st.dataframe(st.session_state["gap_analysis_df"].style.format({
        'InnovateCo Score': '{:.2f}',
        'Benchmark Score': '{:.2f}',
        'Gap': '{:.2f}'
    }))

    st.markdown("---")

    st.markdown("### Visualizing the AI Readiness Gaps by Priority")
    st.markdown(
        \"\"\"
        This bar chart visually represents the size of each gap, color-coded by priority (High, Medium, Low).
        This helps Alex and the Portfolio Managers quickly identify the most critical areas for intervention.
        \"\"\"
    )

    # Plot gap analysis
    fig_gap = plot_gap_analysis(st.session_state["gap_analysis_df"])
    st.pyplot(fig_gap)
    plt.close(fig_gap)

    st.info(
        """
        **Analyst Note:** The dimensions with the largest positive gaps (especially those marked 'High' priority)
        are prime candidates for strategic investment and operational focus immediately following an acquisition.
        For example, a 'High' gap in 'AI Governance' could signal a critical need for new policies and leadership,
        while a gap in 'Data Infrastructure' might require significant capital expenditure.
        This prioritization is key for effective resource allocation.
        """
    )
"""

page_7_scenario_analysis_contents = """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import run_scenario_analysis, plot_scenario_analysis, DIMENSIONS

def main():
    st.subheader("Scenario Analysis: Modeling InnovateCo's Future AI Readiness")

    st.markdown(
        f\"\"\"
        Alex, private equity is all about value creation. This step, **Scenario Analysis**,
        allows us to model how **{st.session_state.get('company_name', 'InnovateCo')}'s** `PE Org-AI-R Score`
        could evolve under different investment and operational strategies.

        By simulating various "what-if" scenarios, we can quantify the potential impact of our initiatives
        and build a compelling case for the Portfolio Managers regarding the ROI of AI investments.
        This moves us from assessment to strategic foresight.
        \"\"\"
    )

    # Ensure necessary session state variables are present
    if "raw_dimension_ratings" not in st.session_state or \\
       "company_industry" not in st.session_state or \\
       "alpha_param" not in st.session_state or \\
       "beta_param" not in st.session_state or \\
       "synergy_score" not in st.session_state:
        st.error("Please ensure you have completed the previous steps to define company, ratings, and PE Org-AI-R parameters.")
        return

    st.markdown("---")

    st.markdown("### Defining Our Strategic Scenarios")
    st.markdown(
        \"\"\"
        We've pre-defined a few typical scenarios based on common investment strategies.
        Each scenario postulates specific improvements in InnovateCo's raw AI readiness
        dimension ratings and potential changes in its synergy score.
        \"\"\"
    )

    # Scenario definitions (can be externalized to utils.py or loaded from a file)
    scenario_definitions = {
        'Optimistic Case (Aggressive AI Investment)': {
            'rating_deltas': {
                'Data Infrastructure': 2, 'AI Governance': 2, 'Technology Stack': 1,
                'Talent': 1, 'Leadership': 1, 'Use Case Portfolio': 2, 'Culture': 1
            },
            'synergy_delta': 20
        },
        'Moderate Case (Targeted Improvements)': {
            'rating_deltas': {
                'Data Infrastructure': 1, 'AI Governance': 1, 'Technology Stack': 1,
                'Use Case Portfolio': 1
            },
            'synergy_delta': 10
        },
        'Pessimistic Case (Minimal Investment)': {
            'rating_deltas': {}, # No planned improvements
            'synergy_delta': -10 # Potential decline in synergy
        }
    }

    st.expander("View Scenario Definitions (Raw Rating & Synergy Deltas)").json(scenario_definitions)

    # Run scenario analysis
    scenario_results_df = run_scenario_analysis(
        st.session_state["raw_dimension_ratings"],
        st.session_state["company_industry"],
        st.session_state["alpha_param"],
        st.session_state["beta_param"],
        st.session_state["synergy_score"],
        scenario_definitions
    )
    st.session_state["scenario_results_df"] = scenario_results_df

    st.markdown("---")

    st.markdown("### Predicted PE Org-AI-R Scores Under Scenarios")
    st.markdown(
        \"\"\"
        This table shows the calculated `PE Org-AI-R Score` for **InnovateCo** under each of the defined scenarios.
        \"\"\"
    )
    st.dataframe(st.session_state["scenario_results_df"].style.format({'PE Org-AI-R Score': '{:.2f}'}))

    st.markdown("---")

    st.markdown("### Visualizing Future AI Readiness Trajectories")
    st.markdown(
        \"\"\"
        This bar chart visually compares **InnovateCo's** `PE Org-AI-R Score` across the base case and various future scenarios.
        It clearly illustrates the potential uplift in AI readiness that targeted investments can achieve.
        \"\"\"
    )
    # Plot scenario analysis
    fig = plot_scenario_analysis(st.session_state["scenario_results_df"])
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        """
        **Analyst Note:** This visualization provides a powerful narrative for Portfolio Managers.
        It quantifies the potential upside of an acquisition, showing how proactive AI investment
        can significantly increase InnovateCo's strategic value. The difference between the
        Pessimistic and Optimistic cases highlights the tangible benefits of VentureBridge Capital's
        active management and AI integration strategy.
        """
    )
"""

page_8_sensitivity_analysis_contents = """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import perform_sensitivity_analysis, plot_sensitivity_analysis, DIMENSIONS

def main():
    st.subheader("Sensitivity Analysis: Identifying Key AI Investment Levers")

    st.markdown(
        f\"\"\"
        Alex, while scenario analysis shows us potential outcomes, **Sensitivity Analysis** drills down
        to identify which specific AI readiness dimensions have the largest impact on **{st.session_state.get('company_name', 'InnovateCo')}'s**
        overall `PE Org-AI-R Score`.

        This is crucial for prioritizing our post-acquisition AI initiatives. We want to know where to
        allocate resources for maximum impact, making our value creation plan as efficient as possible.
        Which 'swing factors' should VentureBridge Capital focus on?
        \"\"\"
    )

    # Ensure necessary session state variables are present
    if "raw_dimension_ratings" not in st.session_state or \\
       "company_industry" not in st.session_state or \\
       "alpha_param" not in st.session_state or \\
       "beta_param" not in st.session_state or \\
       "synergy_score" not in st.session_state:
        st.error("Please ensure you have completed the previous steps to define company, ratings, and PE Org-AI-R parameters.")
        return
    
    # Initialize sensitivity_change_delta in session state
    if "sensitivity_change_delta" not in st.session_state:
        st.session_state["sensitivity_change_delta"] = 1

    st.markdown("---")

    st.markdown("### Configure Sensitivity Test")

    # Slider for raw rating change delta
    st.session_state["sensitivity_change_delta"] = st.slider(
        label=r"Raw Rating Change for Sensitivity Analysis ($\pm$ points)",
        min_value=1,
        max_value=2,
        step=1,
        value=st.session_state["sensitivity_change_delta"],
        key="sensitivity_change_delta_slider"
    )
    st.info(
        r"""
        **Analyst Note:** This slider defines how much we "stress-test" each dimension.
        A $\pm 1$ point change in raw rating for a dimension simulates a moderate improvement or decline,
        while $\pm 2$ points represent a more significant shift. This helps us gauge the robustness
        of our `PE Org-AI-R Score` to changes in individual capabilities.
        """
    )

    # Perform sensitivity analysis
    sensitivity_df = perform_sensitivity_analysis(
        st.session_state["raw_dimension_ratings"],
        st.session_state["company_industry"],
        st.session_state["alpha_param"],
        st.session_state["beta_param"],
        st.session_state["synergy_score"],
        st.session_state["sensitivity_change_delta"]
    )
    st.session_state["sensitivity_df"] = sensitivity_df

    st.markdown("---")

    st.markdown("### Impact of Dimension Changes on PE Org-AI-R Score")
    st.markdown(
        \"\"\"
        This table shows the calculated change in InnovateCo's `PE Org-AI-R Score` if each
        individual AI readiness dimension's raw rating were to increase or decrease
        by the specified `$\pm$ points`.
        \"\"\"
    )
    # Sort for better readability, showing most impactful changes first
    sorted_sensitivity_df = st.session_state["sensitivity_df"].sort_values(by='Impact on PE Org-AI-R', ascending=False)
    st.dataframe(sorted_sensitivity_df.style.format({'Impact on PE Org-AI-R': '{:.2f}'}))

    st.markdown("---")

    st.markdown("### Visualizing Sensitivity: Which Dimensions Move the Needle Most?")
    st.markdown(
        \"\"\"
        This diverging bar chart (a simplified tornado plot) clearly identifies the "swing factors".
        The longer the bar, the more sensitive the `PE Org-AI-R Score` is to changes in that dimension.
        Positive bars represent an increase in score, negative bars a decrease.
        \"\"\"
    )
    # Plot sensitivity analysis
    fig = plot_sensitivity_analysis(st.session_state["sensitivity_df"], st.session_state["sensitivity_change_delta"])
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        """
        **Analyst Note:** The dimensions with the largest absolute impact are our highest-priority targets.
        If improving 'Data Infrastructure' by one point yields a significantly higher increase in `PE Org-AI-R`
        than improving 'Culture' by the same amount, it suggests where VentureBridge Capital should focus
        its initial post-acquisition investments to maximize value creation. This is critical for
        efficient resource allocation and setting strategic priorities.
        """
    )
"""

page_9_exit_readiness_contents = """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import calculate_exit_ai_r, plot_exit_ai_r_components, EXIT_READINESS_WEIGHTS

def main():
    st.subheader("InnovateCo's Exit-AI-R Score: Assessing Attractiveness to Future Buyers")

    st.markdown(
        f\"\"\"
        Alex, the final piece of our assessment for **{st.session_state.get('company_name', 'InnovateCo')}** is its **Exit-AI-R Score**.
        Even as we plan our value creation, VentureBridge Capital always considers the exit.
        This score measures how attractive InnovateCo's AI capabilities would be to a future buyer,
        focusing on factors that make AI value apparent, verifiable, and sustainable to an external party.

        Your role here is to translate our internal understanding into a score that reflects an acquirer's perspective,
        helping us craft a compelling "AI exit narrative" for InnovateCo.
        \"\"\"
    )

    # Initialize session state for exit readiness components if not present
    if "exit_visible_score" not in st.session_state:
        st.session_state["exit_visible_score"] = 30
    if "exit_documented_score" not in st.session_state:
        st.session_state["exit_documented_score"] = 20
    if "exit_sustainable_score" not in st.session_state:
        st.session_state["exit_sustainable_score"] = 25
    
    st.markdown("---")

    st.markdown("### Assess Exit-Readiness Components")
    st.markdown(
        \"\"\"
        Use the sliders to rate InnovateCo's AI capabilities from a prospective buyer's perspective across these three critical dimensions:
        \"\"\"
    )

    # Sliders for Exit-AI-R components
    st.session_state["exit_visible_score"] = st.slider(
        label="Visible Score (0-100) - How apparent are AI capabilities?",
        min_value=0,
        max_value=100,
        step=1,
        value=st.session_state["exit_visible_score"],
        key="exit_visible_score_slider"
    )
    st.info(
        """
        **Analyst Note:** A high Visible Score means InnovateCo's AI value is easy for external parties to see
        (e.g., clear product features, public case studies).
        """
    )

    st.session_state["exit_documented_score"] = st.slider(
        label="Documented Score (0-100) - How well are AI processes and impact recorded?",
        min_value=0,
        max_value=100,
        step=1,
        value=st.session_state["exit_documented_score"],
        key="exit_documented_score_slider"
    )
    st.info(
        """
        **Analyst Note:** A strong Documented Score indicates robust internal records of AI projects,
        performance metrics, and governance, which are critical for due diligence.
        """
    )

    st.session_state["exit_sustainable_score"] = st.slider(
        label="Sustainable Score (0-100) - How durable and transferable are AI capabilities?",
        min_value=0,
        max_value=100,
        step=1,
        value=st.session_state["exit_sustainable_score"],
        key="exit_sustainable_score_slider"
    )
    st.info(
        """
        **Analyst Note:** A high Sustainable Score implies InnovateCo's AI capabilities
        are embedded in robust systems and talent, not reliant on a few individuals,
        making them more valuable post-acquisition.
        """
    )

    st.markdown("---")

    st.markdown("### The Exit-AI-R Score Formula")
    st.markdown(
        r\"\"\"
        The `Exit-AI-R Score` is a weighted average of the three components:
        $$Exit\\text{-AI-R} = w_1 \\cdot Visible + w_2 \\cdot Documented + w_3 \\cdot Sustainable$$
        where $w_1, w_2, w_3$ are the pre-defined weights for each component.
        \"\"\"
    )
    st.markdown(f"Current weights: Visible ($w_1$): {EXIT_READINESS_WEIGHTS['Visible']:.2f}, Documented ($w_2$): {EXIT_READINESS_WEIGHTS['Documented']:.2f}, Sustainable ($w_3$): {EXIT_READINESS_WEIGHTS['Sustainable']:.2f}")


    # Calculate Exit-AI-R Score
    exit_ai_r_score = calculate_exit_ai_r(
        st.session_state["exit_visible_score"],
        st.session_state["exit_documented_score"],
        st.session_state["exit_sustainable_score"],
        EXIT_READINESS_WEIGHTS
    )
    st.session_state["exit_ai_r_score"] = exit_ai_r_score

    st.metric(
        label=f"InnovateCo's Overall Exit-AI-R Score",
        value=f"{st.session_state['exit_ai_r_score']:.2f} / 100",
        delta_color="off"
    )

    st.info(
        f"""
        **Analyst Note:** An `Exit-AI-R Score` of **{st.session_state['exit_ai_r_score']:.2f}** for **{st.session_state.get('company_name', 'InnovateCo')}**
        quantifies its appeal to future acquirers from an AI perspective. This score informs our
        strategy to enhance InnovateCo's "sellability" by focusing on making its AI value
        more apparent, better documented, and truly sustainable.
        """
    )

    st.markdown("---")

    st.markdown("### Visualizing Exit-AI-R Components")
    st.markdown(
        \"\"\"
        This bar chart breaks down InnovateCo's `Exit-AI-R Score` by its contributing components,
        including their respective weights. This helps pinpoint which aspects of "AI sellability"
        are strong and which require improvement to maximize future exit value.
        \"\"\"
    )
    # Plot Exit-AI-R components
    fig = plot_exit_ai_r_components(
        st.session_state["exit_visible_score"],
        st.session_state["exit_documented_score"],
        st.session_state["exit_sustainable_score"],
        EXIT_READINESS_WEIGHTS
    )
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        """
        **Analyst Note:** If, for instance, the 'Documented Score' is low, it suggests a need to
        invest in robust internal reporting and knowledge transfer. If 'Visible Score' is low,
        we might focus on showcasing AI-driven product features and success stories.
        This guides our efforts to build an attractive AI narrative for the next transaction.
        """
    )
"""

app_py_contents = """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="PE-AI Readiness Simulator", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("PE-AI Readiness Simulator: VentureBridge Capital")
st.divider()

st.markdown(\"\"\"
In this lab, you assume the role of **Alex, a Quantitative Analyst at VentureBridge Capital**. Your mission is to rigorously evaluate **'InnovateCo'**, a potential acquisition target in the Manufacturing sector, for its **AI readiness**. This application will guide you through VentureBridge Capital's proprietary, story-driven workflow to:

1.  **Assess InnovateCo's internal AI capabilities** (Idiosyncratic Readiness) across key dimensions.
2.  **Quantify its overall AI maturity** using the `PE Org-AI-R Score`, considering both internal strengths and external market opportunities.
3.  **Identify critical AI readiness gaps** against industry benchmarks, pinpointing areas for strategic investment.
4.  **Model future AI readiness scenarios** and perform sensitivity analysis to identify high-impact improvement areas.
5.  **Evaluate InnovateCo's attractiveness to future buyers** from an AI perspective, yielding an `Exit-AI-R Score`.

Each step mirrors a real-world task in private equity due diligence, helping you provide data-driven insights to inform multi-million dollar investment decisions.
\"\"\")

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
"""

# Create the necessary files
write_file_to_github("requirements.txt", requirements_txt_contents)
write_file_to_github("utils.py", utils_py_contents)
write_file_to_github("application_pages/page_1_welcome.py", page_1_welcome_contents)
write_file_to_github("application_pages/page_2_static_framework.py", page_2_static_framework_contents)
write_file_to_github("application_pages/page_3_raw_idiosyncratic_ratings.py", page_3_raw_idiosyncratic_ratings_contents)
write_file_to_github("application_pages/page_4_normalized_idiosyncratic_readiness.py", page_4_normalized_idiosyncratic_readiness_contents)
write_file_to_github("application_pages/page_5_pe_org_ai_r_score.py", page_5_pe_org_ai_r_score_contents)
write_file_to_github("application_pages/page_6_gap_analysis.py", page_6_gap_analysis_contents)
write_file_to_github("application_pages/page_7_scenario_analysis.py", page_7_scenario_analysis_contents)
write_file_to_github("application_pages/page_8_sensitivity_analysis.py", page_8_sensitivity_analysis_contents)
write_file_to_github("application_pages/page_9_exit_readiness.py", page_9_exit_readiness_contents)
write_file_to_github("app.py", app_py_contents)

# Dynamically import utils to access constants for assertions
# This ensures that constants like DIMENSIONS are available in the test file
# without recreating them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import SYSTEMATIC_OPPORTUNITY_SCORES, SECTOR_DIMENSION_WEIGHTS, INDUSTRY_BENCHMARKS, DIMENSIONS, calculate_idiosyncratic_readiness, calculate_pe_org_ai_r, calculate_exit_ai_r
sys.path.pop(0)

# --- Test Functions ---

def test_1_welcome_page_initial_state_and_inputs():
    at = AppTest.from_file("app.py").run()
    # Select the first page
    at.sidebar.selectbox[0].set_value("1. Welcome & Company Selection").run()

    # Assert initial state
    assert at.session_state["company_name"] == "InnovateCo"
    assert at.session_state["company_industry"] == "Manufacturing"
    assert "raw_dimension_ratings" in at.session_state
    
    # Assert widgets are present and have correct initial values
    assert at.text_input[0].value == "InnovateCo"
    assert at.selectbox[0].value == "Manufacturing"

    # Test changing inputs
    at.text_input[0].set_value("NewCo").run()
    assert at.session_state["company_name"] == "NewCo"
    assert at.text_input[0].value == "NewCo" # Re-rendered value

    at.selectbox[0].set_value("Technology").run()
    assert at.session_state["company_industry"] == "Technology"
    assert at.selectbox[0].value == "Technology" # Re-rendered value

def test_2_static_framework_page_display():
    at = AppTest.from_file("app.py")
    # Set session state to simulate previous page completion
    at.session_state["company_name"] = "TestCompany"
    at.session_state["company_industry"] = "Healthcare"
    at.run()

    # Select the second page
    at.sidebar.selectbox[0].set_value("2. Define Static Framework Parameters").run()

    # Assert subheader is correct
    assert at.subheader[0].value == "Reviewing VentureBridge Capital's AI Readiness Framework"
    
    # Assert DataFrames are displayed
    assert len(at.dataframe) == 3

    # Check content of the first dataframe (SYSTEMATIC_OPPORTUNITY_SCORES)
    expected_sys_opp_df = pd.DataFrame(SYSTEMATIC_OPPORTUNITY_SCORES.items(), columns=['Industry', 'Systematic Opportunity Score (0-100)'])
    pd.testing.assert_frame_equal(at.dataframe[0].value, expected_sys_opp_df)

    # Check content of the applied weights dataframe (current_weights_df)
    current_industry = at.session_state["company_industry"]
    expected_current_weights_df = pd.DataFrame([SECTOR_DIMENSION_WEIGHTS.get(current_industry, {})]).T.reset_index()
    expected_current_weights_df.columns = ['Dimension', 'Weight']
    # Use to_string to compare formatted DataFrames or extract raw data.
    # AppTest dataframe.value directly returns the pandas DataFrame.
    pd.testing.assert_frame_equal(at.dataframe[2].value.round(2), expected_current_weights_df.round(2)) # Rounding for potential float differences

    # Check content of the third dataframe (INDUSTRY_BENCHMARKS)
    expected_benchmarks_df = pd.DataFrame(INDUSTRY_BENCHMARKS.items(), columns=['Dimension', 'Benchmark Score (0-100)'])
    pd.testing.assert_frame_equal(at.dataframe[1].value, expected_benchmarks_df) # Changed index to 1 as current_weights_df is 2nd dataframe

def test_3_raw_idiosyncratic_ratings_page_inputs():
    at = AppTest.from_file("app.py")
    # Set session state to simulate previous page completion
    at.session_state["company_name"] = "InnovateCo"
    at.session_state["company_industry"] = "Manufacturing"
    at.session_state["raw_dimension_ratings"] = {
        'Data Infrastructure': 2,
        'AI Governance': 1,
        'Technology Stack': 3,
        'Talent': 2,
        'Leadership': 3,
        'Use Case Portfolio': 1,
        'Culture': 2
    }
    at.run()

    # Select the third page
    at.sidebar.selectbox[0].set_value("3. Collect Raw Idiosyncratic Readiness Ratings").run()

    # Assert all sliders are present (7 dimensions)
    assert len(at.slider) == len(DIMENSIONS)

    # Assert initial slider values match session state
    for i, dim in enumerate(DIMENSIONS):
        assert at.slider[i].value == at.session_state["raw_dimension_ratings"][dim]

    # Test changing a slider value
    data_infra_slider_index = DIMENSIONS.index('Data Infrastructure')
    at.slider[data_infra_slider_index].set_value(4).run()

    # Assert session state is updated
    assert at.session_state["raw_dimension_ratings"]['Data Infrastructure'] == 4
    # Assert slider value is updated on re-render
    assert at.slider[data_infra_slider_index].value == 4

def test_4_normalized_idiosyncratic_readiness_page_calculations_and_display():
    at = AppTest.from_file("app.py")
    # Set session state to simulate previous pages and custom ratings
    at.session_state["company_name"] = "TestCompany"
    at.session_state["company_industry"] = "Manufacturing"
    custom_raw_ratings = {
        'Data Infrastructure': 3,
        'AI Governance': 2,
        'Technology Stack': 4,
        'Talent': 3,
        'Leadership': 4,
        'Use Case Portfolio': 2,
        'Culture': 3
    }
    at.session_state["raw_dimension_ratings"] = custom_raw_ratings
    at.run()

    # Select the fourth page
    at.sidebar.selectbox[0].set_value("4. Calculate Normalized Idiosyncratic Readiness").run()

    # Assert Idiosyncratic Readiness score is calculated and stored
    assert "idiosyncratic_readiness_score" in at.session_state
    expected_score, expected_df, expected_weights = calculate_idiosyncratic_readiness(
        custom_raw_ratings, at.session_state["company_industry"]
    )
    assert at.session_state["idiosyncratic_readiness_score"] == pytest.approx(expected_score, abs=1e-2)

    # Assert metric display
    assert at.metric[0].value == f"{expected_score:.2f} / 100"

    # Assert dimension_scores_df is stored and displayed
    assert "innovateco_dimension_scores_df" in at.session_state
    pd.testing.assert_frame_equal(at.dataframe[0].value.round(2), expected_df.round(2))

    # Assert plot is generated
    assert len(at.pyplot) == 1
    assert isinstance(at.pyplot[0].figure, plt.Figure)

def test_5_pe_org_ai_r_score_page_calculations_and_display():
    at = AppTest.from_file("app.py")
    # Set session state to simulate previous pages' output
    at.session_state["company_name"] = "TestCompany"
    at.session_state["company_industry"] = "Manufacturing"
    at.session_state["raw_dimension_ratings"] = {dim: 3 for dim in DIMENSIONS} # Default all to 3 for consistency
    idiosyncratic_score, _, _ = calculate_idiosyncratic_readiness(at.session_state["raw_dimension_ratings"], "Manufacturing")
    at.session_state["idiosyncratic_readiness_score"] = idiosyncratic_score
    at.session_state["alpha_param"] = 0.6
    at.session_state["beta_param"] = 0.15
    at.session_state["synergy_score"] = 50
    at.run()

    # Select the fifth page
    at.sidebar.selectbox[0].set_value("5. Compute the Overall PE Org-AI-R Score").run()

    # Assert initial slider values
    assert at.slider[0].value == 0.6
    assert at.slider[1].value == 0.15
    assert at.slider[2].value == 50

    # Test changing sliders
    at.slider[0].set_value(0.7).run() # Change alpha
    at.slider[1].set_value(0.2).run() # Change beta
    at.slider[2].set_value(70).run() # Change synergy

    # Assert session state is updated
    assert at.session_state["alpha_param"] == 0.7
    assert at.session_state["beta_param"] == 0.2
    assert at.session_state["synergy_score"] == 70

    # Calculate expected PE Org-AI-R score based on new parameters
    expected_pe_org_ai_r_score = calculate_pe_org_ai_r(
        at.session_state["idiosyncratic_readiness_score"],
        SYSTEMATIC_OPPORTUNITY_SCORES[at.session_state["company_industry"]],
        at.session_state["alpha_param"],
        at.session_state["beta_param"],
        at.session_state["synergy_score"]
    )

    # Assert PE Org-AI-R score is calculated, stored, and displayed
    assert "pe_org_ai_r_score" in at.session_state
    assert at.session_state["pe_org_ai_r_score"] == pytest.approx(expected_pe_org_ai_r_score, abs=1e-2)
    assert at.metric[0].value == f"{expected_pe_org_ai_r_score:.2f} / 100"

def test_6_gap_analysis_page_calculations_and_display():
    at = AppTest.from_file("app.py")
    # Set session state to simulate previous pages
    at.session_state["company_name"] = "TestCompany"
    at.session_state["company_industry"] = "Manufacturing"
    at.session_state["raw_dimension_ratings"] = {
        'Data Infrastructure': 3, 'AI Governance': 2, 'Technology Stack': 4, 'Talent': 3,
        'Leadership': 4, 'Use Case Portfolio': 2, 'Culture': 3
    }
    _, dimension_scores_df, _ = calculate_idiosyncratic_readiness(at.session_state["raw_dimension_ratings"], "Manufacturing")
    at.session_state["innovateco_dimension_scores_df"] = dimension_scores_df
    at.run()

    # Select the sixth page
    at.sidebar.selectbox[0].set_value("6. Perform Gap Analysis Against Industry Benchmarks").run()

    # Assert gap_analysis_df is calculated and stored
    assert "gap_analysis_df" in at.session_state
    
    # Assert DataFrames are displayed
    assert len(at.dataframe) == 1 # Only one explicit dataframe component, the style.format changes return a Styler object.

    # Assert plot is generated (2 plots in this page)
    assert len(at.pyplot) == 2
    assert isinstance(at.pyplot[0].figure, plt.Figure)
    assert isinstance(at.pyplot[1].figure, plt.Figure)

    # Verify a few values in the gap analysis DataFrame
    gap_df = at.session_state["gap_analysis_df"]
    assert gap_df[gap_df['Dimension'] == 'Data Infrastructure']['InnovateCo Score'].iloc[0] == pytest.approx(50.0)
    assert gap_df[gap_df['Dimension'] == 'Data Infrastructure']['Benchmark Score'].iloc[0] == 65
    assert gap_df[gap_df['Dimension'] == 'Data Infrastructure']['Gap'].iloc[0] == pytest.approx(15.0)
    assert gap_df[gap_df['Dimension'] == 'Data Infrastructure']['Priority'].iloc[0] == 'Medium'

def test_7_scenario_analysis_page_calculations_and_display():
    at = AppTest.from_file("app.py")
    # Set session state to simulate previous pages' output
    at.session_state["company_name"] = "TestCompany"
    at.session_state["company_industry"] = "Manufacturing"
    at.session_state["raw_dimension_ratings"] = {
        'Data Infrastructure': 3, 'AI Governance': 2, 'Technology Stack': 4, 'Talent': 3,
        'Leadership': 4, 'Use Case Portfolio': 2, 'Culture': 3
    }
    idiosyncratic_score, _, _ = calculate_idiosyncratic_readiness(at.session_state["raw_dimension_ratings"], "Manufacturing")
    at.session_state["idiosyncratic_readiness_score"] = idiosyncratic_score
    at.session_state["alpha_param"] = 0.6
    at.session_state["beta_param"] = 0.15
    at.session_state["synergy_score"] = 50
    at.run()

    # Select the seventh page
    at.sidebar.selectbox[0].set_value("7. Conduct Scenario Analysis for Strategic Planning").run()

    # Assert scenario_results_df is calculated and stored
    assert "scenario_results_df" in at.session_state
    
    # Assert DataFrame is displayed
    assert len(at.dataframe) == 1

    # Check that 'Base Case' is present and score is reasonable
    scenario_df = at.session_state["scenario_results_df"]
    assert 'Base Case' in scenario_df['Scenario'].values
    assert scenario_df[scenario_df['Scenario'] == 'Base Case']['PE Org-AI-R Score'].iloc[0] == pytest.approx(calculate_pe_org_ai_r(
        idiosyncratic_score, SYSTEMATIC_OPPORTUNITY_SCORES["Manufacturing"], 0.6, 0.15, 50
    ), abs=1e-2)

    # Assert plot is generated
    assert len(at.pyplot) == 1
    assert isinstance(at.pyplot[0].figure, plt.Figure)

def test_8_sensitivity_analysis_page_calculations_and_display():
    at = AppTest.from_file("app.py")
    # Set session state to simulate previous pages' output
    at.session_state["company_name"] = "TestCompany"
    at.session_state["company_industry"] = "Manufacturing"
    at.session_state["raw_dimension_ratings"] = {
        'Data Infrastructure': 3, 'AI Governance': 2, 'Technology Stack': 4, 'Talent': 3,
        'Leadership': 4, 'Use Case Portfolio': 2, 'Culture': 3
    }
    idiosyncratic_score, _, _ = calculate_idiosyncratic_readiness(at.session_state["raw_dimension_ratings"], "Manufacturing")
    at.session_state["idiosyncratic_readiness_score"] = idiosyncratic_score
    at.session_state["alpha_param"] = 0.6
    at.session_state["beta_param"] = 0.15
    at.session_state["synergy_score"] = 50
    at.session_state["sensitivity_change_delta"] = 1 # Initial value
    at.run()

    # Select the eighth page
    at.sidebar.selectbox[0].set_value("8. Perform Sensitivity Analysis of Key Dimensions").run()

    # Assert initial slider value
    assert at.slider[0].value == 1

    # Test changing slider value
    at.slider[0].set_value(2).run()
    assert at.session_state["sensitivity_change_delta"] == 2

    # Assert sensitivity_df is calculated and stored
    assert "sensitivity_df" in at.session_state
    
    # Assert DataFrame is displayed
    assert len(at.dataframe) == 1

    # Check a value from the sensitivity DataFrame
    sensitivity_df = at.session_state["sensitivity_df"]
    # For 'Data Infrastructure (+2 Raw Rating)'
    data_infra_increase_row = sensitivity_df[sensitivity_df['Change'] == 'Data Infrastructure (+2 Raw Rating)']
    assert not data_infra_increase_row.empty
    
    # Calculate expected impact for Data Infrastructure +2
    original_ratings = at.session_state["raw_dimension_ratings"].copy()
    original_ratings['Data Infrastructure'] = min(original_ratings['Data Infrastructure'] + 2, 5)
    
    increased_idiosyncratic_score, _, _ = calculate_idiosyncratic_readiness(original_ratings, at.session_state["company_industry"])
    increased_org_ai_r = calculate_pe_org_ai_r(
        increased_idiosyncratic_score,
        SYSTEMATIC_OPPORTUNITY_SCORES[at.session_state["company_industry"]],
        at.session_state["alpha_param"],
        at.session_state["beta_param"],
        at.session_state["synergy_score"]
    )
    base_org_ai_r = calculate_pe_org_ai_r(
        idiosyncratic_score,
        SYSTEMATIC_OPPORTUNITY_SCORES[at.session_state["company_industry"]],
        at.session_state["alpha_param"],
        at.session_state["beta_param"],
        at.session_state["synergy_score"]
    )
    expected_impact = increased_org_ai_r - base_org_ai_r
    assert data_infra_increase_row['Impact on PE Org-AI-R'].iloc[0] == pytest.approx(expected_impact, abs=1e-2)

    # Assert plot is generated
    assert len(at.pyplot) == 1
    assert isinstance(at.pyplot[0].figure, plt.Figure)


def test_9_exit_readiness_page_calculations_and_display():
    at = AppTest.from_file("app.py")
    # Set session state to simulate company info
    at.session_state["company_name"] = "TestCompany"
    at.session_state["company_industry"] = "Manufacturing"
    at.session_state["exit_visible_score"] = 30
    at.session_state["exit_documented_score"] = 20
    at.session_state["exit_sustainable_score"] = 25
    at.run()

    # Select the ninth page
    at.sidebar.selectbox[0].set_value("9. Evaluate Exit-Readiness").run()

    # Assert initial slider values
    assert at.slider[0].value == 30
    assert at.slider[1].value == 20
    assert at.slider[2].value == 25

    # Test changing sliders
    at.slider[0].set_value(60).run() # Visible score
    at.slider[1].set_value(70).run() # Documented score

    # Assert session state is updated
    assert at.session_state["exit_visible_score"] == 60
    assert at.session_state["exit_documented_score"] == 70

    # Calculate expected Exit-AI-R score
    expected_exit_ai_r_score = calculate_exit_ai_r(
        at.session_state["exit_visible_score"],
        at.session_state["exit_documented_score"],
        at.session_state["exit_sustainable_score"]
    )

    # Assert Exit-AI-R score is calculated, stored, and displayed
    assert "exit_ai_r_score" in at.session_state
    assert at.session_state["exit_ai_r_score"] == pytest.approx(expected_exit_ai_r_score, abs=1e-2)
    assert at.metric[0].value == f"{expected_exit_ai_r_score:.2f} / 100"

    # Assert plot is generated
    assert len(at.pyplot) == 1
    assert isinstance(at.pyplot[0].figure, plt.Figure)

# A combined test for navigation and initial rendering of the app
def test_app_navigation_and_initial_render():
    at = AppTest.from_file("app.py").run()

    # Test initial page load (Welcome page)
    assert at.subheader[0].value == "Mission Briefing: InnovateCo Acquisition Target"
    assert at.session_state["company_name"] == "InnovateCo"
    assert at.session_state["company_industry"] == "Manufacturing"

    # Navigate to each page and check if the subheader or key content loads
    page_options = [
        "2. Define Static Framework Parameters",
        "3. Collect Raw Idiosyncratic Readiness Ratings",
        "4. Calculate Normalized Idiosyncratic Readiness",
        "5. Compute the Overall PE Org-AI-R Score",
        "6. Perform Gap Analysis Against Industry Benchmarks",
        "7. Conduct Scenario Analysis for Strategic Planning",
        "8. Perform Sensitivity Analysis of Key Dimensions",
        "9. Evaluate Exit-Readiness"
    ]
    
    # Pre-populate session state to allow navigation to later pages without errors
    # This is a common pattern for multi-page apps where pages depend on preceding state.
    # We set minimal required state to prevent `st.error` messages and allow pages to render.
    at.session_state["company_name"] = "NavTestCo"
    at.session_state["company_industry"] = "Technology"
    at.session_state["raw_dimension_ratings"] = {dim: 3 for dim in DIMENSIONS}
    idiosyncratic_score, dimension_scores_df, _ = calculate_idiosyncratic_readiness(at.session_state["raw_dimension_ratings"], "Technology")
    at.session_state["idiosyncratic_readiness_score"] = idiosyncratic_score
    at.session_state["innovateco_dimension_scores_df"] = dimension_scores_df
    at.session_state["alpha_param"] = 0.5
    at.session_state["beta_param"] = 0.2
    at.session_state["synergy_score"] = 60
    at.session_state["sensitivity_change_delta"] = 1 # default for sensitivity analysis

    # Re-run after setting session state
    at.run()

    for i, page_name in enumerate(page_options):
        # Select the page from the sidebar
        # The index for the sidebar selectbox is usually 0 if it's the only one on the sidebar
        at.sidebar.selectbox[0].set_value(page_name).run()
        
        # Assert specific content for each page
        if page_name == "2. Define Static Framework Parameters":
            assert at.subheader[0].value == "Reviewing VentureBridge Capital's AI Readiness Framework"
            assert len(at.dataframe) >= 2 # At least static dataframes should be present
        elif page_name == "3. Collect Raw Idiosyncratic Readiness Ratings":
            assert at.subheader[0].value == "Collecting Raw Idiosyncratic Readiness Ratings for InnovateCo"
            assert len(at.slider) == len(DIMENSIONS)
        elif page_name == "4. Calculate Normalized Idiosyncratic Readiness":
            assert at.subheader[0].value == "InnovateCo's Idiosyncratic Readiness: Unpacking Internal AI Capabilities"
            assert "idiosyncratic_readiness_score" in at.session_state
            assert len(at.pyplot) >= 1
        elif page_name == "5. Compute the Overall PE Org-AI-R Score":
            assert at.subheader[0].value == "Calculating the Overall PE Org-AI-R Score"
            assert "pe_org_ai_r_score" in at.session_state
            assert len(at.slider) == 3 # alpha, beta, synergy sliders
        elif page_name == "6. Perform Gap Analysis Against Industry Benchmarks":
            assert at.subheader[0].value == "InnovateCo's AI Readiness Gap Analysis"
            assert "gap_analysis_df" in at.session_state
            assert len(at.pyplot) >= 2
        elif page_name == "7. Conduct Scenario Analysis for Strategic Planning":
            assert at.subheader[0].value == "Scenario Analysis: Modeling InnovateCo's Future AI Readiness"
            assert "scenario_results_df" in at.session_state
            assert len(at.pyplot) >= 1
        elif page_name == "8. Perform Sensitivity Analysis of Key Dimensions":
            assert at.subheader[0].value == "Sensitivity Analysis: Identifying Key AI Investment Levers"
            assert "sensitivity_df" in at.session_state
            assert len(at.pyplot) >= 1
        elif page_name == "9. Evaluate Exit-Readiness":
            assert at.subheader[0].value == "InnovateCo's Exit-AI-R Score: Assessing Attractiveness to Future Buyers"
            assert "exit_ai_r_score" in at.session_state
            assert len(at.slider) == 3 # visible, documented, sustainable sliders
            assert len(at.pyplot) >= 1

    # Final cleanup (optional, depends on test runner setup)
    # plt.close('all') # Ensures no plots are left open
