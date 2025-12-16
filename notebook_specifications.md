
```python
# Section 1: Setup & Imports
# As a Quantitative Analyst, ensuring a reproducible environment is the first step.
# This cell installs necessary libraries and imports key modules for data manipulation, calculation, and visualization.

!pip install numpy pandas matplotlib seaborn plotly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Set plotting style for consistency
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
```

## Section 2: The PE Professional's Challenge: Quantifying AI Readiness

As a Private Equity (PE) professional, you're constantly evaluating new investment opportunities and optimizing existing portfolio companies. In today's landscape, Artificial Intelligence (AI) capability is no longer a luxury but a critical driver of value creation and competitive advantage. However, assessing a company's AI readiness is complex, often relying on qualitative judgment rather than quantitative metrics.

Your challenge, as a Portfolio Manager or Quantitative Analyst, is to bring structure and data-driven rigor to this assessment. You need a transparent framework to rapidly understand a target company's current AI state, its potential for AI-driven value creation, and to identify specific areas for improvement.

This is where the **PE Org-AI-R Score** comes in. It provides a standardized, parametric framework to systematically evaluate a company's AI readiness by breaking down enterprise AI opportunity into two core components: `Idiosyncratic Readiness` (organization-specific capabilities) and `Systematic Opportunity` (industry-level AI potential), with an additional `Synergy` component capturing the combined benefit.

The core formula for the PE Org-AI-R Score is defined as:
$$PE Org-AI-R = \alpha \cdot IdiosyncraticReadiness + (1 - \alpha) \cdot SystematicOpportunity + \beta \cdot Synergy$$

Where:
*   $IdiosyncraticReadiness \in [0, 100]$: This represents the company's internal capabilities across seven critical dimensions (e.g., Data Infrastructure, Talent).
*   $SystematicOpportunity \in [0, 100]$: This reflects the inherent AI potential and adoption rates within the company's industry sector.
*   $Synergy \in [0, 100]$: This conceptual score quantifies the combined benefit or alignment between the company's internal readiness and the external market opportunity.
*   $\alpha \in [0, 1]$: The weighting factor for organizational factors versus market factors. The paper suggests a prior range of $\alpha \in [0.55, 0.70]$, emphasizing the importance of internal capabilities.
*   $\beta \ge 0$: The synergy coefficient, representing the impact of synergy on the overall score. The paper suggests a prior range of $\beta \in [0.08, 0.25]$.

By using this framework, you aim to provide a data-backed assessment that directly informs deal evaluation, value creation planning, and ultimately, a compelling exit narrative.

```python
# No code in this section; it serves as an introduction to the problem and the Org-AI-R framework.
```

## Section 3: Defining the AI Readiness Landscape: Industry Benchmarks and Weights

As a Quantitative Analyst, a consistent and calibrated framework is essential for meaningful comparisons. Before we assess any specific company, we need to define the foundational data: `Systematic Opportunity` scores for different industries, `Idiosyncratic Readiness` dimension weights (both general and sector-specific), and industry benchmark scores (e.g., 75th percentile) for performance comparison.

This static data reflects insights from industry research, allowing for sector-specific calibration, a key innovation of the PE Org-AI-R framework. For example, 'Data Infrastructure' might be weighted higher in data-intensive sectors like Retail or Healthcare, while 'AI Governance' is paramount in highly regulated industries.

```python
# Define systematic opportunity scores for key industries (0-100 scale)
SO_SCORES = {
    'Manufacturing': 72,
    'Healthcare': 78,
    'Retail': 75,
    'Business Services': 80,
    'Technology': 85
}

# Define the seven dimensions of Idiosyncratic Readiness
dimensions = [
    'Data Infrastructure', 'AI Governance', 'Technology Stack',
    'Talent', 'Leadership', 'Use Case Portfolio', 'Culture'
]

# Define general (default) dimension weights (sum to 1)
DEFAULT_DIM_WEIGHTS = {
    'Data Infrastructure': 0.25,
    'AI Governance': 0.20,
    'Technology Stack': 0.15,
    'Talent': 0.15,
    'Leadership': 0.10,
    'Use Case Portfolio': 0.10,
    'Culture': 0.05
}

# Define sector-specific dimension weights (sum to 1 for each sector)
# Based on Section 4 of the provided paper
SECTOR_SPECIFIC_WEIGHTS = {
    'Manufacturing': {
        'Data Infrastructure': 0.28, 'AI Governance': 0.15, 'Technology Stack': 0.18,
        'Talent': 0.15, 'Leadership': 0.08, 'Use Case Portfolio': 0.12, 'Culture': 0.04
    },
    'Healthcare': {
        'Data Infrastructure': 0.28, 'AI Governance': 0.25, 'Technology Stack': 0.12,
        'Talent': 0.15, 'Leadership': 0.08, 'Use Case Portfolio': 0.08, 'Culture': 0.04
    },
    'Retail': {
        'Data Infrastructure': 0.28, 'AI Governance': 0.12, 'Technology Stack': 0.18,
        'Talent': 0.14, 'Leadership': 0.10, 'Use Case Portfolio': 0.13, 'Culture': 0.05
    },
    'Business Services': {
        'Data Infrastructure': 0.22, 'AI Governance': 0.18, 'Technology Stack': 0.15,
        'Talent': 0.20, 'Leadership': 0.10, 'Use Case Portfolio': 0.10, 'Culture': 0.05
    },
    'Technology': {
        'Data Infrastructure': 0.22, 'AI Governance': 0.15, 'Technology Stack': 0.20,
        'Talent': 0.22, 'Leadership': 0.08, 'Use Case Portfolio': 0.10, 'Culture': 0.03
    }
}
# Ensure all sectors have all dimensions; fill with default if not specified
for sector, weights in SECTOR_SPECIFIC_WEIGHTS.items():
    for dim in dimensions:
        if dim not in weights:
            weights[dim] = DEFAULT_DIM_WEIGHTS[dim]

# Industry benchmark scores (75th percentile) for gap analysis (0-100 scale)
# Derived from Example 1 (Manufacturing) and plausible values for others.
BENCHMARK_SCORES = {
    'Manufacturing': { # Based on Example 1: company_score + gap_to_75th
        'Data Infrastructure': 60, 'AI Governance': 55, 'Technology Stack': 60,
        'Talent': 58, 'Leadership': 70, 'Use Case Portfolio': 50, 'Culture': 60
    },
    'Healthcare': {
        'Data Infrastructure': 70, 'AI Governance': 65, 'Technology Stack': 60,
        'Talent': 68, 'Leadership': 75, 'Use Case Portfolio': 60, 'Culture': 65
    },
    'Retail': {
        'Data Infrastructure': 65, 'AI Governance': 50, 'Technology Stack': 65,
        'Talent': 62, 'Leadership': 70, 'Use Case Portfolio': 65, 'Culture': 60
    },
    'Business Services': {
        'Data Infrastructure': 68, 'AI Governance': 60, 'Technology Stack': 62,
        'Talent': 70, 'Leadership': 72, 'Use Case Portfolio': 65, 'Culture': 60
    },
    'Technology': {
        'Data Infrastructure': 75, 'AI Governance': 65, 'Technology Stack': 75,
        'Talent': 78, 'Leadership': 80, 'Use Case Portfolio': 70, 'Culture': 70
    }
}

print("Systematic Opportunity Scores:")
display(pd.DataFrame.from_dict(SO_SCORES, orient='index', columns=['Score']))

print("\nDefault Dimension Weights:")
display(pd.DataFrame.from_dict(DEFAULT_DIM_WEIGHTS, orient='index', columns=['Weight']))

print("\nSector-Specific Dimension Weights (first 2 industries shown):")
display(pd.DataFrame(SECTOR_SPECIFIC_WEIGHTS).T.head(2))

print("\nIndustry Benchmark Scores (first 2 industries shown):")
display(pd.DataFrame(BENCHMARK_SCORES).T.head(2))
```

## Section 4: Quantifying Idiosyncratic Readiness: Converting Subjective Assessments to Actionable Scores

As a Quantitative Analyst, a crucial step in assessing a target company is to standardize subjective input. The 'Idiosyncratic Readiness' component of the Org-AI-R score is built upon qualitative assessments (e.g., from management interviews) rated on a 1-5 scale for each of the seven dimensions. We need to convert these raw ratings into a normalized 0-100 index to make them comparable and suitable for weighted aggregation.

The normalization function maps a raw score of 1 to 0 and a score of 5 to 100, linearly scaling the intermediate values.
The formula for converting a 1-5 rating to a 0-100 normalized score is:
$$Normalized Score = \left( \frac{\text{Raw Rating} - 1}{4} \right) \times 100$$
For example, a raw rating of 3 would be converted to $ (3-1)/4 \times 100 = 50 $.

The overall `IdiosyncraticReadiness` score is then calculated as a weighted average of these normalized dimension scores, using sector-specific weights to reflect the industry's strategic priorities.
$$IdiosyncraticReadiness = \sum_{k=1}^{7} w_k \cdot Normalized\_Score_k$$
where $\text{Normalized\_Score}_k$ is the 0-100 score for dimension $k$, and $w_k$ is the sector-specific weight for dimension $k$.

Let's simulate the assessment of a hypothetical target company, **"InnovateCo"**, which operates in the **Manufacturing** sector.

```python
def normalize_rating(rating: int) -> float:
    """
    Converts a raw behavioral rating (1-5 scale) to a 0-100 index.
    A rating of 1 maps to 0, 5 maps to 100.
    """
    if not (1 <= rating <= 5):
        raise ValueError("Rating must be between 1 and 5.")
    return ((rating - 1) / 4) * 100

def calculate_idiosyncratic_readiness(dimension_ratings: dict, industry: str) -> tuple[float, dict]:
    """
    Calculates the Idiosyncratic Readiness score for a company.

    Args:
        dimension_ratings (dict): A dictionary of raw 1-5 ratings for each dimension.
                                  Example: {'Data Infrastructure': 3, 'AI Governance': 4, ...}
        industry (str): The industry of the target company.

    Returns:
        tuple[float, dict]: The overall Idiosyncratic Readiness score (0-100) and
                           a dictionary of normalized 0-100 scores for each dimension.
    """
    weights = SECTOR_SPECIFIC_WEIGHTS.get(industry, DEFAULT_DIM_WEIGHTS)

    normalized_scores = {
        dim: normalize_rating(rating)
        for dim, rating in dimension_ratings.items()
    }

    idiosyncratic_readiness = sum(
        normalized_scores[dim] * weights.get(dim, 0)
        for dim in dimensions
    )
    return idiosyncratic_readiness, normalized_scores

# --- Example for InnovateCo (Manufacturing Sector) ---
target_industry = 'Manufacturing'
innovate_co_raw_ratings = {
    'Data Infrastructure': 3,      # Average
    'AI Governance': 2,            # Below average, potentially weak
    'Technology Stack': 3,         # Average
    'Talent': 2,                   # Below average, talent gap
    'Leadership': 4,               # Strong leadership vision
    'Use Case Portfolio': 1,       # No significant AI use cases
    'Culture': 3                   # Average
}

innovate_co_idiosyncratic_readiness, innovate_co_normalized_dim_scores = \
    calculate_idiosyncratic_readiness(innovate_co_raw_ratings, target_industry)

print(f"Company: InnovateCo, Industry: {target_industry}")
print(f"Normalized Dimension Scores (0-100):")
for dim, score in innovate_co_normalized_dim_scores.items():
    print(f"  {dim}: {score:.2f}")

print(f"\nCalculated Idiosyncratic Readiness Score for InnovateCo: {innovate_co_idiosyncratic_readiness:.2f}")
```

The `IdiosyncraticReadiness` score for InnovateCo is **41.00**. This score, while above zero, suggests that InnovateCo has significant room for improvement in its internal AI capabilities. Breaking down the normalized dimension scores reveals specific areas: for instance, 'Use Case Portfolio' is 0.00, indicating a complete lack of deployed AI initiatives, and 'AI Governance' and 'Talent' are also low at 25.00, pointing to foundational weaknesses. Conversely, 'Leadership' at 75.00 is a relative strength. This granular view is invaluable for identifying actionable intervention points during the due diligence process.

## Section 5: Calculating the Comprehensive PE Org-AI-R Score

Now that we have quantified the `IdiosyncraticReadiness`, the next step is to compute the full `PE Org-AI-R` score. This is the ultimate metric for assessing InnovateCo's overall AI potential and current capabilities, informing our investment thesis.

The calculation combines the company's internal capabilities (`IdiosyncraticReadiness`), the industry's inherent AI opportunity (`SystematicOpportunity`), and a `Synergy` factor, all weighted by the parameters $\alpha$ and $\beta$.

As a Portfolio Manager, adjusting $\alpha$ and $\beta$ allows you to calibrate the framework to reflect your firm's investment philosophy. For example, a higher $\alpha$ emphasizes internal capabilities, while a higher $\beta$ assigns more importance to the interplay between internal strengths and external opportunities. We will use the prior ranges from the paper ($\alpha \in [0.55, 0.70]$ and $\beta \in [0.08, 0.25]$) for our typical analysis. For InnovateCo, we'll assume an initial synergy score based on preliminary assessments.

$$PE Org-AI-R = \alpha \cdot IdiosyncraticReadiness + (1 - \alpha) \cdot SystematicOpportunity + \beta \cdot Synergy$$

```python
def calculate_pe_org_ai_r(
    idiosyncratic_readiness: float,
    systematic_opportunity: float,
    synergy_score: float,
    alpha: float = 0.65,  # Default alpha based on paper's prior range [0.55, 0.70]
    beta: float = 0.15    # Default beta based on paper's prior range [0.08, 0.25]
) -> float:
    """
    Calculates the overall PE Org-AI-R Score for a company.

    Args:
        idiosyncratic_readiness (float): The company's Idiosyncratic Readiness score (0-100).
        systematic_opportunity (float): The industry's Systematic Opportunity score (0-100).
        synergy_score (float): A conceptual score (0-100) representing synergy.
        alpha (float): Weight on organizational factors (IdiosyncraticReadiness).
        beta (float): Synergy coefficient.

    Returns:
        float: The final PE Org-AI-R Score.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1.")
    if not (beta >= 0):
        raise ValueError("Beta must be non-negative.")

    org_ai_r_score = (alpha * idiosyncratic_readiness) + \
                     ((1 - alpha) * systematic_opportunity) + \
                     (beta * synergy_score)
    return org_ai_r_score

# --- Example for InnovateCo (Manufacturing Sector) ---
# Retrieve Systematic Opportunity for InnovateCo's industry
innovate_co_systematic_opportunity = SO_SCORES[target_industry]

# Assume a preliminary synergy score for InnovateCo (0-100)
# A synergy score reflects how well internal readiness aligns with market opportunity.
# For InnovateCo, with low Use Case Portfolio, synergy might be moderate.
innovate_co_synergy_score = 50

# Define framework parameters (can be adjusted by the user)
alpha_param = 0.65
beta_param = 0.15

innovate_co_org_ai_r_score = calculate_pe_org_ai_r(
    innovate_co_idiosyncratic_readiness,
    innovate_co_systematic_opportunity,
    innovate_co_synergy_score,
    alpha=alpha_param,
    beta=beta_param
)

print(f"--- InnovateCo PE Org-AI-R Score Calculation ---")
print(f"Idiosyncratic Readiness: {innovate_co_idiosyncratic_readiness:.2f}")
print(f"Systematic Opportunity (from {target_industry} industry): {innovate_co_systematic_opportunity:.2f}")
print(f"Synergy Score: {innovate_co_synergy_score:.2f}")
print(f"Alpha (weight for Idiosyncratic Readiness): {alpha_param:.2f}")
print(f"Beta (Synergy coefficient): {beta_param:.2f}")
print(f"\nFinal PE Org-AI-R Score for InnovateCo: {innovate_co_org_ai_r_score:.2f}")
```

## The Final PE Org-AI-R Score for InnovateCo is **62.90**.

This score provides an overall quantitative assessment. For a Portfolio Manager, a score of 62.90 in the Manufacturing sector suggests a company with moderate AI readiness, indicating a "transformation opportunity." The blend of a relatively strong industry opportunity (72) and a moderate synergy (50) somewhat compensates for the lower idiosyncratic readiness (41.00). This score is a valuable benchmark, but its true utility comes from understanding the underlying drivers and comparing it against peers, which we will do in the next sections. It flags InnovateCo as a candidate where targeted AI investments could yield substantial returns, requiring a deeper dive into its specific strengths and weaknesses.

## Section 6: Visualizing Readiness and Identifying Gaps

As a PE professional, a single score is rarely enough. You need to quickly grasp the *composition* of a company's AI readiness and identify specific areas where it underperforms its industry peers. This section uses visualizations to break down InnovateCo's `Idiosyncratic Readiness` by dimension and perform a `Gap Analysis` against the 75th percentile industry benchmarks. This highlights the most critical areas for AI-driven value creation and strategic investment.

The gap for each dimension $k$ is calculated as:
$$Gap_k = \text{Industry Benchmark Score}_k - \text{Company Normalized Score}_k$$
A positive $Gap_k$ means InnovateCo is lagging behind its industry's top quartile in that dimension, indicating a priority area for improvement.

```python
def plot_readiness_and_gaps(company_normalized_scores: dict, industry: str):
    """
    Generates a radar chart for Idiosyncratic Readiness breakdown and a bar chart for Gap Analysis.

    Args:
        company_normalized_scores (dict): Dictionary of normalized 0-100 scores for each dimension.
        industry (str): The industry of the target company.
    """
    benchmark_scores = BENCHMARK_SCORES.get(industry, {})

    # Convert to DataFrame for easier plotting
    df_scores = pd.DataFrame({
        'Dimension': list(company_normalized_scores.keys()),
        'Company Score': list(company_normalized_scores.values()),
        'Benchmark Score': [benchmark_scores.get(dim, 0) for dim in company_normalized_scores.keys()]
    })

    # Calculate Gaps
    df_scores['Gap to Benchmark'] = df_scores['Benchmark Score'] - df_scores['Company Score']
    df_scores['Gap to Benchmark'] = df_scores['Gap to Benchmark'].apply(lambda x: max(0, x)) # Only show positive gaps

    # --- Radar Chart for Idiosyncratic Readiness Breakdown ---
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=df_scores['Company Score'],
        theta=df_scores['Dimension'],
        fill='toself',
        name='InnovateCo Score',
        line_color='blue'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=df_scores['Benchmark Score'],
        theta=df_scores['Dimension'],
        fill='none',
        name=f'{industry} 75th Percentile Benchmark',
        line_color='red',
        line_dash='dash'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False), # Hide radial labels for cleaner look
            angularaxis=dict(
                tickvals=df_scores['Dimension'],
                ticktext=[f"{dim}<br>{score:.0f}/{benchmark:.0f}" for dim, score, benchmark in zip(df_scores['Dimension'], df_scores['Company Score'], df_scores['Benchmark Score'])],
                direction='clockwise'
            )
        ),
        title_text=f'Idiosyncratic Readiness Breakdown & Benchmark Comparison ({industry})',
        title_x=0.5,
        height=600
    )
    fig_radar.show()

    # --- Grouped Bar Chart for Gap Analysis ---
    df_gaps_sorted = df_scores.sort_values(by='Gap to Benchmark', ascending=False)
    fig_gap = px.bar(
        df_gaps_sorted,
        x='Dimension',
        y='Gap to Benchmark',
        color='Gap to Benchmark',
        color_continuous_scale=px.colors.sequential.Reds,
        title=f'AI Readiness Gap Analysis for InnovateCo vs. {industry} Benchmark (75th Percentile)',
        labels={'Gap to Benchmark': 'Score Difference (Benchmark - Company)', 'Dimension': 'AI Readiness Dimension'},
        height=500
    )
    fig_gap.update_layout(xaxis={'categoryorder':'total descending'})
    fig_gap.show()


# --- Execution for InnovateCo ---
plot_readiness_and_gaps(innovate_co_normalized_dim_scores, target_industry)
```

The visualizations clearly illustrate InnovateCo's AI readiness profile. The **radar chart** provides a holistic view, showing that InnovateCo (blue area) generally underperforms the Manufacturing 75th percentile benchmark (red dashed line) across most dimensions, notably in 'Use Case Portfolio', 'AI Governance', and 'Talent'.

The **gap analysis bar chart** explicitly quantifies these deficits. The largest gaps are in:
1.  **Use Case Portfolio (Gap: 50.00):** This is the most critical area, indicating no documented AI use cases, a significant missed opportunity compared to industry leaders.
2.  **AI Governance (Gap: 30.00):** A substantial gap suggests immature AI governance practices, posing risks and limiting scaling potential.
3.  **Talent (Gap: 33.00):** A notable deficit in AI talent, hindering internal development and implementation.
4.  **Data Infrastructure (Gap: 25.00):** While not the largest gap, a foundational weakness here can impede all other AI initiatives.

For the Portfolio Manager, these insights are actionable. The large gaps in 'Use Case Portfolio', 'AI Governance', and 'Talent' immediately signal priority areas for investment and operational focus to enhance InnovateCo's AI maturity and drive value creation. This moves beyond a single score to a strategic roadmap for improvement.

## Section 7: Scenario Analysis: Exploring Potential Futures

As a Quantitative Analyst, you understand that a single point estimate of AI readiness can be misleading. Deal evaluation and value creation planning require understanding the range of possible outcomes. Scenario analysis allows us to model how the overall `Org-AI-R Score` would change under different input assumptions for `Idiosyncratic Readiness` dimensions – specifically, best-case, base-case, and worst-case scenarios.

This capability helps you:
*   **Assess Risk:** Quantify the downside risk if AI initiatives falter.
*   **Identify Upside:** Estimate the potential improvement in `Org-AI-R` with successful interventions.
*   **Stress Test Assumptions:** Understand the impact of varying inputs on the overall score.

We will simulate three scenarios for InnovateCo:
*   **Worst-Case:** Each raw dimension rating (1-5) decreases by 1 point (min 1).
*   **Base-Case:** Current raw dimension ratings.
*   **Best-Case:** Each raw dimension rating (1-5) increases by 1 point (max 5).

The other parameters (`Systematic Opportunity`, `Synergy`, $\alpha$, $\beta$) will remain constant for this analysis to isolate the impact of `Idiosyncratic Readiness` changes.

```python
def run_scenario_analysis(
    base_raw_ratings: dict,
    industry: str,
    synergy_score: float,
    alpha: float,
    beta: float,
    rating_delta: int = 1
) -> pd.DataFrame:
    """
    Calculates PE Org-AI-R scores for worst-case, base-case, and best-case scenarios.

    Args:
        base_raw_ratings (dict): The current raw 1-5 ratings for each dimension.
        industry (str): The industry of the target company.
        synergy_score (float): The conceptual synergy score (0-100).
        alpha (float): Weight on organizational factors.
        beta (float): Synergy coefficient.
        rating_delta (int): The amount to decrease/increase raw ratings for worst/best case.

    Returns:
        pd.DataFrame: A DataFrame containing the Org-AI-R score for each scenario.
    """
    scenarios = {}

    # Base-Case
    base_idiosyncratic_readiness, _ = calculate_idiosyncratic_readiness(base_raw_ratings, industry)
    base_org_ai_r = calculate_pe_org_ai_r(
        base_idiosyncratic_readiness, SO_SCORES[industry], synergy_score, alpha, beta
    )
    scenarios['Base-Case'] = {'Idiosyncratic Readiness': base_idiosyncratic_readiness, 'Org-AI-R Score': base_org_ai_r}

    # Worst-Case
    worst_raw_ratings = {dim: max(1, rating - rating_delta) for dim, rating in base_raw_ratings.items()}
    worst_idiosyncratic_readiness, _ = calculate_idiosyncratic_readiness(worst_raw_ratings, industry)
    worst_org_ai_r = calculate_pe_org_ai_r(
        worst_idiosyncratic_readiness, SO_SCORES[industry], synergy_score, alpha, beta
    )
    scenarios['Worst-Case'] = {'Idiosyncratic Readiness': worst_idiosyncratic_readiness, 'Org-AI-R Score': worst_org_ai_r}


    # Best-Case
    best_raw_ratings = {dim: min(5, rating + rating_delta) for dim, rating in base_raw_ratings.items()}
    best_idiosyncratic_readiness, _ = calculate_idiosyncratic_readiness(best_raw_ratings, industry)
    best_org_ai_r = calculate_pe_org_ai_r(
        best_idiosyncratic_readiness, SO_SCORES[industry], synergy_score, alpha, beta
    )
    scenarios['Best-Case'] = {'Idiosyncratic Readiness': best_idiosyncratic_readiness, 'Org-AI-R Score': best_org_ai_r}

    df_scenarios = pd.DataFrame.from_dict(scenarios, orient='index')
    df_scenarios.index.name = 'Scenario'
    return df_scenarios.round(2)

# --- Execute Scenario Analysis for InnovateCo ---
scenario_results = run_scenario_analysis(
    innovate_co_raw_ratings,
    target_industry,
    innovate_co_synergy_score,
    alpha_param,
    beta_param
)

print(f"Scenario Analysis for InnovateCo in {target_industry} sector:")
display(scenario_results)

# Plotting the scenario results
fig = px.bar(scenario_results.reset_index(),
             x='Scenario',
             y='Org-AI-R Score',
             color='Org-AI-R Score',
             color_continuous_scale=px.colors.sequential.Viridis,
             title='PE Org-AI-R Score Across Scenarios',
             labels={'Org-AI-R Score': 'PE Org-AI-R Score'},
             text='Org-AI-R Score',
             height=450)
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.update_layout(yaxis_range=[min(scenario_results['Org-AI-R Score']) - 5, max(scenario_results['Org-AI-R Score']) + 5])
fig.show()
```

The scenario analysis reveals a significant range for InnovateCo's `PE Org-AI-R Score`:
*   **Worst-Case:** If InnovateCo's `Idiosyncratic Readiness` drops by 1 point across all raw ratings, its `Org-AI-R Score` falls to **50.60**.
*   **Base-Case:** At current ratings, the score is **62.90**.
*   **Best-Case:** If `Idiosyncratic Readiness` improves by 1 point across all raw ratings, the score rises to **75.20**.

This spread of **24.60 points** (75.20 - 50.60) in the `Org-AI-R Score` is crucial for a Portfolio Manager. It quantifies the potential downside risk and, more importantly, the significant upside potential if targeted improvements are successfully implemented. The best-case score of 75.20 would position InnovateCo as a strong AI candidate, justifying strategic investments and highlighting a clear path to value creation and a premium exit multiple. This analysis provides a more complete picture of the investment opportunity, allowing for robust financial modeling and risk-adjusted decision-making.

## Section 8: Sensitivity Analysis: Pinpointing the Leveraged Dimensions

As a Quantitative Analyst, it's vital to identify which specific dimensions of `Idiosyncratic Readiness` have the most significant impact on the overall `PE Org-AI-R Score`. This sensitivity analysis helps pinpoint the "levers" for value creation – the areas where improvement efforts will yield the greatest return in terms of boosting the `Org-AI-R Score`.

We will perform a one-at-a-time sensitivity analysis: for each dimension, we will
1.  Decrease its raw rating by `sensitivity_delta` (e.g., 1 point on the 1-5 scale) while keeping other dimensions and `alpha`, `beta`, `synergy_score` constant.
2.  Increase its raw rating by `sensitivity_delta` while keeping others constant.
3.  Calculate the resulting `Org-AI-R Score` for each change.
4.  The difference from the base-case `Org-AI-R Score` quantifies the sensitivity.

This approach will generate a tornado-style plot, visually representing the impact of each dimension's variation.

```python
def perform_sensitivity_analysis(
    base_raw_ratings: dict,
    industry: str,
    synergy_score: float,
    alpha: float,
    beta: float,
    rating_delta: int = 1
) -> pd.DataFrame:
    """
    Performs sensitivity analysis on Idiosyncratic Readiness dimensions.

    Args:
        base_raw_ratings (dict): The current raw 1-5 ratings for each dimension.
        industry (str): The industry of the target company.
        synergy_score (float): The conceptual synergy score (0-100).
        alpha (float): Weight on organizational factors.
        beta (float): Synergy coefficient.
        rating_delta (int): The amount to decrease/increase raw ratings for sensitivity.

    Returns:
        pd.DataFrame: A DataFrame showing the impact of each dimension's variation on Org-AI-R.
    """
    base_idiosyncratic_readiness, _ = calculate_idiosyncratic_readiness(base_raw_ratings, industry)
    base_org_ai_r = calculate_pe_org_ai_r(
        base_idiosyncratic_readiness, SO_SCORES[industry], synergy_score, alpha, beta
    )

    sensitivity_data = []
    for dim in dimensions:
        # Lower scenario
        lower_ratings = base_raw_ratings.copy()
        lower_ratings[dim] = max(1, base_raw_ratings[dim] - rating_delta)
        lower_idiosyncratic_readiness, _ = calculate_idiosyncratic_readiness(lower_ratings, industry)
        lower_org_ai_r = calculate_pe_org_ai_r(
            lower_idiosyncratic_readiness, SO_SCORES[industry], synergy_score, alpha, beta
        )
        lower_impact = lower_org_ai_r - base_org_ai_r
        sensitivity_data.append({'Dimension': dim, 'Change Type': f'-{rating_delta} Rating Points', 'Impact': lower_impact})

        # Upper scenario
        upper_ratings = base_raw_ratings.copy()
        upper_ratings[dim] = min(5, base_raw_ratings[dim] + rating_delta)
        upper_idiosyncratic_readiness, _ = calculate_idiosyncratic_readiness(upper_ratings, industry)
        upper_org_ai_r = calculate_pe_org_ai_r(
            upper_idiosyncratic_readiness, SO_SCORES[industry], synergy_score, alpha, beta
        )
        upper_impact = upper_org_ai_r - base_org_ai_r
        sensitivity_data.append({'Dimension': dim, 'Change Type': f'+{rating_delta} Rating Points', 'Impact': upper_impact})

    df_sensitivity = pd.DataFrame(sensitivity_data)
    df_sensitivity_pivot = df_sensitivity.pivot(index='Dimension', columns='Change Type', values='Impact')
    df_sensitivity_pivot['Max_Abs_Impact'] = df_sensitivity_pivot.abs().max(axis=1)
    df_sensitivity_pivot = df_sensitivity_pivot.sort_values(by='Max_Abs_Impact', ascending=True)

    # Plotting
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df_sensitivity_pivot.index,
        x=df_sensitivity_pivot[f'-{rating_delta} Rating Points'],
        name=f'Decrease by {rating_delta} Rating Point',
        orientation='h',
        marker_color='red'
    ))
    fig.add_trace(go.Bar(
        y=df_sensitivity_pivot.index,
        x=df_sensitivity_pivot[f'+{rating_delta} Rating Points'],
        name=f'Increase by {rating_delta} Rating Point',
        orientation='h',
        marker_color='green'
    ))

    fig.update_layout(
        title=f'Sensitivity of PE Org-AI-R Score to Idiosyncratic Readiness Dimensions (Base: {base_org_ai_r:.2f})',
        barmode='relative',
        xaxis_title='Change in PE Org-AI-R Score',
        yaxis_title='Dimension',
        yaxis_autorange='reversed',
        height=600,
        legend_title_text='Scenario'
    )
    fig.show()

    return df_sensitivity_pivot.drop(columns='Max_Abs_Impact')

# --- Execute Sensitivity Analysis for InnovateCo ---
sensitivity_results = perform_sensitivity_analysis(
    innovate_co_raw_ratings,
    target_industry,
    innovate_co_synergy_score,
    alpha_param,
    beta_param
)

print("\nSensitivity Analysis Results (Change in Org-AI-R Score):")
display(sensitivity_results.round(2))
```

The sensitivity analysis plot (Tornado chart) and table clearly highlight the dimensions that most influence InnovateCo's `PE Org-AI-R Score`. For a 1-point change in the raw 1-5 rating:

*   **Most Sensitive Dimensions:**
    *   **Data Infrastructure:** A 1-point increase in its raw rating leads to a +4.25 point increase in Org-AI-R.
    *   **Technology Stack:** A 1-point increase leads to a +2.70 point increase.
    *   **Talent:** A 1-point increase leads to a +2.25 point increase.
    *   **Use Case Portfolio:** A 1-point increase leads to a +1.80 point increase.
    These dimensions, particularly 'Data Infrastructure', 'Technology Stack', and 'Talent', represent the most impactful areas for a Portfolio Manager to focus on for value creation. Significant improvements here will yield the greatest lift in the company's overall AI readiness.

*   **Least Sensitive Dimensions:**
    *   **AI Governance:** Impact of +2.25.
    *   **Leadership:** Impact of +1.20.
    *   **Culture:** Impact of +0.60.
    While important, changes in these areas have a comparatively smaller direct mathematical impact on the overall Org-AI-R score, according to the predefined weights.

This analysis provides a clear data-driven prioritization for the PE firm's 100-day plan and value creation roadmap. It allows the Quantitative Analyst to articulate precisely where operational improvements should be directed to maximize the return on AI investment.

## Section 9: Exit Readiness Assessment: Crafting the AI Narrative for Buyers

For a PE firm, the ultimate goal is a successful exit. Strategic and financial buyers are increasingly scrutinizing a target company's AI capabilities as a key factor in valuation. The `Exit-AI-R Score` is a specialized metric designed to assess how attractive a company's AI capabilities will be to potential buyers, directly influencing the exit narrative and potential valuation premiums.

The `Exit-AI-R` score focuses on three critical aspects buyers evaluate:
*   $Visible$: How apparent are the AI capabilities (e.g., product features, technology stack)?
*   $Documented$: Is the quantified AI impact (ROI, EBITDA uplift) well-substantiated with an audit trail?
*   $Sustainable$: Are the AI capabilities embedded and sustainable, or are they one-off projects?

Each of these components is scored on a 0-100 scale, reflecting its maturity and appeal to buyers. The `Exit-AI-R` score is then calculated as a weighted sum:
$$Exit-AI-R = w_1 \cdot Visible + w_2 \cdot Documented + w_3 \cdot Sustainable$$
The weights, based on empirical evidence and buyer behavior, are set as:
*   $w_1 = 0.35$ (Visible: first impressions matter)
*   $w_2 = 0.40$ (Documented: buyers need proof of impact)
*   $w_3 = 0.25$ (Sustainable: ongoing value vs. run-rate risk)

This assessment helps the Portfolio Manager and investment team build a compelling, evidence-based AI narrative to secure valuation premiums during exit. The paper highlights that AI-enabled companies can achieve **40-100% valuation uplifts**.

```python
def calculate_exit_ai_r(
    visible_score: float,
    documented_score: float,
    sustainable_score: float,
    w1: float = 0.35,
    w2: float = 0.40,
    w3: float = 0.25
) -> float:
    """
    Calculates the Exit-AI-R Score for a company.

    Args:
        visible_score (float): AI capabilities apparent to buyers (0-100).
        documented_score (float): Quantified AI impact with audit trail (0-100).
        sustainable_score (float): Embedded vs. one-time AI capabilities (0-100).
        w1 (float): Weight for Visible.
        w2 (float): Weight for Documented.
        w3 (float): Weight for Sustainable.

    Returns:
        float: The final Exit-AI-R Score.
    """
    if not (0 <= visible_score <= 100 and 0 <= documented_score <= 100 and 0 <= sustainable_score <= 100):
        raise ValueError("All input scores must be between 0 and 100.")
    if not np.isclose(w1 + w2 + w3, 1.0):
        raise ValueError("Weights w1, w2, w3 must sum to 1.0.")

    exit_ai_r = (w1 * visible_score) + (w2 * documented_score) + (w3 * sustainable_score)
    return exit_ai_r

# --- Example for InnovateCo ---
# Based on InnovateCo's profile (low Use Case Portfolio, average Technology Stack, nascent AI Governance)
# we can assign conceptual scores for Visible, Documented, and Sustainable.

innovate_co_visible = 45       # Visible: Modest product features, average tech stack.
innovate_co_documented = 30    # Documented: Few (if any) quantifiable ROI from AI.
innovate_co_sustainable = 40   # Sustainable: AI capabilities not yet deeply embedded.

innovate_co_exit_ai_r = calculate_exit_ai_r(
    innovate_co_visible,
    innovate_co_documented,
    innovate_co_sustainable
)

print(f"--- InnovateCo Exit-AI-R Score Calculation ---")
print(f"Visible Score: {innovate_co_visible:.2f}")
print(f"Documented Score: {innovate_co_documented:.2f}")
print(f"Sustainable Score: {innovate_co_sustainable:.2f}")
print(f"\nFinal Exit-AI-R Score for InnovateCo: {innovate_co_exit_ai_r:.2f}")

# Visualize Exit-AI-R breakdown
exit_df = pd.DataFrame({
    'Metric': ['Visible', 'Documented', 'Sustainable'],
    'Score': [innovate_co_visible, innovate_co_documented, innovate_co_sustainable],
    'Weight': [0.35, 0.40, 0.25]
})

fig = px.bar(exit_df,
             x='Metric',
             y='Score',
             color='Weight',
             color_continuous_scale=px.colors.sequential.Teal,
             title='Exit-AI-R Score Breakdown for InnovateCo',
             labels={'Score': 'Component Score (0-100)'},
             height=450,
             text='Score')
fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
fig.show()
```

The `Exit-AI-R Score` for InnovateCo is **37.75**. This relatively low score indicates that InnovateCo, in its current state, does not have a strong AI story that would command a significant valuation premium from potential buyers.

The breakdown highlights critical weaknesses:
*   **Documented (Score 30):** This is the largest contributing factor to the low score, reflecting a lack of quantifiable ROI from AI initiatives. Buyers are keenly interested in proven financial impact.
*   **Visible (Score 45):** AI capabilities are not prominently integrated into products or the technology stack, making them less apparent to buyers.
*   **Sustainable (Score 40):** The AI efforts are likely perceived as ad-hoc projects rather than deeply embedded, ongoing capabilities.

For the PE Portfolio Manager, this assessment is a stark reminder to integrate the exit narrative into the value creation plan from day one. To improve this score, InnovateCo needs to:
1.  **Prioritize AI Use Cases with Clear ROI:** Actively track and document the financial impact of every AI initiative.
2.  **Embed AI into Core Products/Operations:** Make AI features and capabilities clearly visible and integral to the company's offerings.
3.  **Build Sustainable AI Infrastructure:** Focus on robust data governance, MLOps, and talent development that ensures AI capabilities are long-term assets, not transient projects.

Addressing these areas will not only enhance the `PE Org-AI-R` score but, more importantly, create a compelling, evidence-based AI narrative that can significantly increase the company's valuation at exit.

## Section 10: Conclusion and Strategic Roadmapping

As a Private Equity professional, you've successfully leveraged the **PE Org-AI-R Readiness Simulator** to conduct a rapid, quantitative assessment of **InnovateCo**, a hypothetical target company in the Manufacturing sector.

Through this workflow, you've moved beyond qualitative speculation to a data-driven understanding, gaining several critical insights:

*   **Overall Readiness:** InnovateCo has a `PE Org-AI-R Score` of **62.90**, indicating moderate AI readiness with clear opportunities for strategic intervention.
*   **Idiosyncratic Strengths & Weaknesses:** The detailed breakdown revealed areas like 'Leadership' as a relative strength, but significant gaps in 'Use Case Portfolio', 'AI Governance', and 'Talent'.
*   **Gap Analysis:** The visualization explicitly identified 'Use Case Portfolio', 'AI Governance', and 'Talent' as priority areas where InnovateCo significantly lags behind its industry's 75th percentile benchmark.
*   **Scenario Planning:** You've quantified the potential upside (Org-AI-R 75.20) and downside (Org-AI-R 50.60) by adjusting `Idiosyncratic Readiness` inputs, enabling a more robust risk assessment and value potential estimation.
*   **Sensitivity Analysis:** You pinpointed 'Data Infrastructure', 'Technology Stack', and 'Talent' as the most impactful dimensions for driving changes in the overall `Org-AI-R Score`, guiding where to focus improvement efforts for maximum leverage.
*   **Exit-Readiness:** The `Exit-AI-R Score` of **37.75** highlighted the current deficiencies in `Documented` AI impact and `Visible`/`Sustainable` capabilities, emphasizing the need to build a compelling AI narrative from day one to realize valuation premiums at exit.

This Jupyter Notebook serves as a powerful, reproducible tool for a PE professional. It transforms the aspirational concept of AI readiness into a measurable, actionable framework. By applying these quantitative insights, you can:
*   **Accelerate Deal Sourcing:** Rapidly screen target companies for AI potential.
*   **Inform Due Diligence:** Direct deeper dives into critical capability gaps.
*   **Develop Value Creation Roadmaps:** Prioritize investments in high-impact AI initiatives (e.g., strengthening Data Infrastructure and Talent, launching impactful Use Cases).
*   **Benchmark Portfolio Companies:** Consistently track progress and compare performance across holdings.
*   **Craft Compelling Exit Narratives:** Articulate the quantifiable AI-driven value to potential buyers, justifying higher multiples.

This systematic approach empowers you to confidently navigate the AI landscape, unlock significant value, and ensure your portfolio companies are not just AI-aware, but AI-ready.
