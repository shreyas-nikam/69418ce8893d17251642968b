# PE-AI Readiness Simulator - QuantUniversity Lab Project

## Table of Contents
1.  [Project Title and Description](#project-title-and-description)
2.  [Features](#features)
3.  [Getting Started](#getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
4.  [Usage](#usage)
5.  [Project Structure](#project-structure)
6.  [Technology Stack](#technology-stack)
7.  [Contributing](#contributing)
8.  [License](#license)
9.  [Contact](#contact)

---

## 1. Project Title and Description

### PE-AI Readiness Simulator - QuantUniversity Lab Project

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

This Streamlit application, developed as part of a QuantUniversity lab project, serves as a **PE-AI Readiness Simulator**. It empowers quantitative analysts in Private Equity (PE) firms, such as **Alex at VentureBridge Capital**, to conduct a robust, quantitative assessment of a target company's AI readiness.

The primary objective is to evaluate potential acquisition targets (e.g., **InnovateCo** in the Manufacturing sector) by moving beyond superficial claims to deeply understand their internal AI capabilities, external market opportunities, and the synergy between the two. The simulator provides a structured framework to quantify AI maturity, identify specific investment opportunities and risks, and ultimately inform multi-million dollar investment decisions.

The application guides the user through an **end-to-end narrative** comprising a nine-step process, allowing for interactive data input, model application, and UI component manipulation to derive actionable insights for Portfolio Managers.

---

## 2. Features

The PE-AI Readiness Simulator provides a comprehensive suite of tools organized into a sequential workflow:

*   **1. Welcome & Company Selection**: Define the target company and its industry for the assessment.
*   **2. Define Static Framework Parameters**: Configure foundational data, dimension weights, and systematic market opportunity parameters.
*   **3. Collect Raw Idiosyncratic Readiness Ratings**: Input raw data gathered during due diligence for the target company across key AI dimensions (e.g., Data Infrastructure, Talent, AI Governance).
*   **4. Calculate Normalized Idiosyncratic Readiness**: Process raw ratings into a quantitative score representing the target company's internal AI maturity.
*   **5. Compute the Overall PE Org-AI-R Score**: Combine internal readiness with external market opportunity and synergy scores to derive a holistic "Private Equity Organizational AI-Readiness" (PE Org-AI-R) score.
*   **6. Perform Gap Analysis Against Industry Benchmarks**: Benchmark the target company's AI readiness against aggregated industry peers to identify strengths and weaknesses.
*   **7. Conduct Scenario Analysis for Strategic Planning**: Model future AI readiness scores under various investment strategies or operational improvements.
*   **8. Perform Sensitivity Analysis of Key Dimensions**: Determine which AI dimensions have the most significant impact on the overall readiness score, guiding resource allocation.
*   **9. Evaluate Exit-Readiness**: Assess the target company's AI appeal from an exit strategy perspective, considering factors important to future buyers.
*   **Interactive Multi-Page Interface**: Seamless navigation between different assessment stages using a sidebar menu.
*   **Session State Management**: Persistent storage of data and calculated results across different pages for a continuous workflow.
*   **Data-Driven Insights**: Utilizes Pandas for data manipulation and analysis to present quantitative results.

---

## 3. Getting Started

Follow these instructions to set up and run the PE-AI Readiness Simulator on your local machine.

### Prerequisites

*   **Python**: Version 3.8 or higher.
*   **Git**: For cloning the repository.

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/pe-ai-readiness-simulator.git
    cd pe-ai-readiness-simulator
    ```
    *(Note: Replace `your-username/pe-ai-readiness-simulator` with the actual repository URL if available)*

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment**:
    *   **On Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies**:
    Create a `requirements.txt` file in the root directory with the following content:
    ```
    streamlit
    pandas
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

---

## 4. Usage

To run the Streamlit application:

1.  **Ensure your virtual environment is activated** (as described in Installation Step 3).
2.  **Navigate to the project's root directory** (where `app.py` is located).
3.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

    This command will open the application in your default web browser (usually at `http://localhost:8501`).

### Basic Usage Instructions:

*   **Navigation**: Use the `Navigation` selectbox in the sidebar on the left to move between the different stages of the AI readiness assessment.
*   **Input Data**: Interact with sliders, text inputs, and other widgets to provide company-specific data and parameters.
*   **View Results**: Each page will display calculated scores, charts, and tables based on your inputs and the application's models.
*   **Sequential Workflow**: The application is designed to be followed sequentially from "1. Welcome & Company Selection" to "9. Evaluate Exit-Readiness" for a complete assessment. Data entered and calculated in earlier steps will persist and be used in subsequent steps via Streamlit's session state.

---

## 5. Project Structure

The project follows a modular structure to organize different functional pages of the Streamlit application.

```
pe-ai-readiness-simulator/
├── app.py                            # Main Streamlit application entry point
├── requirements.txt                  # List of Python dependencies
├── application_pages/                # Directory containing individual Streamlit pages
│   ├── welcome_company_selection.py  # Page 1: Welcome & Company Selection
│   ├── define_framework_parameters.py# Page 2: Define Static Framework Parameters
│   ├── collect_raw_readiness_ratings.py # Page 3: Collect Raw Ratings
│   ├── calculate_idiosyncratic_readiness.py # Page 4: Calculate Normalized Readiness
│   ├── compute_overall_pe_org_ai_r_score.py # Page 5: Compute Overall Score
│   ├── perform_gap_analysis.py       # Page 6: Perform Gap Analysis
│   ├── conduct_scenario_analysis.py  # Page 7: Conduct Scenario Analysis
│   ├── perform_sensitivity_analysis.py # Page 8: Perform Sensitivity Analysis
│   └── evaluate_exit_readiness.py    # Page 9: Evaluate Exit-Readiness
└── README.md                         # This README file
```

---

## 6. Technology Stack

*   **Python**: The core programming language.
*   **Streamlit**: The open-source app framework used to build and deploy interactive web applications.
*   **Pandas**: A powerful data manipulation and analysis library, used extensively for handling structured data.
*   **Streamlit Session State (`st.session_state`)**: Utilized for managing and persisting data across different pages and user interactions within the application.

---

## 7. Contributing

As this is primarily a lab project, direct contributions might be limited. However, feedback, bug reports, and suggestions for improvement are always welcome!

If you wish to contribute:

1.  **Fork** the repository.
2.  **Create a new branch** (`git checkout -b feature/your-feature-name`).
3.  **Make your changes**.
4.  **Commit your changes** (`git commit -m 'feat: Add new feature X'`).
5.  **Push to the branch** (`git push origin feature/your-feature-name`).
6.  **Open a Pull Request**.

---

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if a LICENSE file is provided. Otherwise, state "This project is open-source and intended for educational purposes, released under a permissive license (e.g., MIT).").

---

## 9. Contact

For any questions, issues, or feedback regarding this PE-AI Readiness Simulator, please contact:

*   **QuantUniversity Support**: [support@quantuniversity.com](mailto:support@quantuniversity.com)
*   **Project Maintainer**: [your.email@example.com](mailto:your.email@example.com) (Replace with an actual email if desired)
*   **QuantUniversity Website**: [https://www.quantuniversity.com](https://www.quantuniversity.com)