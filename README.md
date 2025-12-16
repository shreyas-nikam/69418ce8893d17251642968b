# PE-AI Readiness Simulator: VentureBridge Capital

![VentureBridge Capital Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

This Streamlit application, titled "**PE-AI Readiness Simulator: VentureBridge Capital**," serves as a simulated lab environment for a **Quantitative Analyst at VentureBridge Capital**. The core mission is to perform comprehensive AI readiness due diligence on a hypothetical acquisition target, **'InnovateCo'**, operating in the Manufacturing sector.

The application guides users through VentureBridge Capital's proprietary, story-driven workflow to conduct a rigorous evaluation of InnovateCo's AI maturity. Each step mirrors real-world tasks in private equity due diligence, designed to help users provide data-driven insights essential for multi-million dollar investment decisions.

The simulation focuses on the following key objectives:

1.  **Assess InnovateCo's internal AI capabilities** (Idiosyncratic Readiness) across key dimensions.
2.  **Quantify its overall AI maturity** using the `PE Org-AI-R Score`, considering both internal strengths and external market opportunities.
3.  **Identify critical AI readiness gaps** against industry benchmarks, pinpointing areas for strategic investment.
4.  **Model future AI readiness scenarios** and perform sensitivity analysis to identify high-impact improvement areas.
5.  **Evaluate InnovateCo's attractiveness to future buyers** from an AI perspective, yielding an `Exit-AI-R Score`.

## Features

The PE-AI Readiness Simulator offers a comprehensive suite of tools and functionalities, structured as a multi-page workflow accessible via the sidebar navigation:

*   **1. Welcome & Company Selection**: Initiate the assessment and select the target company (InnovateCo).
*   **2. Define Static Framework Parameters**: Configure the foundational parameters and weighting for the AI readiness assessment framework.
*   **3. Collect Raw Idiosyncratic Readiness Ratings**: Input raw data and qualitative assessments for InnovateCo's internal AI capabilities across various dimensions.
*   **4. Calculate Normalized Idiosyncratic Readiness**: Process raw ratings into normalized scores for consistent evaluation.
*   **5. Compute the Overall PE Org-AI-R Score**: Calculate InnovateCo's overall AI maturity score, incorporating both internal and external factors.
*   **6. Perform Gap Analysis Against Industry Benchmarks**: Compare InnovateCo's AI readiness with predefined industry benchmarks to identify strengths and weaknesses.
*   **7. Conduct Scenario Analysis for Strategic Planning**: Model different investment and improvement scenarios to project future AI readiness levels.
*   **8. Perform Sensitivity Analysis of Key Dimensions**: Analyze how changes in specific AI readiness dimensions impact the overall score.
*   **9. Evaluate Exit-Readiness**: Determine InnovateCo's attractiveness to future buyers from an AI perspective, calculating its `Exit-AI-R Score`.
*   **Interactive User Interface**: Powered by Streamlit, offering intuitive data input fields, dynamic visualizations, and a streamlined user experience.

## Getting Started

To get a copy of the project up and running on your local machine for development and testing purposes, follow these steps.

### Prerequisites

You will need the following installed on your system:

*   **Python 3.8+**
*   **pip** (Python package installer)
*   **git** (for cloning the repository)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd pe-ai-readiness-simulator
    ```
    *(Replace `<repository_url>` with the actual URL of your repository)*

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies**:
    Create a `requirements.txt` file in the root of your project with the following content:
    ```
    streamlit
    pandas
    matplotlib
    seaborn
    ```
    Then, install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the dependencies are installed, you can run the Streamlit application directly:

1.  **Ensure your virtual environment is activated** (if you created one).
2.  **Run the application from the project root directory**:
    ```bash
    streamlit run streamlit_app.py
    ```

The application will open in your default web browser (usually at `http://localhost:8501`). You can then navigate through the different assessment stages using the sidebar menu.

## Project Structure

The project is organized to promote modularity and ease of development:

```
.
├── streamlit_app.py                      # Main entry point of the Streamlit application.
├── application_pages/                    # Directory containing individual page modules.
│   ├── __init__.py                       # Makes 'application_pages' a Python package.
│   ├── page_1_welcome.py                 # Welcome page and company selection logic.
│   ├── page_2_static_framework.py        # Framework parameter definition page.
│   ├── page_3_raw_idiosyncratic_ratings.py # Raw ratings collection page.
│   ├── page_4_normalized_idiosyncratic_readiness.py # Normalized readiness calculation page.
│   ├── page_5_pe_org_ai_r_score.py       # Overall PE Org-AI-R Score computation page.
│   ├── page_6_gap_analysis.py            # Gap analysis against benchmarks page.
│   ├── page_7_scenario_analysis.py       # Scenario analysis for strategic planning page.
│   ├── page_8_sensitivity_analysis.py    # Sensitivity analysis of key dimensions page.
│   ├── page_9_exit_readiness.py          # Exit-readiness evaluation page.
├── requirements.txt                      # List of Python dependencies.
└── README.md                             # This README file.
```

The `streamlit_app.py` acts as the orchestrator, dynamically loading and displaying content from the modules within the `application_pages` directory based on user selections in the sidebar.

## Technology Stack

*   **Python 3.x**: The core programming language.
*   **Streamlit**: For building the interactive web application user interface.
*   **Pandas**: For data manipulation and analysis.
*   **Matplotlib**: For creating static, animated, and interactive visualizations.
*   **Seaborn**: A high-level data visualization library based on Matplotlib, providing a simpler interface for drawing attractive statistical graphics.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or find any bugs, please feel free to:

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

Please ensure your code adheres to good practices and includes appropriate tests if applicable.

## License

This project is licensed under the MIT License. See the `LICENSE.md` file (to be created) for details.

## Contact

For any inquiries or further information regarding the PE-AI Readiness Simulator, please reach out to the project maintainers or the team at VentureBridge Capital / QuantUniversity.

*   **Project Link:** [Your Project Repository Link Here](https://github.com/yourusername/pe-ai-readiness-simulator) (Replace with actual link)
*   **Organization:** QuantUniversity ([www.quantuniversity.com](https://www.quantuniversity.com/))
