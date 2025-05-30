# LLM Calibrator

This project provides a Python-based tool to assess the calibration of Large Language Models (LLMs), specifically focusing on how well their use of hedging language aligns with the factual correctness of their answers. It uses Google's Gemini models via their API.


## About

Imagine a world teeming with digital sages, the Large Language Models, or LLMs for short. These aren't your everyday oracles; they've read nearly every book, every article, every whisper on the Internet. Ask them anything, and they'll weave you an answer, often with a voice of serene confidence.

But here's the twist in our tale: even these wise LLMs have their moments of doubt. Sometimes, they're not entirely sure. Other times, they might be confidently mistaken! Just like the case of the infamous 'strawberry' issue.

Now, how can we tell the difference? How do we know when their confident tone truly reflects certainty, and when it's just a well-practiced poker face?

Here comes our LLM-Calibrator.

## Overview

The core idea is to present an LLM with a set of questions. For each question, the tool:
1.  Queries the LLM to get an answer.
2.  Evaluates if the LLM's answer is factually correct (based on provided ground truth).
3.  Detects the presence and extent of "hedging" language (e.g., "might," "could," "possibly," "I'm not sure") in the LLM's response.
4.  Calculates an overall "calibration score" that reflects whether the LLM tends to hedge more when it's incorrect and less when it's correct.

## Features

-   Loads question sets from a JSON file.
-   Interacts with Google's Gemini LLMs (e.g., `gemini-1.5-flash`).
-   Evaluates factual correctness of answers (currently uses simple string matching; questions with `null` ground truth are considered "unanswerable" and treated as correctly handled if the LLM attempts an answer).
-   Detects hedging language by counting occurrences of predefined hedge words and phrases.
-   Calculates a calibration score.
-   Reports detailed results for each question and the overall score.

## Project Structure

```
llm_calibrator.git/
├── datasets/
│   └── questions.json      # Sample questions for the LLM
├── sources/
│   └── llm_calibrator_core.py # Core LLMCalibrator class and logic
├── .env                    # For API key (user-created)
├── llm_calibrator_exp.ipynb # Jupyter notebook for running experiments
└── README.md               # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/aditripaul/llm_calibrator_core
    cd llm_calibrator_core.git
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install google-generativeai nltk python-dotenv
    ```
    The `llm_calibrator_core.py` script will attempt to download the `vader_lexicon` for NLTK if it's not found.

4.  **Set up your API Key:**
    Create a file named `.env` in the root directory of the project (`llm_calibrator.git/`) and add your Google API key:
    ```
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    ```
    Replace `"YOUR_GOOGLE_API_KEY"` with your actual key.

## Usage

The primary way to use this tool is through the Jupyter notebook:

1.  **Open and run `llm_calibrator_exp.ipynb`:**
    This notebook demonstrates the full workflow:
    *   Cell 1: Creates a sample `datasets/questions.json` file. You can modify this file or replace it with your own set of questions. Each question should have a `question` field, an optional `ground_truth` field (use `null` for questions where factual correctness isn't applicable or known, or if you want to test the LLM's response to unanswerable questions), and an `answerable` flag.
    *   Cell 2: Loads the `GOOGLE_API_KEY` from the `.env` file.
    *   Cell 3: Imports the `LLMCalibrator` class, instantiates it, loads data, queries the LLM for each question, evaluates correctness and hedging, and then reports the results and the overall calibration score.

You can also interact with the `LLMCalibrator` class directly in other Python scripts if needed.

## Understanding the Calibration Score

The calibration score is calculated as:

```
Calibration Score = (Average Hedge Score of Incorrect Answers) - (Average Hedge Score of Correct Answers)
```

-   **Hedge Score:** For each LLM answer, this is a simple count of how many predefined hedge words/phrases (e.g., "might", "could", "I'm not sure") are present.
-   **Interpretation:**
    -   **Positive Score:** Generally desirable. It suggests the LLM uses more hedging language when its answers are incorrect and less (is more confident) when its answers are correct.
    -   **Negative Score:** Suggests the LLM might be poorly calibrated by this metric. It could mean the LLM hedges *less* when it's incorrect (overconfident in errors) or, as seen in the sample run, it can be heavily influenced if the dataset primarily contains questions the LLM answers correctly (leading to a low or zero average hedge score for incorrect answers).
    -   **Score Around Zero:** Indicates little difference in average hedging between correct and incorrect answers.

**Note on `ground_truth: null`:** In the current implementation, if a question in `questions.json` has `ground_truth: null`, the `is_factually_correct` method returns `True`. This means such questions contribute to the "correct answers" pool when calculating the calibration score. This is useful for evaluating how an LLM responds to inherently unanswerable or speculative questions.

## Potential Future Improvements

-   **Sophisticated Factual Correctness:** Implement more advanced methods for evaluating factual correctness beyond simple string matching (e.g., semantic similarity, using another LLM as an evaluator, knowledge base lookups).
-   **Advanced Hedge Detection:**
    -   Use more nuanced linguistic feature analysis or machine learning models trained to detect hedging and uncertainty.
    -   Consider the intensity or type of hedge.
-   **Broader Calibration Metrics:** Explore other established calibration metrics from the research literature.
-   **LLM Support:** Abstract the LLM interaction layer to easily support other LLM providers and models.
-   **Question Datasets:** Curate or generate more diverse and challenging question datasets, including a balanced set of questions the LLM is likely to get right and wrong.
-   **Error Analysis:** Provide more detailed error analysis for miscalibrated instances.
-   **Configuration:** Allow easier configuration of hedge word lists, model names, etc.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

## License

This project is licensed under the GNU General Public License v3.0. The authors and the github project must be acknowledged in written form to use any material of the project and derived work must be released under GPL v3.0 license.
