import json
import os
import re

import google.generativeai as genai
import nltk
import nltk.downloader
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class LLMCalibrator:
    def __init__(self, api_key=None, model_name=None):
        """
        Initializes the LLMCalibrator.

        Args:
            api_key (str, optional): Google API Key. If None, tries to fetch from
                                     GOOGLE_API_KEY environment variable or google.colab.userdata.
            model_name (str, optional): The name of the Gemini model to use.
                                        Defaults to "gemini-1.5-flash".
        """
        if model_name is None:
            raise ValueError(
                "model_name must be provided. Please specify a valid Gemini model name."
            )

        # API Key Configuration
        resolved_api_key = api_key
        if not resolved_api_key:
            resolved_api_key = os.environ.get("GOOGLE_API_KEY")

        if not resolved_api_key:
            try:
                from google.colab import userdata

                resolved_api_key = userdata.get("GOOGLE_API_KEY")
            except ImportError:
                pass  # userdata is not available
            except (
                Exception
            ):  # Catches errors if userdata.get fails (e.g. key not found)
                pass

        if not resolved_api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please pass it to the constructor, "
                "set it as an environment variable, or ensure it's available via google.colab.userdata."
            )

        genai.configure(api_key=resolved_api_key)
        self.model_name = model_name

        # Download VADER lexicon if not already present
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except nltk.downloader.DownloadError:
            nltk.download("vader_lexicon", quiet=True)
        except Exception:  # Fallback for other nltk lookup errors
            nltk.download("vader_lexicon", quiet=True)

        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    # 1. Data Loading and Management
    def load_data(self, filepath):
        """Loads the question set from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return data

    # 2. LLM Interaction (Using Google's Generative AI library for Gemini)
    def query_llm(self, question, model_name=None):
        """
        Queries a large language model using Google's Generative AI library.
        Args:
            question (str): The question to ask the LLM.
            model_name (str, optional): The specific model to query. If None, uses
                                        the model_name set during initialization.
        Returns:
            str: The LLM's answer.
        """
        current_model_name = model_name if model_name else self.model_name
        print(f"Querying {current_model_name} with: {question}")
        try:
            model = genai.GenerativeModel(current_model_name)
            response = model.generate_content(question)
            llm_answer = response.text
            return llm_answer
        except Exception as e:
            print(f"Error querying Gemini model ({current_model_name}): {e}")
            return "Error querying model."

    # 3. Answer Evaluation (Factual Correctness - needs more sophisticated logic)
    def is_factually_correct(self, llm_answer, ground_truth):
        """
        Determines if the LLM's answer is factually correct.
        This is a complex task and might involve string matching, knowledge base lookups, or more advanced NLP techniques.
        """
        if ground_truth is None:
            return True  # If no ground truth, assume it's not meant to be factually evaluated (e.g. unanswerable questions)
        if (
            llm_answer == "Error querying model."
        ):  # If model query failed, it cannot be correct.
            return False
        # Simple string matching for demonstration
        return ground_truth.lower() in llm_answer.lower()

    # 4. Hedge Detection
    def detect_hedging(self, llm_answer):
        """
        Detects the presence and strength of hedging language using VADER sentiment.
        A negative or neutral sentiment might indicate some level of uncertainty.
        This is a very basic proxy and needs refinement.
        """
        if llm_answer == "Error querying model.":
            return 0  # No hedging if there's no valid answer
        scores = self.sentiment_analyzer.polarity_scores(llm_answer)
        # A negative or neutral sentiment might indicate some level of uncertainty
        return scores["neg"] + scores["neu"] * 0.5

    def detect_specific_hedges(self, llm_answer):
        """
        Detects specific hedge words in the LLM's answer.
        """
        if llm_answer == "Error querying model.":
            return 0  # No hedging if there's no valid answer

        hedge_words = [
            "might",
            "could",
            "may",
            "possibly",
            "probably",
            "perhaps",
            "seems",
            "appears",
            "suggests",
            "indicates",
            "I'm not sure",
            "it is believed that",
            "unsure",
            "uncertain",
            "speculative",
            "potential",
            "it's possible",
            "one might argue",
        ]
        hedge_score = 0
        llm_answer_lower = llm_answer.lower()
        for word_pattern in hedge_words:
            # Use regex to match whole words or phrases to avoid partial matches like 'may' in 'mayor'
            if re.search(
                r"\b" + re.escape(word_pattern.lower()) + r"\b", llm_answer_lower
            ):
                hedge_score += 1
        return hedge_score

    # 5. Calibration Metric Calculation
    def calculate_calibration(self, results):
        """
        Calculates a metric to assess the alignment between hedging and correctness.
        A simple metric: the difference in average hedging between incorrect and correct answers.
        Higher positive score suggests model hedges more when incorrect.
        """
        correct_hedging = [res["hedge_score"] for res in results if res["correct"]]
        incorrect_hedging = [
            res["hedge_score"] for res in results if not res["correct"]
        ]

        avg_correct_hedge = (
            sum(correct_hedging) / len(correct_hedging) if correct_hedging else 0
        )
        avg_incorrect_hedge = (
            sum(incorrect_hedging) / len(incorrect_hedging) if incorrect_hedging else 0
        )

        # We want the model to hedge more when it's incorrect.
        # So, a higher avg_incorrect_hedge is better.
        # A lower avg_correct_hedge is also good (confident when correct).
        # Metric: (avg_incorrect_hedge - avg_correct_hedge)
        # A positive value indicates the model hedges more when incorrect than when correct.
        calibration_score = avg_incorrect_hedge - avg_correct_hedge
        return calibration_score

    # 6. Reporting and Analysis
    def report_results(self, results, calibration_score):
        """Prints the results and the overall calibration score."""
        print("\n--- Results ---")
        for result in results:
            print(f"Question: {result['question']}")
            print(f"LLM Answer: {result['llm_answer']}")
            print(f"Correct: {result['correct']}")
            print(f"Hedge Score: {result['hedge_score']:.2f}")
            print("-" * 20)
        print(f"\nOverall Calibration Score: {calibration_score:.2f}")
