"""
Evaluation and scoring logic for the OpenRCA benchmark.

Contains the prediction evaluation function (ported from OpenRCA's evaluate.py)
and the reward/metric functions used by the Verifiers rubric.
"""

import itertools
import json
import re
from datetime import datetime


def _time_within_one_minute(time1_str: str, time2_str: str) -> bool:
    """Check if two timestamp strings are within 1 minute of each other."""
    time_format = "%Y-%m-%d %H:%M:%S"
    try:
        t1 = datetime.strptime(time1_str.strip(), time_format)
        t2 = datetime.strptime(time2_str.strip(), time_format)
        return abs(t1 - t2).total_seconds() <= 60
    except ValueError:
        return False


def evaluate_prediction(
    prediction: str, scoring_points: str
) -> tuple[list[str], list[str], float]:
    """Evaluate a prediction against OpenRCA scoring criteria.

    This is a faithful port of the evaluate() function from the original
    OpenRCA benchmark (main/evaluate.py).

    Args:
        prediction: The model's prediction text containing JSON root cause info.
        scoring_points: The scoring criteria string from the query dataset.

    Returns:
        A tuple of (passed_criteria, failed_criteria, score).
    """
    predict_pattern = (
        r"{\s*"
        r'(?:"root cause occurrence datetime":\s*"(.*?)")?,?\s*'
        r'(?:"root cause component":\s*"(.*?)")?,?\s*'
        r'(?:"root cause reason":\s*"(.*?)")?\s*}'
    )
    predict_matches = re.findall(predict_pattern, prediction)

    predict_results = []
    for match in predict_matches:
        datetime_str, component, reason = match
        if datetime_str or component or reason:
            predict_results.append(
                {
                    "root cause occurrence datetime": datetime_str,
                    "root cause component": component,
                    "root cause reason": reason,
                }
            )

    prediction_length = len(predict_results)

    component_pattern = (
        r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)"
    )
    reason_pattern = (
        r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)"
    )
    time_pattern = (
        r"The (?:\d+-th|only) root cause occurrence time is within "
        r"1 minutes \(i\.e\., <=1min\) of ([^\n]+)"
    )

    components = re.findall(component_pattern, scoring_points)
    reasons = re.findall(reason_pattern, scoring_points)
    times = re.findall(time_pattern, scoring_points)

    scoringpoints_length = max(len(components), len(reasons), len(times), 1)
    scores_num = len(components) + len(reasons) + len(times)

    if scores_num == 0:
        return [], [], 0.0

    scores_get = 0
    passing_criteria: list[str] = []

    if scoringpoints_length == prediction_length:
        best_score = -1
        for perm in itertools.permutations(predict_results):
            current_score = 0
            current_passing: list[str] = []
            for i in range(scoringpoints_length):
                if len(components) == scoringpoints_length:
                    if perm[i]["root cause component"] == components[i]:
                        current_score += 1
                        current_passing.append(components[i])
                if len(reasons) == scoringpoints_length:
                    if perm[i]["root cause reason"] == reasons[i]:
                        current_score += 1
                        current_passing.append(reasons[i])
                if len(times) == scoringpoints_length:
                    if _time_within_one_minute(
                        times[i],
                        perm[i]["root cause occurrence datetime"],
                    ):
                        current_score += 1
                        current_passing.append(times[i])
            if current_score > best_score:
                best_score = current_score
                passing_criteria = current_passing
        scores_get = best_score

    failing_criteria = list(
        set(components + reasons + times) - set(passing_criteria)
    )

    final_score = scores_get / scores_num
    return passing_criteria, failing_criteria, round(final_score, 2)


# ─── Reward / metric functions ───────────────────────────────────────────


async def openrca_score(completion: list[dict[str, str]], answer: str) -> float:
    """Score the model's prediction against OpenRCA scoring criteria.

    Extracts the JSON prediction from the model's final response and evaluates
    it against the scoring points from the dataset.
    """
    prediction_text = ""
    for msg in reversed(completion):
        if msg.get("role") == "assistant" and msg.get("content"):
            prediction_text = msg["content"]
            break

    if not prediction_text:
        return 0.0

    _, _, score = evaluate_prediction(prediction_text, answer)
    return score


async def difficulty_metric(info: str) -> float:
    """Track task difficulty as a metric (0=easy, 1=medium, 2=hard)."""
    parsed = json.loads(info) if isinstance(info, str) else info
    difficulty_map = {"easy": 0.0, "medium": 1.0, "hard": 2.0}
    return difficulty_map.get(parsed.get("difficulty", ""), -1.0)
