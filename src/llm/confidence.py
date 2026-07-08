"""
Confidence estimation.
"""

class ConfidenceCalculator:

    @staticmethod
    def calculate(retrieved, reranked, citations):

        score = 0

        if retrieved >= 8:
            score += 20

        if reranked >= 4:
            score += 40

        if citations >= 3:
            score += 40

        if score >= 80:
            return "High"

        if score >= 50:
            return "Medium"

        return "Low"