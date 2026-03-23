from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

from core.config import calibration, retrieval
from core.evaluator import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class CalibrationStep:
    cycle: int
    confidence_before: float
    s_c: float
    f_c: float
    k_before: int
    k_after: int
    lambda_1_before: float
    lambda_1_after: float
    lambda_2_before: float
    lambda_2_after: float
    wiki_docs_fetched: int = 0
    confidence_after: float | None = None


class Calibrator:

    def should_calibrate(self, eval_result: EvaluationResult) -> bool:
        return eval_result.confidence < calibration.confidence_threshold

    def adjust_parameters(
        self,
        eval_result: EvaluationResult,
        current_k: int,
        current_lambda_1: float,
        current_lambda_2: float,
        cycle: int,
    ) -> CalibrationStep:
        c_f = eval_result.confidence
        s_c = eval_result.semantic_consistency
        f_c = eval_result.factual_correctness

        delta_k = math.ceil((1.0 - c_f) * calibration.k_increment_multiplier)
        new_k = current_k + delta_k

        step = calibration.lambda_adjust_step
        if s_c < f_c:
            new_l1 = min(0.9, current_lambda_1 + step)
            new_l2 = max(0.1, current_lambda_2 - step)
        else:
            new_l1 = max(0.1, current_lambda_1 - step)
            new_l2 = min(0.9, current_lambda_2 + step)

        total = new_l1 + new_l2
        if total > 0:
            new_l1 /= total
            new_l2 /= total

        logger.info(
            "Calibration cycle %d: C_f=%.3f, k: %d→%d, λ₁: %.2f→%.2f",
            cycle, c_f, current_k, new_k, current_lambda_1, new_l1,
        )

        return CalibrationStep(
            cycle=cycle,
            confidence_before=c_f,
            s_c=s_c,
            f_c=f_c,
            k_before=current_k,
            k_after=new_k,
            lambda_1_before=current_lambda_1,
            lambda_1_after=round(new_l1, 2),
            lambda_2_before=current_lambda_2,
            lambda_2_after=round(new_l2, 2),
        )
