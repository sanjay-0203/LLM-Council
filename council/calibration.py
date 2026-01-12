"""
Confidence calibration for LLM Council.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict
import math

from loguru import logger


@dataclass
class CalibrationSample:
    """A sample for calibration."""
    model: str
    predicted_confidence: float
    was_correct: bool
    category: str = "general"


@dataclass
class CalibrationResult:
    """Result of calibration analysis."""
    model: str
    expected_calibration_error: float
    overconfidence: float
    underconfidence: float
    calibration_curve: Dict[str, float]
    sample_count: int
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "ece": round(self.expected_calibration_error, 4),
            "overconfidence": round(self.overconfidence, 4),
            "underconfidence": round(self.underconfidence, 4),
            "calibration_curve": {
                k: round(v, 4) for k, v in self.calibration_curve.items()
            },
            "sample_count": self.sample_count,
            "recommendation": self.recommendation
        }


class ConfidenceCalibrator:
    """
    Calibrates model confidence based on historical performance.
    
    Uses Expected Calibration Error (ECE) to measure how well
    a model's confidence matches its actual accuracy.
    """
    
    def __init__(
        self,
        num_bins: int = 10,
        min_samples_per_bin: int = 5
    ):
        self.num_bins = num_bins
        self.min_samples_per_bin = min_samples_per_bin
        
        # Store samples by model
        self.samples: Dict[str, List[CalibrationSample]] = defaultdict(list)
        
        # Calibration adjustments per model
        self.adjustments: Dict[str, float] = {}
    
    def add_sample(
        self,
        model: str,
        predicted_confidence: float,
        was_correct: bool,
        category: str = "general"
    ):
        """Add a calibration sample."""
        sample = CalibrationSample(
            model=model,
            predicted_confidence=max(0.0, min(1.0, predicted_confidence)),
            was_correct=was_correct,
            category=category
        )
        
        self.samples[model].append(sample)
    
    def calculate_ece(self, model: str) -> CalibrationResult:
        """
        Calculate Expected Calibration Error for a model.
        
        ECE = Σ (|bin_count| / total) * |accuracy(bin) - confidence(bin)|
        """
        samples = self.samples.get(model, [])
        
        if len(samples) < self.num_bins:
            return CalibrationResult(
                model=model,
                expected_calibration_error=0.0,
                overconfidence=0.0,
                underconfidence=0.0,
                calibration_curve={},
                sample_count=len(samples),
                recommendation="Insufficient samples for calibration"
            )
        
        # Create bins
        bin_boundaries = [i / self.num_bins for i in range(self.num_bins + 1)]
        bins: Dict[int, List[CalibrationSample]] = defaultdict(list)
        
        for sample in samples:
            bin_idx = min(
                int(sample.predicted_confidence * self.num_bins),
                self.num_bins - 1
            )
            bins[bin_idx].append(sample)
        
        # Calculate ECE
        total_samples = len(samples)
        ece = 0.0
        calibration_curve = {}
        overconfidence = 0.0
        underconfidence = 0.0
        
        for bin_idx in range(self.num_bins):
            bin_samples = bins[bin_idx]
            
            if not bin_samples:
                continue
            
            bin_size = len(bin_samples)
            bin_weight = bin_size / total_samples
            
            # Average confidence in bin
            avg_confidence = sum(s.predicted_confidence for s in bin_samples) / bin_size
            
            # Actual accuracy in bin
            accuracy = sum(1 for s in bin_samples if s.was_correct) / bin_size
            
            # Calibration gap
            gap = abs(accuracy - avg_confidence)
            ece += bin_weight * gap
            
            # Track over/under confidence
            if avg_confidence > accuracy:
                overconfidence += bin_weight * (avg_confidence - accuracy)
            else:
                underconfidence += bin_weight * (accuracy - avg_confidence)
            
            # Store for curve
            bin_label = f"{bin_boundaries[bin_idx]:.1f}-{bin_boundaries[bin_idx + 1]:.1f}"
            calibration_curve[bin_label] = {
                "confidence": avg_confidence,
                "accuracy": accuracy,
                "count": bin_size
            }
        
        # Generate recommendation
        if ece < 0.05:
            recommendation = "Well calibrated. No adjustment needed."
        elif overconfidence > underconfidence:
            adjustment = -overconfidence * 0.5
            recommendation = f"Overconfident. Reduce confidence by {abs(adjustment):.1%}"
            self.adjustments[model] = adjustment
        else:
            adjustment = underconfidence * 0.5
            recommendation = f"Underconfident. Increase confidence by {adjustment:.1%}"
            self.adjustments[model] = adjustment
        
        return CalibrationResult(
            model=model,
            expected_calibration_error=ece,
            overconfidence=overconfidence,
            underconfidence=underconfidence,
            calibration_curve=calibration_curve,
            sample_count=len(samples),
            recommendation=recommendation
        )
    
    def calibrate_confidence(
        self,
        model: str,
        raw_confidence: float
    ) -> float:
        """Apply calibration adjustment to a confidence score."""
        adjustment = self.adjustments.get(model, 0.0)
        calibrated = raw_confidence + adjustment
        
        return max(0.0, min(1.0, calibrated))
    
    def calibrate_all(self) -> Dict[str, CalibrationResult]:
        """Calculate calibration for all models."""
        results = {}
        for model in self.samples:
            results[model] = self.calculate_ece(model)
        return results
    
    def get_model_reliability(self, model: str) -> float:
        """Get overall reliability score for a model."""
        result = self.calculate_ece(model)
        
        if result.sample_count < self.min_samples_per_bin * self.num_bins:
            return 0.5  # Neutral if insufficient data
        
        # Reliability = 1 - ECE
        return max(0.0, 1.0 - result.expected_calibration_error)
    
    def clear_samples(self, model: Optional[str] = None):
        """Clear calibration samples."""
        if model:
            self.samples[model] = []
            self.adjustments.pop(model, None)
        else:
            self.samples.clear()
            self.adjustments.clear()
    
    def export_data(self) -> Dict[str, Any]:
        """Export calibration data."""
        return {
            "samples_per_model": {
                model: len(samples)
                for model, samples in self.samples.items()
            },
            "adjustments": self.adjustments,
            "calibrations": {
                model: self.calculate_ece(model).to_dict()
                for model in self.samples
            }
        }
    
    def import_data(self, data: Dict[str, Any]):
        """Import calibration data."""
        self.adjustments = data.get("adjustments", {})
        # Samples would need to be reconstructed if needed
