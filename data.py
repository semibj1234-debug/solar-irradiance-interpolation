"""Existing train-only transform and seed helper, extracted without math changes."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding="utf-8-sig"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def child_seed(seed, *keys):
    return int(np.random.SeedSequence([seed, *keys]).generate_state(1)[0] % (2**31-1))

@dataclass(frozen=True)
class TrainingTransform:
    coordinate_min: list
    coordinate_span: list
    response_mean: float
    response_sd: float

    @classmethod
    def fit(cls, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.shape != (len(y), 2) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Invalid training data")
        span, sd = np.ptp(x, axis=0), float(y.std(ddof=0))
        if np.any(span <= 0) or sd <= 1e-12:
            raise ValueError("Degenerate training coordinate axis or response; cannot scale")
        return cls(x.min(axis=0).tolist(), span.tolist(), float(y.mean()), sd)

    def coordinates(self, x):
        return (np.asarray(x, float) - self.coordinate_min) / self.coordinate_span

    def response(self, y):
        return (np.asarray(y, float) - self.response_mean) / self.response_sd

    def inverse_prediction(self, prediction):
        return {"mean": prediction["mean"] * self.response_sd + self.response_mean,
                "latent_variance": prediction["latent_variance"] * self.response_sd**2,
                "observation_variance": prediction["observation_variance"] * self.response_sd**2}

    def manifest(self):
        return dict(asdict(self), response_ddof=0, coordinate_fit="training subset only",
                    mean_function="zero after global training standardization")
