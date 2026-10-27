"""H-EVAL: state-conditioned evaluation of EEG sleep-onset (first Stage 2) forecasting on Sleep-EDF Expanded.

Subpackages
  heval.data        Sleep-EDF ingestion: EDF/EDF+ readers, lights-off metadata, eligibility, person split, windows
  heval.models      SpecLR (spectral logistic regression), RawCNN, Stage-only / Stage+time references, wake-to-N1 TCNs
  heval.evaluation  Brier/log-loss/calibration/AUROC, person-cluster bootstrap, signed stage attribution, held-out rules
  heval.analysis    TRAIN-only exploratory wake-to-first-N1 analysis

Every function keeps the arithmetic of the original research implementation; each module names its source.
"""
__version__ = '0.2.0'
