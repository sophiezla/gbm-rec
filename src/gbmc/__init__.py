"""gbmc: can MRI-based models beat distance at predicting where glioblastoma recurs?

Reproducibility contract (plan section 13), carried over from gbm-4d
------------------------------------------------------------------
1. Every stochastic operation draws from :func:`gbmc.rng.stable_seed`, never
   from ``hash()`` or an implicit global RNG.
2. Every script that writes to ``results/`` wraps its work in
   :func:`gbmc.provenance.run_manifest`, and any reportable number is written
   only after :func:`gbmc.provenance.require_clean_tree`.
3. Nothing outside ``config.yaml`` holds a tunable constant; undecided ones are
   ``null`` and raise when read (:func:`gbmc.config.get`).
"""

__version__ = "0.1.0"
