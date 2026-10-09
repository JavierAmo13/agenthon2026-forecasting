
"""v3 - Track-2 forecaster: regime-scaled joint block bootstrap + text overlay.

The engine replaces point-model machinery entirely. Per the organizers'
solver playbook (reverse-engineered from the winning submissions), the
scoring is CRPS-based and rewards: honest width, fat tails, correct joint
(path) dependence, and modest text-conditioned adjustments.
"""
