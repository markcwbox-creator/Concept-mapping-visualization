"""Concept Collider — a map of concepts extracted from an open-weight LLM.

Pipeline: concepts -> vectors -> corrected space -> kNN graph -> pairs -> web.
See docs/DESIGN.md for why each stage is shaped the way it is.
"""

__version__ = "0.1.0"
