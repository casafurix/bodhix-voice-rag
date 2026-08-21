"""Target interface for the organizer-provided rag-local-eval-loop suite —
see wiring-in-the-eval-loop.pdf and TARGET_INTERFACE.md in that tool's own
repo. Not part of BodhiX's own runtime; `embedder.py`/`generator.py` here
are thin adapters onto the real code in `api/`, so the eval loop grades
what POST /ask actually does.
"""
