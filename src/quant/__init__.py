"""PC-side quantization toolkit (PLAN.md Phase 4).

Turns a deployment configuration into a concrete ONNX artifact, cheaply rejects
the ones that cannot work, and scores the survivors on the optimization
validation split. Phase 6's search drives this; nothing here decides *which*
configs to try.
"""
