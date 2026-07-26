"""Code that must run identically on the PC and on the Raspberry Pi.

Everything in this package is **torch-free** -- numpy and onnxruntime only --
because the Pi agent installs neither torch nor torchvision (see
requirements-pi.txt). Phase 3 deploys this directory to the device verbatim.

That constraint is the point, not an inconvenience. Preprocessing or accuracy
logic that is reimplemented on the device rather than shared with it is the
project's classic silent failure: the Pi reports a lower number, and there is
no way to tell whether the cause is INT8 kernels (the thing being measured) or
a transcription slip (a bug). Sharing the file makes the second explanation
impossible.

Do not import torch, torchvision, or anything from `src.data` here.
"""
