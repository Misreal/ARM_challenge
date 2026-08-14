"""Code that must run identically on the PC and on the Raspberry Pi.

Everything here is torch-free (numpy and onnxruntime only) and deployed to
the device verbatim, on purpose: logic reimplemented on the Pi instead of
shared with it is the project's classic silent failure, since a lower Pi
number then can't be told apart from a transcription slip. Do not import
torch, torchvision, or anything from `src.data` here.
"""
