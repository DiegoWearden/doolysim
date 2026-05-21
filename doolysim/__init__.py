"""
doolysim — discrete-event simulator for vLLM serving.

Predicts end-to-end LLM serving performance by emulating vLLM's scheduler
with learned latency models trained from profiled operation timings.
"""

__all__ = ['core', 'emulator', 'benchmark', 'analysis', 'tools', 'request', 'utils']
