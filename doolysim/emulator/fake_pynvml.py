import vllm.third_party.pynvml as pynvml

from .fake_cuda import _get_fake_gpu_device_count

def apply_fake_pynvml_patch():
    pynvml.nvmlInit = lambda *a, **k: None
    pynvml.nvmlShutdown = lambda *a, **k: None
    pynvml.nvmlDeviceGetCount = lambda *a, **k: _get_fake_gpu_device_count()
    print("[Fake pynvml] Patches applied!")