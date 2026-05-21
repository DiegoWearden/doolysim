import os
from . import fake_cuda, fake_pynvml, fake_vllm

# disable GPU to avoid CUDA initialization
os.environ["VLLM_SKIP_P2P_CHECK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_FAKE_TP"] = "1"

# apply emulation patches
fake_pynvml.apply_fake_pynvml_patch()
fake_cuda.apply_fake_cuda()
fake_vllm.apply_vllm_patch()

from vllm import LLM

def emulate_init(model, args={}):
    # Create simulator with limited blocks to potentially trigger preemption
    llm = LLM(model=model, load_format="dummy", enforce_eager=True, **args)
    return llm

if __name__ == "__main__":
    llm = emulate_init(model="gpt2", args={"max_model_len": 512})