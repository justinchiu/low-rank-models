

spec2 = importlib.util.spec_from_file_location(
    "get_logmm",
    "hmm_runners/logmm.py",
)
foo2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(foo2)

import torch

# just default to 256 i guess?

class LogBmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        out = 
        ctx.save_for_backward()
        return out

