# target_api=torch.addmm
# model=synthetic
# source_atheris_doc=fixture
import sys

import atheris
import torch
from torch import addmm as my_addmm


@atheris.instrument_func
def TestOneInput(data: bytes) -> None:
    try:
        inp = torch.zeros(2, 2)
        a = torch.ones(2, 2)
        b = torch.ones(2, 2)
        my_addmm(inp, a, b)
    except Exception:
        return


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
