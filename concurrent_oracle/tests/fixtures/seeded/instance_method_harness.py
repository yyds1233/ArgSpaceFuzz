# target_api=torch.Tensor.add_
# model=synthetic
# source_atheris_doc=fixture
import sys

import atheris
import torch


@atheris.instrument_func
def TestOneInput(data: bytes) -> None:
    try:
        x = torch.zeros(2, 2)
        y = torch.ones(2, 2)
        x.add_(y)
    except Exception:
        return


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
