# target_api=torch.fx.experimental.proxy_tensor.make_fx
# model=synthetic
# source_atheris_doc=fixture
import sys

import atheris
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from concurrent_oracle.tests.fixtures.seeded import trace_models as tm


@atheris.instrument_func
def TestOneInput(data: bytes) -> None:
    try:
        make_fx(tm.tiny_fn)(torch.randn(2, 2))
    except Exception:
        return


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
