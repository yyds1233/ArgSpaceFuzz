# target_api=torch.fx.symbolic_trace
# model=synthetic
# source_atheris_doc=fixture
import sys

import atheris
import torch

from concurrent_oracle.tests.fixtures.seeded import trace_models as tm


@atheris.instrument_func
def TestOneInput(data: bytes) -> None:
    try:
        torch.fx.symbolic_trace(tm.TinyModule())
    except Exception:
        return


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
