# target_api=concurrent_oracle.tests.fixtures.seeded.lost_update.update_
# model=synthetic
# source_atheris_doc=fixture
import sys

import atheris
import torch

from concurrent_oracle.tests.fixtures.seeded import lost_update as lu


@atheris.instrument_func
def TestOneInput(data: bytes) -> None:
    try:
        x = torch.zeros(1, dtype=torch.float32)
        lu.update_(x, delay=0.01)
    except Exception:
        return


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
