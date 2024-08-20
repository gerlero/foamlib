import subprocess
import sys
from pathlib import Path
from typing import Optional, Union
from warnings import warn

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence


class CalledProcessError(subprocess.CalledProcessError):
    """Exception raised when a process fails and `check=True`."""

    def __str__(self) -> str:
        msg = super().__str__()
        if self.stderr:
            msg += f"\n{self.stderr}"
        return msg


class CalledProcessWarning(Warning):
    """Warning raised when a process prints to stderr and `check=True`."""


def check_returncode(
    retcode: int,
    cmd: Union[Sequence[Union[str, Path]], str, Path],
    stderr: Optional[str],
) -> None:
    if retcode != 0:
        raise CalledProcessError(retcode, cmd, None, stderr)
    elif stderr:
        warn(f"Command {cmd} printed to stderr.\n{stderr}", CalledProcessWarning)
