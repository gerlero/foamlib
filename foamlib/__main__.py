"""A command-line interface for the 'foamlib' package."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated

from . import AsyncFoamCase, AsyncSlurmFoamCase, __version__
from ._util import async_to_sync

app = typer.Typer(help=__doc__)


@app.command()
@async_to_sync
async def run(
    cases: Annotated[
        list[Path] | None,
        typer.Argument(help="Case directories", show_default="."),
    ] = None,
    slurm: Annotated[
        bool | None,
        typer.Option(
            help="Use Slurm for running cases.", show_default="use Slurm if available"
        ),
    ] = None,
    max_cpus: Annotated[
        int,
        typer.Option(
            help="Maximum number of concurrent processes (for non-Slurm runs).",
        ),
    ] = AsyncFoamCase.max_cpus,
) -> None:
    """Run one or more OpenFOAM cases."""
    if cases is None:
        cases = [Path.cwd()]

    AsyncFoamCase.max_cpus = max_cpus

    if slurm is None:
        await asyncio.gather(
            *(AsyncSlurmFoamCase(case).run(fallback=True) for case in cases)
        )
    elif slurm:
        await asyncio.gather(*(AsyncSlurmFoamCase(case).run() for case in cases))
    else:
        await asyncio.gather(*(AsyncFoamCase(case).run() for case in cases))


@app.command()
@async_to_sync
async def clean(
    cases: Annotated[
        list[Path] | None,
        typer.Argument(help="Case directories", show_default="."),
    ],
) -> None:
    """Clean one or more OpenFOAM cases."""
    if cases is None:
        cases = [Path.cwd()]

    await asyncio.gather(*(AsyncFoamCase(case).clean() for case in cases))


def _version_callback(*, show: bool) -> None:
    if show:
        typer.echo(f"foamlib {__version__}")
        raise typer.Exit


@app.callback()
def common(  # noqa: D103
    *,
    version: Annotated[
        bool,
        typer.Option(
            "--version", help="Show version and exit.", callback=_version_callback
        ),
    ] = False,
) -> None:
    pass


if __name__ == "__main__":
    app()
