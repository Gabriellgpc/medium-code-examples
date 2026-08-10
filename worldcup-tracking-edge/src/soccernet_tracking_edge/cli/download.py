"""Download the SoccerNet-Tracking test split (needs the NDA password).

The NDA gates raw broadcast video, not results: publishing aggregate metrics and
a few attributed frames is allowed. Get the password by filling the form at
https://www.soccer-net.org/data and pass it via ``--password`` or the
``SOCCERNET_PASSWORD`` env var. We fetch the ``tracking-2023`` test split (public
GT, ~15–20 GB) into ``data/soccernet``; you then work on a single sequence dir.
"""

from __future__ import annotations

import os

import click
from loguru import logger

from soccernet_tracking_edge.config import SOCCERNET_DIR


@click.command()
@click.option("--password", envvar="SOCCERNET_PASSWORD", default=None,
              help="SoccerNet NDA password (or set SOCCERNET_PASSWORD).")
@click.option("--split", default="test", show_default=True,
              type=click.Choice(["train", "test", "challenge"]))
def download(password: str | None, split: str) -> None:
    """Download the SoccerNet-Tracking (2023) split into data/soccernet."""
    if not password:
        raise click.ClickException(
            "No SoccerNet password. Fill the NDA at https://www.soccer-net.org/data, "
            "then pass --password or set SOCCERNET_PASSWORD."
        )
    from SoccerNet.Downloader import SoccerNetDownloader

    SOCCERNET_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(SOCCERNET_DIR / "hf"))  # keep caches off root
    logger.info(f"Downloading SoccerNet tracking-2023 split={split} → {SOCCERNET_DIR}")
    dl = SoccerNetDownloader(LocalDirectory=str(SOCCERNET_DIR))
    dl.password = password
    dl.downloadDataTask(task="tracking-2023", split=[split])
    logger.info("Done. Point --sequence at one dir under the extracted split.")
