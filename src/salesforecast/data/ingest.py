"""Download and load the raw retail dataset."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from salesforecast.config import Config
from salesforecast.utils import get_logger, write_json

log = get_logger(__name__)


@dataclass
class RawData:
    sales: pd.DataFrame
    items: pd.DataFrame
    categories: pd.DataFrame
    shops: pd.DataFrame
    test: pd.DataFrame


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(cfg: Config, force: bool = False) -> dict[str, str]:
    """Fetch every raw file into ``raw_dir`` and record a checksum manifest."""
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    base = cfg.data["source_url"].rstrip("/")
    manifest: dict[str, str] = {}
    for filename in cfg.data["files"].values():
        target = cfg.raw_dir / filename
        if target.exists() and not force:
            log.info("%s already present, skipping", filename)
        else:
            url = f"{base}/{filename}"
            log.info("Downloading %s", url)
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                tmp = target.with_suffix(target.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                tmp.replace(target)
        manifest[filename] = _sha256(target)
    write_json(manifest, cfg.raw_dir / "MANIFEST.json")
    return manifest


def load_raw(cfg: Config) -> RawData:
    """Read raw CSVs with compact dtypes."""
    sales = pd.read_csv(
        cfg.raw_file("sales"),
        dtype={
            "date_block_num": "int16",
            "shop_id": "int16",
            "item_id": "int32",
            "item_price": "float64",
            "item_cnt_day": "float32",
        },
    )
    sales["date"] = pd.to_datetime(sales["date"], format="%d.%m.%Y")
    items = pd.read_csv(
        cfg.raw_file("items"), dtype={"item_id": "int32", "item_category_id": "int16"}
    )
    categories = pd.read_csv(cfg.raw_file("categories"), dtype={"item_category_id": "int16"})
    shops = pd.read_csv(cfg.raw_file("shops"), dtype={"shop_id": "int16"})
    test = pd.read_csv(
        cfg.raw_file("test"), dtype={"ID": "int32", "shop_id": "int16", "item_id": "int32"}
    )
    log.info(
        "Loaded %s sales rows, %s items, %s categories, %s shops, %s forecast pairs",
        f"{len(sales):,}",
        f"{len(items):,}",
        len(categories),
        len(shops),
        f"{len(test):,}",
    )
    return RawData(sales=sales, items=items, categories=categories, shops=shops, test=test)
