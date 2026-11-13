"""Descriptive features parsed from shop, category and item names (Russian text)."""

from __future__ import annotations

import re

import pandas as pd

ONLINE_SHOP_PREFIXES = ("Интернет-магазин", "Цифровой", "Выездная")
MALL_TYPES = ("ТЦ", "ТРЦ", "ТРК", "ТК", "МТРЦ")

# English labels for the top-level category families (used in reports and the dashboard).
FAMILY_EN = {
    "Аксессуары": "Accessories",
    "Билеты": "Tickets",
    "Доставка": "Delivery",
    "Игровые": "Game consoles",
    "Игры": "Games",
    "Карты": "Payment cards",
    "Кино": "Movies",
    "Книги": "Books",
    "Музыка": "Music",
    "PC": "PC hardware",
    "Подарки": "Gifts",
    "Программы": "Software",
    "Служебные": "Service",
    "Чистые": "Blank media",
    "Элементы": "Batteries",
}


def _encode(values: pd.Series) -> pd.Series:
    """Deterministic label encoding (sorted unique values -> 0..n-1)."""
    cats = sorted(values.unique())
    return values.map({v: i for i, v in enumerate(cats)}).astype("int16")


def shop_features(shops: pd.DataFrame) -> pd.DataFrame:
    df = shops.copy()
    name = df["shop_name"].str.replace(r'[!"«»,.]', " ", regex=True).str.split().fillna("")
    city = name.str[0].fillna("unknown")
    is_online = df["shop_name"].str.startswith(ONLINE_SHOP_PREFIXES)
    city = city.where(~is_online, "online")

    def mall_type(tokens: list[str]) -> str:
        for tok in tokens:
            if tok in MALL_TYPES:
                return tok
        return "street"

    shop_type = name.map(mall_type).where(~is_online, "online")
    df["shop_city"] = city
    df["shop_type"] = shop_type
    df["shop_city_code"] = _encode(city)
    df["shop_type_code"] = _encode(shop_type)
    df["shop_is_online"] = is_online.astype("int8")
    return df


def category_features(categories: pd.DataFrame) -> pd.DataFrame:
    df = categories.copy()
    parts = df["item_category_name"].str.split(" - ", n=1)
    head = parts.str[0].str.strip()
    sub = parts.str[1].fillna(head).str.strip()
    # The head is e.g. "Игры PC" / "Игровые консоли" / "Карты оплаты (Кино, Музыка, Игры)";
    # its first word is a stable top-level family.
    family = head.str.split().str[0]
    df["cat_family"] = family
    df["cat_family_en"] = family.map(FAMILY_EN).fillna(family)
    df["cat_subtype"] = sub
    df["cat_family_code"] = _encode(family)
    df["cat_subtype_code"] = _encode(sub)
    df["cat_is_digital"] = (
        df["item_category_name"].str.contains("Цифра", regex=False).astype("int8")
    )
    return df


def _clean_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[\[\(].*?[\]\)]", " ", name)  # drop bracketed qualifiers
    name = re.sub(r"[^a-zа-яё0-9]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def item_features(items: pd.DataFrame, min_group_size: int) -> pd.DataFrame:
    df = items.copy()
    clean = df["item_name"].map(_clean_name)
    first_word = clean.str.split().str[0].fillna("other")
    counts = first_word.value_counts()
    group = first_word.where(first_word.map(counts) >= min_group_size, "other")
    df["item_name_group"] = group
    df["item_name_group_code"] = _encode(group)
    lower = df["item_name"].str.lower()
    df["item_is_digital"] = lower.str.contains("цифр", regex=False).astype("int8")
    df["item_is_preorder"] = lower.str.contains("предзаказ", regex=False).astype("int8")
    return df
