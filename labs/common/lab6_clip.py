"""Lab 6, Teil C: Bildmaterial für CLIP zero-shot, offline erzeugt.

Vier Dokumenttypen werden mit PIL und matplotlib gerendert (Inhalt und Layout
hängen vom Seed ab), dazu Ausschnitte der beiden Fotos, die scikit-learn
mitliefert (china.jpg, flower.jpg; Flickr, CC BY 2.0, Urheberangaben in
sklearn/datasets/images/README.txt). Es wird nichts aus dem Netz geladen.
"""
from __future__ import annotations

import io
import textwrap
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common.paths import MODELS_DIR

MODEL_DIR = MODELS_DIR / "clip-b32"
DOC_TYPES = ["gesetz", "tabelle", "formular", "diagramm", "foto"]
W, H = 600, 780


@lru_cache(maxsize=None)
def _font(size: int, bold: bool = False):
    from matplotlib import font_manager

    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def _page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), "white")
    return img, ImageDraw.Draw(img)


def law(rng: np.random.Generator, df) -> Image.Image:
    """Seite mit aufeinanderfolgenden Normen aus dem Gesetzeskorpus."""
    i = int(rng.integers(len(df) - 30))
    img, d = _page()
    fs = int(rng.integers(13, 17))
    d.text((50, 40), df.iloc[i].gesetz, font=_font(fs), fill="gray")
    d.line((50, 40 + fs + 8, W - 50, 40 + fs + 8), fill="gray")
    y = 90
    while y < H - 80:
        row = df.iloc[i]
        d.text((50, y), f"{row.enbez} {row.titel}"[:60], font=_font(fs + 2, True), fill="black")
        y += fs + 16
        for paragraph in row.text.split("\n"):
            for line in textwrap.wrap(paragraph, width=int(540 / (fs * 0.55))):
                if y > H - 50:
                    return img
                d.text((50, y), line, font=_font(fs), fill="black")
                y += int(fs * 1.45)
            y += fs // 2
        y += fs
        i += 1
    return img


def table(rng: np.random.Generator, df=None) -> Image.Image:
    img, d = _page()
    columns = ["Jahr", "Umsatz", "Kosten", "Ergebnis", "Anteil"][: int(rng.integers(3, 6))]
    rows = int(rng.integers(10, 22))
    x0, y0, cw, rh = 40, 80, (W - 80) // len(columns), int(rng.integers(26, 34))
    d.text((x0, 30), "Übersicht " + str(int(rng.integers(2015, 2026))), font=_font(18, True), fill="black")
    for r in range(rows + 1):
        y = y0 + r * rh
        if r == 0:
            d.rectangle((x0, y, x0 + cw * len(columns), y + rh), fill="#dde4ee")
        elif r % 2 == 0 and rng.random() < 0.7:
            d.rectangle((x0, y, x0 + cw * len(columns), y + rh), fill="#f3f3f3")
        for c, name in enumerate(columns):
            text = name if r == 0 else (str(2000 + r) if c == 0 else f"{rng.uniform(1, 999):,.1f}".replace(",", "."))
            d.text((x0 + c * cw + 8, y + 6), text, font=_font(14, r == 0), fill="black")
    for r in range(rows + 2):
        d.line((x0, y0 + r * rh, x0 + cw * len(columns), y0 + r * rh), fill="gray")
    for c in range(len(columns) + 1):
        d.line((x0 + c * cw, y0, x0 + c * cw, y0 + (rows + 1) * rh), fill="gray")
    return img


def form(rng: np.random.Generator, df=None) -> Image.Image:
    img, d = _page()
    title = rng.choice(["Antrag auf Erstattung", "Anmeldung", "Änderungsmitteilung", "Antrag auf Genehmigung"])
    d.text((40, 35), str(title), font=_font(22, True), fill="black")
    fields = ["Name", "Vorname", "Straße, Nr.", "PLZ, Ort", "Geburtsdatum", "Telefon", "E-Mail",
              "Aktenzeichen", "Betrag", "Begründung"]
    y = 100
    for f in rng.permutation(fields)[: int(rng.integers(6, 10))]:
        d.text((40, y), f + ":", font=_font(14), fill="black")
        d.rectangle((190, y - 4, W - 40, y + 22), outline="black")
        y += 48
    y += 10
    for opt in ["ja", "nein", "teilweise"][: int(rng.integers(2, 4))]:
        d.rectangle((40, y, 56, y + 16), outline="black")
        d.text((66, y - 1), opt, font=_font(14), fill="black")
        y += 30
    d.line((40, H - 90, 260, H - 90), fill="black")
    d.text((40, H - 82), "Datum", font=_font(12), fill="black")
    d.line((W - 260, H - 90, W - 40, H - 90), fill="black")
    d.text((W - 260, H - 82), "Unterschrift", font=_font(12), fill="black")
    return img


def chart(rng: np.random.Generator, df=None) -> Image.Image:
    # Figure ohne pyplot: ändert das Backend des Notebooks nicht
    from matplotlib.figure import Figure

    fig = Figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.subplots()
    n = int(rng.integers(5, 12))
    x = np.arange(n)
    kind = rng.choice(["balken", "linie", "beides"])
    if kind in ("balken", "beides"):
        ax.bar(x, rng.uniform(2, 10, n), color=rng.choice(["#3b6ea5", "#c0504d", "#6a9a3a"]))
    if kind in ("linie", "beides"):
        ax.plot(x, np.cumsum(rng.normal(0.5, 1, n)) + 5, marker="o", color="black")
    ax.set_xlabel("Quartal")
    ax.set_ylabel("Wert")
    ax.set_title("Entwicklung " + str(int(rng.integers(2015, 2026))))
    ax.grid(alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return Image.open(buf).convert("RGB")


@lru_cache(maxsize=1)
def _photos():
    # Dateien direkt lesen: der Import von sklearn.datasets dauert auf langsamen Rechnern Sekunden
    from importlib.util import find_spec
    from pathlib import Path

    folder = Path(find_spec("sklearn").submodule_search_locations[0]) / "datasets" / "images"
    return [np.asarray(Image.open(folder / n).convert("RGB")) for n in ("china.jpg", "flower.jpg")]


def photo(rng: np.random.Generator, df=None) -> Image.Image:
    pictures = _photos()
    a = pictures[int(rng.integers(len(pictures)))]
    h, w = a.shape[:2]
    s = int(rng.uniform(0.55, 0.9) * min(h, w))
    y, x = int(rng.integers(0, h - s)), int(rng.integers(0, w - s))
    return Image.fromarray(a[y : y + s, x : x + s]).resize((W, W))


RENDER = {"gesetz": law, "tabelle": table, "formular": form, "diagramm": chart, "foto": photo}


def images(df, per_type: int = 8, seed: int = 1) -> tuple[list[Image.Image], list[str]]:
    """per_type Bilder je Dokumenttyp; ein anderer Seed liefert einen neuen, gleich gebauten Satz."""
    rng = np.random.default_rng(seed)
    imgs, labels = [], []
    for t in DOC_TYPES:
        for _ in range(per_type):
            imgs.append(RENDER[t](rng, df))
            labels.append(t)
    return imgs, labels


def gallery(imgs, labels, n_per_type: int = 1, title=None):
    """Zeigt n_per_type Bilder je Typ nebeneinander."""
    import matplotlib.pyplot as plt

    selection = [i for t in DOC_TYPES for i in [j for j, l in enumerate(labels) if l == t][:n_per_type]]
    fig, axes = plt.subplots(1, len(selection), figsize=(2.2 * len(selection), 2.9))
    for ax, i in zip(np.atleast_1d(axes), selection):
        ax.imshow(imgs[i])
        ax.set_title(title[i] if title else labels[i], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
