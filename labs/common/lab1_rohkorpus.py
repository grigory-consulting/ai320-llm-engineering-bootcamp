"""Roher Korpus für Lab 1: Gesetzeskorpus plus das, was eine echte Ernte mitbringt.

`build_raw_corpus()` setzt ihn reproduzierbar (Seed 1) aus fünf Quellen zusammen:

- gii-xml           alle Normen aus normen.parquet und die Einzelnormen ohne Text, die der
                    XML-Abzug von gesetze-im-internet.de enthält (Platzhalter wie "(weggefallen)";
                    Liste in reference/lab1_gii_auszug.json, abgerufen am 2026-10-27)
- gii-xml-nachlauf  ein zweiter Lauf desselben Abrufs, der einen Teil der Normen erneut eingesammelt
                    hat, teils mit anderem Leerraum oder in Unicode-Form NFD
- gii-html          eine Ernte derselben Normen über die HTML-Einzelseiten; die Zeilen um den Normtext
                    entsprechen dem, was eine einfache Textextraktion der Seiten liefert (geprüft an
                    § 103 und § 535 BGB, Art 102 GG, § 4 AO, § 1 SGB 5 am 2026-10-27)
- intern-synthetisch  Bescheide und E-Mails mit personenbezogenen Daten; Namen, Adressen, Rufnummern,
                    IBANs und E-Mail-Adressen sind erfunden (Mustermann, example.org, Bankleitzahl 00000000)
- web-forum         Beiträge aus einem Rechtsforum: Werbung, Texte in anderen Sprachen und einige
                    deutsche Fragen und Antworten (selbst verfasst)

`pii_annotations(raw)` liefert zu den internen Texten, welche Angaben darin stecken (ein kleines
annotiertes Testset für Recall-Messungen).
"""
from __future__ import annotations

import json
import random
import unicodedata

import pandas as pd

from common import corpus
from common.paths import LAB_DIR, SEED

EXCERPT = LAB_DIR / "reference" / "lab1_gii_auszug.json"
FETCH_DATE = "2026-10-27"
N_RERUN = 250
N_HTML = 400

# Navigationszeilen einer Einzelnorm-Seite, so wie sie eine einfache Textextraktion liefert
_HEADER_NAV = ["zurück", "weiter", "Nichtamtliches Inhaltsverzeichnis"]
_FOOTER_NAV = ["zum Seitenanfang", "Impressum", "Datenschutz", "Barrierefreiheitserklärung", "Feedback-Formular"]


def _excerpt() -> dict:
    return json.loads(EXCERPT.read_text(encoding="utf-8"))


def _url_html(slug: str, designation: str) -> str:
    nr = designation.split(" ", 1)[1].strip()
    page = f"art_{nr}" if designation.startswith("Art") else f"__{nr}"
    return f"https://www.gesetze-im-internet.de/{slug}/{page}.html"


def _html_text(head: dict, designation: str, title: str, text: str) -> str:
    abbr = head["amtabk"] or head["jurabk"]
    name = head["langue"] + (f" ({head['amtabk']})" if head["amtabk"] else "")
    rows = [f"{designation} {abbr} - Einzelnorm", *_HEADER_NAV, name, designation]
    if title:
        rows.append(title)
    return "\n".join([*rows, *text.split("\n"), *_FOOTER_NAV])


def _whitespace_variant(text: str, rng: random.Random) -> str:
    kind = rng.randrange(3)
    if kind == 0:
        return text.replace("\n", "\r\n")
    if kind == 1:
        return text.replace(". ", ".  ")
    return "\n".join(z + " " for z in text.split("\n")) + "\n"


# ---------------------------------------------------------------- synthetische interne Texte
_FIRST_NAMES = ["Erika", "Max", "Petra", "Otto", "Gabi", "Klaus", "Anna", "Jan", "Lena", "Fritz", "Ute", "Tim"]
_SURNAMES = ["Mustermann", "Musterfrau", "Beispiel", "Probe", "Testmann", "Platzhalter", "Normalverbraucher", "Exempel"]
_STREETS = ["Musterstraße", "Beispielweg", "Probeallee", "Testgasse", "Platzhalterring"]
_PLACES = [("10115", "Berlin"), ("60311", "Frankfurt am Main"), ("20095", "Hamburg"), ("50667", "Köln"), ("80331", "München")]
# Rufnummern aus Blöcken, die erkennbar keinem Anschluss gehören sollen; mehrere Schreibweisen
_NETWORKS = [("030", "23125"), ("069", "90009"), ("040", "66969"), ("0221", "4710"), ("089", "99998")]


def _iban(rng: random.Random) -> str:
    account = f"{rng.randrange(10**9, 10**10)}"
    bban = "00000000" + account
    check_set = 98 - int(bban + "131400") % 97      # DE = 13 14, Prüfziffer nach ISO 13616
    raw = f"DE{check_set:02d}{bban}"
    return " ".join(raw[i:i + 4] for i in range(0, len(raw), 4))


def _phone(rng: random.Random) -> str:
    area_code, block = rng.choice(_NETWORKS)
    extension = f"{rng.randrange(100, 1000)}"
    form = rng.randrange(5)
    if form == 0:
        return f"{area_code} {block}-{extension}"
    if form == 1:
        return f"+49 {area_code[1:]} {block}{extension}"
    if form == 2:
        return f"({area_code}) {block} {extension}"
    if form == 3:
        return f"{area_code}/{block}-{extension}"
    return f"+49 (0){area_code[1:]} {block}-{extension}"


def _person(rng: random.Random) -> dict:
    fwd, after = rng.choice(_FIRST_NAMES), rng.choice(_SURNAMES)
    postcode, place = rng.choice(_PLACES)
    domain = rng.choice(["example.org", "example.com", "example.net"])
    return {
        "name": f"{fwd} {after}", "anrede": "Frau" if fwd in {"Erika", "Petra", "Gabi", "Anna", "Lena", "Ute"} else "Herr",
        "nachname": after, "strasse": f"{rng.choice(_STREETS)} {rng.randrange(1, 120)}", "plz": postcode, "ort": place,
        "email": f"{fwd.lower()}.{after.lower()}@{domain}", "telefon": _phone(rng), "iban": _iban(rng),
    }


def _date(rng: random.Random) -> str:
    return f"{rng.randrange(1, 29):02d}.{rng.randrange(1, 10):02d}.2026"


def _amount(rng: random.Random) -> str:
    euro = rng.randrange(40, 9000)
    return f"{euro:,}".replace(",", ".") + f",{rng.randrange(100):02d} Euro"


_RBB_OBJECTION = (
    "Rechtsbehelfsbelehrung: Gegen diesen Bescheid kann innerhalb eines Monats nach seiner Bekanntgabe Einspruch "
    "eingelegt werden. Der Einspruch ist bei dem oben genannten Finanzamt schriftlich einzureichen, diesem "
    "elektronisch zu übermitteln oder dort zur Niederschrift zu erklären. Die Frist für die Einlegung des "
    "Einspruchs beginnt mit Ablauf des Tages, an dem Ihnen dieser Bescheid bekannt gegeben worden ist."
)
_RBB_APPEAL = (
    "Rechtsbehelfsbelehrung: Gegen diesen Bescheid kann innerhalb eines Monats nach Bekanntgabe Widerspruch "
    "erhoben werden. Der Widerspruch ist schriftlich oder zur Niederschrift bei der Behörde einzulegen, die den "
    "Bescheid erlassen hat. Die Frist ist auch gewahrt, wenn der Widerspruch innerhalb der Frist bei der "
    "Widerspruchsbehörde eingeht."
)
_RBB_FINE = (
    "Rechtsbehelfsbelehrung: Gegen diesen Bußgeldbescheid kann innerhalb von zwei Wochen nach Zustellung bei der "
    "oben bezeichneten Verwaltungsbehörde schriftlich oder zur Niederschrift Einspruch eingelegt werden. Bei einem "
    "Einspruch kann auch eine für den Betroffenen nachteiligere Entscheidung getroffen werden."
)


def _notice(kind: str, p: dict, rng: random.Random) -> tuple[str, dict]:
    datum, amount = _date(rng), _amount(rng)
    contact = f"Telefon {_phone(rng)}" if rng.random() < 0.5 else f"Tel. {_phone(rng)}"
    clerk = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_SURNAMES)}"
    clerk_phone = contact.split(" ", 1)[1]
    if kind == "steuer":
        tax_no = f"{rng.randrange(10, 99)}/{rng.randrange(100, 999)}/{rng.randrange(10000, 99999)}"
        text = (
            f"Finanzamt {p['ort']}\nSachbearbeitung: {clerk}, {contact}\n"
            f"{p['anrede']} {p['name']}\n{p['strasse']}\n{p['plz']} {p['ort']}\n"
            f"Steuernummer {tax_no}\nBescheid für 2025 über Einkommensteuer vom {datum}\n"
            f"Sehr geehrte{'r' if p['anrede'] == 'Herr' else ''} {p['anrede']} {p['nachname']},\n"
            f"die Einkommensteuer wird auf {amount} festgesetzt. "
            + rng.choice([
                "Die Erstattung überweisen wir auf das Konto mit der IBAN {iban}.",
                "Der Erstattungsbetrag wird auf Ihr Konto IBAN {iban} überwiesen.",
                "Bitte zahlen Sie den Betrag bis zum Fälligkeitstag; eine Lastschrift von der IBAN {iban} ist vereinbart.",
            ]).format(iban=p["iban"])
            + " Der Bescheid ergeht nach § 155 AO.\n" + _RBB_OBJECTION
        )
        pii = {"NAME": [p["name"], clerk], "TELEFON": [clerk_phone], "IBAN": [p["iban"]], "EMAIL": []}
    elif kind == "wohngeld":
        case_no = f"WG-{rng.randrange(1000, 9999)}/26"
        text = (
            f"Stadtverwaltung {p['ort']}, Wohngeldstelle\nAktenzeichen {case_no}\n"
            f"{p['anrede']} {p['name']}, {p['strasse']}, {p['plz']} {p['ort']}\n"
            f"Bescheid über Wohngeld vom {datum}\n"
            f"Auf Ihren Antrag vom {_date(rng)} wird Ihnen Wohngeld in Höhe von monatlich {amount} bewilligt. "
            f"Die Zahlung erfolgt auf das Konto mit der IBAN {p['iban']}. "
            f"Bei Rückfragen erreichen Sie {clerk} unter {contact} oder per E-Mail an wohngeld@stadt.example.org.\n"
            + _RBB_APPEAL
        )
        pii = {"NAME": [p["name"], clerk], "TELEFON": [clerk_phone], "IBAN": [p["iban"]], "EMAIL": ["wohngeld@stadt.example.org"]}
    else:
        case_no = f"OWi {rng.randrange(100, 999)}.{rng.randrange(10000, 99999)}.6"
        kmh = rng.randrange(11, 40)
        text = (
            f"Zentrale Bußgeldstelle\nAktenzeichen {case_no}\n"
            f"{p['anrede']} {p['name']}, geboren am {rng.randrange(1, 29):02d}.{rng.randrange(1, 13):02d}."
            f"{rng.randrange(1950, 2004)}, {p['strasse']}, {p['plz']} {p['ort']}\n"
            f"Bußgeldbescheid vom {datum}\n"
            f"Ihnen wird vorgeworfen, am {_date(rng)} die zulässige Höchstgeschwindigkeit außerhalb geschlossener "
            f"Ortschaften um {kmh} km/h überschritten zu haben. Gegen Sie wird eine Geldbuße festgesetzt. "
            f"Zahlungen bitte unter Angabe des Aktenzeichens; Rückfragen an {clerk}, {contact}.\n"
            + _RBB_FINE
        )
        pii = {"NAME": [p["name"], clerk], "TELEFON": [clerk_phone], "IBAN": [], "EMAIL": []}
    return text, pii


def _email(p: dict, rng: random.Random) -> tuple[str, dict]:
    datum = _date(rng)
    concern = rng.choice([
        ("Rückfrage zu meinem Steuerbescheid", f"in meinem Bescheid vom {datum} wurden die Fahrtkosten nicht "
         "berücksichtigt. Ich habe die Belege mit der Erklärung eingereicht und bitte um Prüfung."),
        ("Wohngeldantrag: fehlende Unterlagen", f"Sie hatten mit Schreiben vom {datum} Unterlagen nachgefordert. "
         "Die Mietbescheinigung meines Vermieters reiche ich hiermit nach."),
        ("Einspruch Bußgeldbescheid", f"gegen den Bußgeldbescheid vom {datum} lege ich Einspruch ein. Ich war zu "
         "dem genannten Zeitpunkt nicht der Fahrer des Fahrzeugs."),
        ("Bitte um Ratenzahlung", f"die Nachzahlung aus dem Bescheid vom {datum} kann ich nicht auf einmal leisten. "
         "Ich bitte um Ratenzahlung in sechs Monatsraten."),
        ("Änderung meiner Bankverbindung", "bitte überweisen Sie künftige Zahlungen auf mein neues Konto. "
         "Das alte Konto wird zum Monatsende aufgelöst."),
    ])
    with_iban = concern[0] == "Änderung meiner Bankverbindung" or rng.random() < 0.3
    with_phone = rng.random() < 0.7
    rows = [
        f"Von: {p['name']} <{p['email']}>",
        "An: poststelle@behoerde.example.org",
        f"Betreff: {concern[0]}",
        "",
        "Sehr geehrte Damen und Herren,",
        concern[1] + (f" Meine IBAN lautet {p['iban']}." if with_iban else ""),
        rng.choice(["Für Rückfragen stehe ich gern zur Verfügung.", "Vielen Dank für Ihre Mühe.",
                    "Ich bitte um eine kurze Bestätigung des Eingangs."]),
        "",
        "Mit freundlichen Grüßen",
        p["name"],
        f"{p['strasse']}, {p['plz']} {p['ort']}",
    ]
    if with_phone:
        rows.append(rng.choice(["Tel.: ", "Mobil ", "Telefon: ", ""]) + p["telefon"])
    pii = {"NAME": [p["name"]], "EMAIL": [p["email"], "poststelle@behoerde.example.org"],
           "TELEFON": [p["telefon"]] if with_phone else [], "IBAN": [p["iban"]] if with_iban else []}
    return "\n".join(rows), pii


# ---------------------------------------------------------------- Forum: Werbung, andere Sprachen, echte Beiträge
_ADS = [
    "Kredit ohne Schufa! Sofortzusage in 5 Minuten!\nKredit ohne Schufa! Sofortzusage in 5 Minuten!\n"
    "Jetzt beantragen: https://kredit.example.com/?ref=forum\nKredit ohne Schufa! Sofortzusage in 5 Minuten!",
    "Anwalt Berlin günstig\nAnwalt Hamburg günstig\nAnwalt München günstig\nAnwalt Köln günstig\n"
    "Anwalt Frankfurt günstig\nhttps://anwalt-guenstig.example.net\nhttps://anwalt-guenstig.example.net/kontakt",
    "Abnehmen ohne Diät, 10 Kilo in 2 Wochen!!!\nAbnehmen ohne Diät, 10 Kilo in 2 Wochen!!!\n"
    "Bestellen Sie jetzt unter https://schlank.example.com",
    "BONUS 500 EURO im Online-Casino\nBONUS 500 EURO im Online-Casino\nBONUS 500 EURO im Online-Casino\n"
    "https://casino.example.org/bonus",
    "Mit Krypto 3.000 Euro am Tag verdienen. Klicken Sie hier: https://krypto.example.net\n"
    "Mit Krypto 3.000 Euro am Tag verdienen. Klicken Sie hier: https://krypto.example.net",
    "Abmahnung erhalten? Wir helfen sofort!\nAbmahnung erhalten? Wir helfen sofort!\n"
    "Kostenlose Ersteinschätzung: https://abmahnung.example.com\nAbmahnung erhalten? Wir helfen sofort!",
    "Gewinnspiel: iPhone gewinnen!\nNur heute: iPhone gewinnen!\nGewinnspiel: iPhone gewinnen!\n"
    "Teilnahme unter https://gewinn.example.org",
    "Scheidung online, schnell und billig\nScheidung online, schnell und billig\n"
    "https://scheidung.example.net https://scheidung.example.net/preise",
    "Führerschein zurück ohne MPU, garantiert\nFührerschein zurück ohne MPU, garantiert\n"
    "Infos: https://mpu.example.com",
    "Steuern sparen mit unserem Geheimtipp\nSteuern sparen mit unserem Geheimtipp\nSteuern sparen mit unserem Geheimtipp\n"
    "Jetzt lesen: https://steuertipp.example.org",
]
_FOREIGN = [
    "In Germany, a landlord may terminate a residential lease only with a legitimate interest. The notice period "
    "depends on how long the tenancy has lasted and can be up to nine months for the landlord.",
    "Does anyone know whether a verbal contract is binding in Germany? I agreed to buy a used car from a neighbour "
    "and now he wants more money than we discussed on the phone.",
    "The German Civil Code distinguishes between the obligation to transfer ownership and the transfer itself. "
    "This separation principle confuses many students in their first semester.",
    "My employer wants me to work twelve hours a day during the trade fair. Is that allowed under the working time "
    "rules, and do I get extra time off afterwards?",
    "We are moving to Munich next spring. Which documents do we need to register our new address, and how long do "
    "we have after the move?",
    "En Allemagne, le contrat de bail peut être résilié par le locataire avec un préavis de trois mois. Le "
    "propriétaire doit justifier d'un intérêt légitime pour mettre fin au contrat.",
    "Bonjour à tous, est-ce que quelqu'un sait comment contester une amende de stationnement reçue pendant les "
    "vacances en Bavière ? Le délai me semble très court.",
    "Czy ktoś wie, jak długo trwa okres wypowiedzenia umowy o pracę w Niemczech? Pracuję tutaj od dwóch lat i "
    "chciałbym zmienić pracodawcę.",
    "Dzień dobry, otrzymałem decyzję podatkową z urzędu skarbowego i nie rozumiem, dlaczego muszę dopłacić podatek "
    "za zeszły rok.",
    "Merhaba, Almanya'da kira sözleşmesini feshetmek için ne kadar önceden haber vermem gerekiyor? Ev sahibim "
    "depozitoyu geri vermek istemiyor.",
    "Buongiorno, qualcuno sa se in Germania il datore di lavoro può chiedere straordinari senza preavviso? Il mio "
    "contratto non dice nulla in proposito.",
    "Goedemiddag, weet iemand hoe lang je in Duitsland de tijd hebt om bezwaar te maken tegen een belastingaanslag? "
    "Ik woon sinds kort in Aken.",
]


_GERMAN = [
    "Hallo zusammen, mein Vermieter hat mir zum Monatsende gekündigt, ohne einen Grund zu nennen. Reicht das nach "
    "§ 573 BGB überhaupt aus, und welche Frist gilt für ihn nach § 573c BGB? Die Normen habe ich hier gelesen:\n"
    "https://www.gesetze-im-internet.de/bgb/__573.html\nhttps://www.gesetze-im-internet.de/bgb/__573c.html",
    "Der Vermieter braucht für die Kündigung ein berechtigtes Interesse, zum Beispiel Eigenbedarf, und muss die "
    "Gründe im Kündigungsschreiben angeben. Die Frist beträgt drei Monate und verlängert sich nach fünf und nach "
    "acht Jahren Mietdauer um jeweils drei Monate. Eine Kündigung ohne Begründung ist unwirksam.",
    "Hat jemand Erfahrung mit einem Einspruch gegen den Steuerbescheid? Das Finanzamt hat meine Fahrtkosten nicht "
    "anerkannt. Nach § 355 AO habe ich dafür einen Monat Zeit. Reicht ein formloses Schreiben, oder muss der "
    "Einspruch über ELSTER kommen? https://www.gesetze-im-internet.de/ao_1977/__355.html",
]


def build_raw_corpus(seed: int = SEED) -> pd.DataFrame:
    """Roher Korpus mit den Spalten doc_id, quelle, url, abruf, gesetz, rechtsgebiet, enbez, titel, text."""
    rng = random.Random(seed)
    excerpt = _excerpt()
    head = excerpt["gesetze"]
    slug_of = {k["jurabk"]: s for s, k in head.items()}
    norms = corpus.load()
    rows, counter = [], {}

    def new_doc(prefix, source, url, fetched_at, text, law_slug=None, area_of_law=None, designation=None, title=None):
        counter[prefix] = counter.get(prefix, 0) + 1
        rows.append({"doc_id": f"{prefix}-{counter[prefix]:05d}", "quelle": source,
                       "url": url, "abruf": fetched_at, "gesetz": law_slug, "rechtsgebiet": area_of_law,
                       "enbez": designation, "titel": title, "text": text})

    xml_url = "https://www.gesetze-im-internet.de/{}/xml.zip"
    for r in norms.itertuples():
        new_doc("xml", "gii-xml", xml_url.format(slug_of[r.gesetz]), f"{FETCH_DATE}T09:00:00Z", r.text,
            r.gesetz, r.rechtsgebiet, r.enbez, r.titel)
    area = dict(norms[["gesetz", "rechtsgebiet"]].drop_duplicates().values)
    for p in excerpt["platzhalter"]:
        law_abbr = head[p["slug"]]["jurabk"]
        # Ein einfacher Harvester nimmt den Inhalt der Norm, bei leerem Inhalt die Überschrift
        text = p["text"] if p["text"] not in ("", "-") else (p["titel"] or p["text"] or "-")
        new_doc("xml", "gii-xml", xml_url.format(p["slug"]), f"{FETCH_DATE}T09:00:00Z", text,
            law_abbr, area[law_abbr], p["enbez"], p["titel"])

    rerun = rng.sample(range(len(norms)), N_RERUN)
    for k, i in enumerate(rerun):
        r = norms.iloc[i]
        if k % 10 < 4:
            text = r.text
        elif k % 10 < 7:
            text = _whitespace_variant(r.text, rng)
        else:
            text = unicodedata.normalize("NFD", r.text)
        new_doc("xml2", "gii-xml-nachlauf", xml_url.format(slug_of[r.gesetz]), f"{FETCH_DATE}T14:00:00Z", text,
            r.gesetz, r.rechtsgebiet, r.enbez, r.titel)

    with_content = [i for i, t in enumerate(norms["text"]) if not t.startswith("(") or "\n" in t]
    for i in rng.sample(with_content, N_HTML):          # HTML-Ernte nur von Normen mit Inhalt
        r = norms.iloc[i]
        slug = slug_of[r.gesetz]
        new_doc("html", "gii-html", _url_html(slug, r.enbez), f"{FETCH_DATE}T11:00:00Z",
            _html_text(head[slug], r.enbez, r.titel, r.text), r.gesetz, r.rechtsgebiet, r.enbez, r.titel)

    global _PII
    _PII = {}
    for k in range(60):
        p = _person(rng)
        if k < 30:
            text, pii = _notice(["steuer", "wohngeld", "bussgeld"][k % 3], p, rng)
        else:
            text, pii = _email(p, rng)
        new_doc("intern", "intern-synthetisch", None, None, text)
        _PII[rows[-1]["doc_id"]] = pii

    forum = [(t, "werbung") for t in _ADS] + [(t, "fremdsprache") for t in _FOREIGN] + [(t, "beitrag") for t in _GERMAN]
    for k, (text, _) in enumerate(forum):
        new_doc("forum", "web-forum", f"https://forum.example.org/thread/{4711 + k}", f"{FETCH_DATE}T16:00:00Z", text)
    return pd.DataFrame(rows)


_PII: dict[str, dict] = {}


def pii_annotations(raw: pd.DataFrame | None = None) -> dict[str, dict[str, list[str]]]:
    """Annotierte PII der internen Texte: doc_id -> {"NAME": [...], "EMAIL": [...], "TELEFON": [...], "IBAN": [...]}."""
    if not _PII:
        build_raw_corpus()
    if raw is None:
        return dict(_PII)
    return {k: v for k, v in _PII.items() if k in set(raw["doc_id"])}
