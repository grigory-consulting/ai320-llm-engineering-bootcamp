# AI320: Einrichtung der Kursrechner

Kurs AI320 LLM Engineering Bootcamp, qSkills Nürnberg, 09.–13.11.2026. Trainer: Dr.-Ing. Grigory Devadze. Die Einrichtung übernimmt ein Skript aus dem Kurs-Repository auf GitHub.

## Voraussetzungen

- Debian 13 (x86_64) auf den vorhandenen Kursrechnern (i7-6700, 32 GB RAM). Eine GPU ist nicht nötig.
- Ein Kursbenutzer je Rechner, mit dem die Teilnehmenden arbeiten. Für die Einrichtung braucht er sudo, im Kurs nicht mehr.
- Mindestens 15 GB frei im Home-Verzeichnis des Kursbenutzers. Die Einrichtung belegt rund 5 GB, die Übungen schreiben im Kurs weitere Dateien.
- Diese Pakete:

  ```bash
  sudo apt-get install -y git curl ca-certificates tar libgomp1
  ```

## Einrichtung je Rechner

1. Als Kursbenutzer, nicht als root, das Repository klonen und das Skript aufrufen:

   ```bash
   git clone https://github.com/grigory-consulting/ai320-llm-engineering-bootcamp.git ~/ai320
   cd ~/ai320
   bash scripts/setup_debian13.sh
   ```

   Das Skript legt Python 3.11 und die Python-Pakete in `~/ai320` an, installiert llama.cpp (Build b9290) nach `/opt/llama-b9290` mit Links in `/usr/local/bin` und lädt Textkorpus und Modelle (rund 2,5 GB). Ein zweiter Aufruf schadet nicht. Den Ordner danach nicht mehr verschieben oder umbenennen.
2. Am Ende läuft ein Setup-Check und zeigt eine Liste mit GRÜN, GELB und ROT. Erwartet ist alles GRÜN bis auf die Zeile `labs/.env`: Diese Datei mit dem API-Schlüssel bringt der Trainer im Kurs mit. Steht eine andere Zeile auf ROT, nennt sie die Ursache und den Befehl, der sie behebt. Hilft das nicht, melden Sie sich beim Trainer.

Den Check können Sie jederzeit wiederholen:

```bash
cd ~/ai320/labs
../.venv/bin/python _src/lab0_setup_check.py
```

## Mehrere Rechner

Die Modelle müssen nicht auf jedem Rechner aus dem Netz kommen. Auf den übrigen Rechnern klonen Sie wie oben, kopieren `~/ai320/labs/models/` vom ersten Rechner an dieselbe Stelle und rufen das Skript so auf:

```bash
bash scripts/setup_debian13.sh --ohne-modelle
```

Ein Abbild des fertigen ersten Rechners geht ebenfalls, solange Benutzername und Pfad `~/ai320` gleich bleiben.

## Im Kurs

- Jupyter starten: `cd ~/ai320/labs && ../.venv/bin/jupyter lab`
- Mehrere Übungen lasten alle CPU-Kerne aus. Das ist gewollt.
