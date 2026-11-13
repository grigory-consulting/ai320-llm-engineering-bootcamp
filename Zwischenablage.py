diff --git a/shop/rabatt.py b/shop/rabatt.py
index 3f1c2a9..8be47d1 100644
--- a/shop/rabatt.py
+++ b/shop/rabatt.py
@@ -1,12 +1,37 @@
 """Preisberechnung für den Webshop."""
 
 from decimal import Decimal
+
+# Rabattstaffel: ab Bestellwert (inklusive) gilt der Prozentsatz.
+# 0 bis 99,99 EUR: 0 %, 100 bis 499,99 EUR: 5 %, ab 500 EUR: 10 %
+STAFFEL = [
+    (Decimal("500"), Decimal("0.10")),
+    (Decimal("100"), Decimal("0.05")),
+]
+
+
+def rabattsatz(bestellwert: Decimal) -> Decimal:
+    """Liefert den Rabattsatz für einen Bestellwert."""
+    if bestellwert < 0:
+        raise ValueError("Bestellwert darf nicht negativ sein")
+    for grenze, satz in STAFFEL:
+        if bestellwert > grenze:
+            return satz
+    return Decimal("0")
+
+
+def rabattierter_preis(bestellwert: Decimal) -> Decimal:
+    """Bestellwert abzüglich Staffelrabatt, auf Cent gerundet."""
+    preis = bestellwert * (Decimal("1") - rabattsatz(bestellwert))
+    return preis.quantize(Decimal("0.01"))
+
 
 def brutto(netto: Decimal, mwst: Decimal = Decimal("0.19")) -> Decimal:
     """Nettopreis plus Mehrwertsteuer."""
     return (netto * (Decimal("1") + mwst)).quantize(Decimal("0.01"))
 
 
-def endpreis(netto: Decimal) -> Decimal:
-    return brutto(netto)
+def endpreis(netto: Decimal) -> Decimal:
+    """Endpreis: erst Staffelrabatt auf den Nettowert, dann Mehrwertsteuer."""
+    return brutto(rabattierter_preis(netto))




class Zusammenfassung(BaseModel):
    title: str
    changes: list[str]
    risks: list[str]
    need_tests: bool 

def make_tasks(analyst, writer, **task2_kwargs): 
    task1 = Task(
        description=(
            "Analyse this unified diff:\n\n{diff}\n\n"
            "List every functional change and every risk. Check boundary conditions "
            "against the comments in the code and check whether tests were added."
        ),
        expected_output=(
            "Two bullet lists titled 'Changes' and 'Risks'. "
            "Each bullet is one sentence of at most 20 words and names the affected function."
        ),
        agent=analyst,
    )
    task2 = Task(
        description=(
            "Write a changelog entry from the analysis you received as context. "
            "Write title, changes and risks in German. Each item in changes and risks is "
            "one sentence of at most 120 characters. "
            "Set needs_tests to true if the analysis reports missing tests."
        ),
        expected_output=(
            "A Zusammenfassung: title (one line), changes (list of short sentences), "
            "risks (list of short sentences), needs_tests (bool)."
        ),
        agent=writer,
        context=[task1],
        output_pydantic=Zusammenfassung,
        **task2_kwargs,
    )

    return task1, task2

task1, task2 = make_tasks(analyst, writer)

crew = Crew(agents=[analyst, writer], tasks=[task1,task2], process=Process.sequential, verbose=True)
diff_text = Path("diff.txt").read_text(encoding="utf-8")
result = await crew.kickoff_async(inputs={"diff": diff_text})
summary = result.pydantic


