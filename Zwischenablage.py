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
