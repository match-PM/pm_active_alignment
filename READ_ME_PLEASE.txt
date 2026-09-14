ACTIVE-ALIGNMENT-SIMULATION
Betriebs-, Installations- und Auswertungsanleitung
=================================================

Stand und Zweck
---------------

Diese Anleitung beschreibt die Vorbereitung, Durchführung und Auswertung
automatisierter Active-Alignment-Versuche in der ROS-2-Simulationsumgebung.
Sie fasst die wichtigsten Abhängigkeiten, Konfigurationsschritte,
Sicherheitsmechanismen und Annahmen der Datensortierung zusammen.

Die Anleitung ersetzt nicht die Kommentare im Quellcode. Insbesondere
Master_Test_Makyr.py enthält weitere dokumentierte Sonder-, Debug- und
Konfigurationsmöglichkeiten. Werden andere als die hier beschriebenen
Standardfunktionen benötigt, sollte der betreffende Codeabschnitt vor der
Ausführung vollständig gelesen werden.


1. Verwendete Hauptkomponenten
------------------------------

Master_Test_Makyr.py
    Zentrale Vorbereitung eines Versuchsmodus. Das Skript berechnet die
    Suchraumgrenzen, bestimmt die aktiven Gelenke, erzeugt reproduzierbare
    Problem-Seeds und überschreibt die zugehörigen JSON-Dateien.

batch_config.json
    Legt fest, welche Seed-Gruppen und Algorithmen ausgeführt werden, wie
    viele vollständige Wiederholungsläufe stattfinden und welche
    Algorithmusparameter verwendet werden.

active_alignment_skill_node.py
    Stellt die Optimierungsverfahren als ROS-2-Action-Server bereit und
    enthält die Batch-Ausführung, Messwerterfassung und Ergebnisspeicherung.

dummy_value_publisher.py
    Berechnet und veröffentlicht das simulierte Rückkopplungssignal. Hier
    wird auch der verwendete Rauschmodus festgelegt.

bringupfirst.py
    Startet die für die Versuche benötigte Simulationsumgebung. Falls die
    lokale Datei tatsächlich anders benannt ist, muss der lokale Dateiname
    verwendet werden.

Librarian.py
    Liest Ergebnisordner ein, erkennt Versuchsmerkmale, entfernt exakte
    Dubletten aus der Sortierung, sondert bekannte fehlerhafte oder nicht
    eindeutig klassifizierbare alte Nelder-Mead-Daten aus und kopiert gültige
    Läufe in eine einheitliche Bibliotheksstruktur.

graph_configurator.py
    Lädt und filtert die sortierten Läufe, führt die budgetbezogene
    Analysebereinigung aus und erzeugt die Vergleichsgrafiken sowie die
    zugehörigen TXT-Ausgaben. Die Rohdaten werden dabei nicht verändert.


2. System- und Installationsvoraussetzungen
-------------------------------------------

Vorausgesetzt werden die bereits eingerichtete ROS-2-Humble-Umgebung, der
Colcon-Workspace und sämtliche ROS-Pakete der Simulationsumgebung. Der hier
verwendete Workspace wird beispielhaft unter folgendem Pfad angenommen:

    /home/<BENUTZERNAME>/ros2_humble

Für die Python-Komponenten werden unter anderem NumPy, SciPy und objgraph
benötigt. Librarian und Graph Configurator benötigen zusätzlich pandas,
Matplotlib und PyQt6. Py-BOBYQA benötigt das externe Paket Py-BOBYQA.

Vor der Nutzung von Py-BOBYQA einmalig ausführen:

    /usr/bin/python3 -m pip install --user Py-BOBYQA==1.5.0

Installation kontrollieren:

    /usr/bin/python3 -c "import pybobyqa; print(pybobyqa.__version__)"

Die wichtigsten Python-Imports können gemeinsam geprüft werden:

    /usr/bin/python3 -c "import numpy, scipy, objgraph, pybobyqa"
    /usr/bin/python3 -c "import pandas, matplotlib, PyQt6"

Falls ein Import fehlt, sollte nur das betreffende Paket nachinstalliert
werden. Für Python-Pakete darf nicht unkontrolliert sudo pip verwendet
werden. Die verwendete Py-BOBYQA-Version sollte für die gesamte
Versuchsreihe unverändert bleiben.


3. Versuchsmodus in Master_Test_Makyr.py
----------------------------------------

Der Hauptmodus wird über folgenden Bereich gewählt:

    # ---- Experiment mode (overrides manual joint toggles) ----
    # Options: "5D", "2D_TRANS", "2D_ROT", "3D_TRANS", "3D_ROT"
    MODE = "2D_ROT"

Regulär verwendete Modi:

    "2D_TRANS"   X und Y
    "2D_ROT"     A und B
    "3D_TRANS"   X, Y und Z
    "5D"         X, Y, Z, A und B

WICHTIG:

In der geprüften Fassung darf MODE nicht auf None gesetzt werden, obwohl der
Kommentar dies als manuelle Alternative bezeichnet. get_enabled_joints()
behandelt None nicht und löst einen ValueError aus. Der Modus "3D_ROT"
aktiviert ebenfalls nur A und B, da die C-Achse dauerhaft ausgeschlossen ist.
Er entspricht daher funktional "2D_ROT" und darf nicht als echte
dreidimensionale Rotationskonfiguration ausgewertet werden.

Eine Änderung der Modellparameter

    W0
    LAMBDA
    N_INDEX
    SEARCH_FACTOR

verändert das Kopplungsmodell beziehungsweise die Suchraumgrenzen. Solche
Änderungen dürfen nicht unmarkiert mit bestehenden Datensätzen
zusammengefasst werden.


4. Erzeugung der Startpositionen
--------------------------------

Die Seed-Erzeugung wird über folgenden Bereich gesteuert:

    GENERATE_SEED_POSES = True

    SEED_POWER_LEVELS = [2.0, 20.0, 60.0]
    SEEDS_PER_LEVEL = 3
    SEED_BASE = 31248769

SEED_POWER_LEVELS
    Definiert die gewünschten Gruppen der anfänglichen Kopplung.

SEEDS_PER_LEVEL
    Legt die Anzahl unterschiedlicher geometrischer Ausgangspositionen je
    Startsignalgruppe fest.

SEED_BASE
    Bestimmt die reproduzierbare Erzeugung dieser Ausgangspositionen.

GENERATE_SEED_POSES = True überschreibt beim Ausführen von
Master_Test_Makyr.py die Datei seed_poses.json. Eine benötigte bestehende
Seed-Datei muss daher zuvor gesichert werden. Soll ein bereits erzeugter
Seed-Katalog unverändert weiterverwendet werden, ist die erneute Erzeugung
zu deaktivieren oder das Master-Skript nicht erneut auszuführen.

Nach einer Änderung des Freiheitsgradmodus müssen die Seeds neu erzeugt
werden. Eine Seed-Pose für 2D_ROT beschreibt nicht dieselbe geometrische
Aufgabe wie ein Seed mit derselben Zielkopplung in 5D.

Die Werte in batch_config.json unter seed_groups müssen mit den tatsächlich
erzeugten Leistungsgruppen und der vorhandenen Seed-Anzahl übereinstimmen.
SEEDS_PER_LEVEL = 3 erlaubt beispielsweise höchstens drei vorbereitete Seeds
je Leistungsgruppe.

Der Problem-Seed legt die geometrische Ausgangssituation fest. Er soll nicht
erzwingen, dass ein stochastisches Verfahren in allen Wiederholungsläufen
dieselbe Kandidatenfolge erzeugt. Die Wiederholungen untersuchen das
statistische Verhalten bei derselben Ausgangsgeometrie.


5. Wirkung von Master_Test_Makyr.py
-----------------------------------

Beim Ausführen des Master-Skripts werden insbesondere folgende Dateien
erzeugt oder überschrieben:

    spawn_test_frames.json
    alignment_config.json
    test_active_alignment.rsap.json
    seed_poses.json, falls GENERATE_SEED_POSES = True

Das Skript überträgt den gewählten MODE in die Liste der aktiven Gelenke,
entfernt nicht verwendete Gelenke aus der Action-Server-Beschreibung und
setzt die aus dem Kopplungsmodell berechneten translatorischen und
rotatorischen Grenzen. Die C-Achse wird entfernt.

Empfohlener Aufruf aus dem Verzeichnis des Skripts:

    python3 Master_Test_Makyr.py

Die Konsolenausgabe muss anschließend kontrolliert werden. Insbesondere
sollten MODE, aktive Translations- und Rotationsachsen, Suchraumgrenzen,
Leistungsgruppen und erzeugte Seed-Anzahl den geplanten Versuchen
entsprechen.


6. Konfiguration der Batch-Ausführung
-------------------------------------

batch_config.json enthält zwei voneinander zu unterscheidende Anzahlen:

iterations
    Anzahl vollständiger Wiederholungsläufe eines Algorithmus je Problem.
    iterations = 0 deaktiviert den betreffenden Algorithmus.

max_iterations
    Maximale Zahl der Optimierungsbewertungen innerhalb eines Laufs. Die
    Initialmessung besitzt den Analyseindex 0 und zählt nicht zu diesem
    Budget.

Vor dem Start ist für jeden Task zu kontrollieren:

    - exakte Algorithmusbezeichnung;
    - gewünschte Anzahl vollständiger Läufe;
    - Bewertungsbudget;
    - Signalschwellwert;
    - Schrittweite beziehungsweise Trust-Region-Parameter;
    - Bewegungs- und Wartezeiten.

Für Py-BOBYQA wird bei npt = null automatisch die vollständige quadratische
Interpolation verwendet. Dies entspricht sechs Modellpunkten in 2D und
21 Modellpunkten in 5D. Bei max_iterations = 80 enthält ein vollständiger
Lauf höchstens eine Initialmessung und 80 weitere Messungen.

Vor einer großen Versuchsreihe sollten alle nicht benötigten Tasks durch
iterations = 0 deaktiviert werden. Andernfalls werden sie zusätzlich
ausgeführt.


7. Wahl des Rauschmodus
-----------------------

Der Rauschmodus wird im Dummy-Value-Publisher über den dort vorgesehenen
String festgelegt. Zulässige Werte sind:

    "none"
    "low"
    "high"

Die Schreibweise sollte exakt und kleingeschrieben übernommen werden. Nach
einer Änderung des Rauschmodus muss die geänderte Datei gespeichert, das
betroffene ROS-2-Paket neu gebaut und die laufende Node neu gestartet werden.
Ein bereits laufender Publisher übernimmt eine reine Quellcodeänderung nicht.

Für jeden Rauschmodus sollte eine eigene Simulationssitzung beziehungsweise
ein eigener Session-Ordner erzeugt werden. Vor dem Batch-Start muss anhand
der Publisher-Ausgabe oder eines kurzen Testlaufs kontrolliert werden, dass
der beabsichtigte Modus tatsächlich aktiv ist.


8. Wichtige Annahme des Librarian zur Rauscherkennung
-----------------------------------------------------

Der Rauschmodus wird nicht zuverlässig aus dem Namen eines einzelnen
Run-Ordners abgelesen. Der Librarian fasst die Runs nach ihrem unmittelbaren
Quellordner zusammen und leitet für diese Gruppe einen gemeinsamen
Rauschmodus aus der Streuung der Anfangssignalabweichungen ab. Anschließend
wird dieser erkannte Modus allen Runs dieses Quellordners zugeordnet.

Daraus folgt die zentrale Sortierannahme:

    ALLE RUN-ORDNER MIT DEMSELBEN UNMITTELBAREN ELTERNORDNER MÜSSEN MIT
    DEMSELBEN RAUSCHMODELL ERZEUGT WORDEN SEIN.

Run-Ordner unterschiedlicher Algorithmen und unterschiedlich benannte Runs,
beispielsweise Hill_Climb_1_... und Hill_Climb_2_..., dürfen gemeinsam in
einem Session-Ordner liegen, wenn sie alle unter derselben Rauschbedingung
erzeugt wurden. Läufe aus "none", "low" und "high" dürfen dagegen nicht in
denselben unmittelbaren Session-Ordner gemischt werden.

Ein übergeordneter Scan-Ordner darf mehrere getrennte Session-Ordner
enthalten, beispielsweise:

    Simulationsdaten/
        Session_2D_ROT_NONE/
            Hill_Climb_1_...
            Py-BOBYQA_1_...
        Session_2D_ROT_LOW/
            Hill_Climb_1_...
            Py-BOBYQA_1_...
        Session_2D_ROT_HIGH/
            Hill_Climb_1_...
            Py-BOBYQA_1_...

Diese Struktur ist gültig, weil jeder unmittelbare Elternordner intern nur
einen Rauschmodus enthält. Werden verschiedene Rauschmodelle innerhalb
desselben Session-Ordners gemischt, kann der Sorter den gesamten Ordner
falsch klassifizieren und anschließend ungeeignete 3-Sigma-Grenzen anwenden.


9. Speichern und Bauen des ROS-2-Pakets
---------------------------------------

Vor jedem Build müssen alle Änderungen in Visual Studio Code gespeichert
werden. Dazu entweder Strg+S verwenden oder kontrollieren, dass kein Punkt
beziehungsweise keine Kennzeichnung für ungespeicherte Änderungen am
Dateireiter vorhanden ist. Colcon baut den gespeicherten Dateistand und
nicht den noch offenen, ungespeicherten Editorinhalt.

Danach in einem Terminal ausführen:

    cd /home/<BENUTZERNAME>/ros2_humble
    source /opt/ros/humble/setup.bash
    colcon build --packages-select active_alignment_skills
    source /home/<BENUTZERNAME>/ros2_humble/install/setup.bash

Für den Rechner brotato211-X670-AORUS-ELITE-AX lautet der Workspace-Aufruf
entsprechend:

    cd /home/brotato211/ros2_humble
    colcon build --packages-select active_alignment_skills

Nach jedem erfolgreichen Build muss der Workspace in jedem neu geöffneten
Terminal erneut gesourct werden:

    source /opt/ros/humble/setup.bash
    source /home/<BENUTZERNAME>/ros2_humble/install/setup.bash

Alte, noch laufende Nodes müssen vor dem Test beendet und neu gestartet
werden. Andernfalls kann trotz erfolgreichem Build weiterhin der alte
Python-Prozess ausgeführt werden.


10. Empfohlene Reihenfolge vor einem Batch
------------------------------------------

1. Gewünschten MODE in Master_Test_Makyr.py einstellen.
2. Seed-Leistungsgruppen, Seed-Anzahl und SEED_BASE kontrollieren.
3. Entscheiden, ob seed_poses.json neu erzeugt werden darf.
4. Master_Test_Makyr.py mit Strg+S speichern und anschließend ausführen.
5. Erzeugte Konfigurationsdateien und Konsolenausgaben kontrollieren.
6. seed_groups, tasks, iterations und Algorithmusparameter in
   batch_config.json überprüfen.
7. Gewünschten Rauschmodus none, low oder high im Dummy-Value-Publisher
   einstellen.
8. Sämtliche bearbeiteten Python- und JSON-Dateien mit Strg+S speichern.
9. Das Paket active_alignment_skills mit Colcon neu bauen.
10. Den Workspace anschließend neu sourcen.
11. Die gesamte Simulationsumgebung für den neuen Versuchsblock frisch
    starten.
12. Zunächst einen kurzen Probelauf mit wenigen Wiederholungen durchführen.
13. Den Probelauf anhand von history.csv, metadata.json und den
    Terminalmeldungen überprüfen.
14. Erst nach erfolgreicher Prüfung die vollständige Batch-Anzahl
    aktivieren.
   
15. Aufgrund von Erfahrung ist es Sinnvoll nicht mehr als 6000 runs auf einmal 
zu Simulieren da ich keine Stabilität gewährleisten kann (Untested Territory)

GANZ WICHTIG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Bei jeder Änderung des Freiheitsgradmodus, beispielsweise von 2D_TRANS
auf 2D_ROT oder 5D, muss diese Reihenfolge einschließlich Build, Sourcing
und vollständigem Neustart der Simulationsumgebung erneut durchgeführt
werden. Master_Test_Makyr.py passt die verwendeten Konfigurationsdateien
und die für den gewählten Modus benötigten Gelenk-Subscriber an. Die
Subscriber eines bereits laufenden Active-Alignment-Nodes können jedoch
nicht zuverlässig während der laufenden Sitzung auf einen anderen
Freiheitsgradmodus umgestellt werden. Eine bloße Änderung von MODE oder
ein erneuter Start des Batch-Optimizers innerhalb derselben Sitzung reicht
daher nicht aus. Ohne Neustart könnten weiterhin die Subscriber und
Gelenkeinstellungen des vorherigen Modus aktiv sein, wodurch die
aufgezeichneten Läufe nicht eindeutig der gewünschten
Freiheitsgradkonfiguration entsprechen würden.

GANZ WICHTIG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

-----------------------------------------------

Wenn ein Batch beendet ist und neue Koordinaten, Freiheitsgrade,
Seed-Gruppen, Rauschbedingungen oder wesentliche Algorithmusparameter
untersucht werden sollen, wird ein vollständiger Neustart der
Simulationsumgebung über bringupfirst.py empfohlen.

Es ist nicht nachgewiesen, dass die Umgebung nach exakt 3000 Läufen instabil
wird. Aufgrund früherer Probleme, der beobachteten Laufzeitentwicklung und
der Vielzahl bereits korrigierter Zustands- und Resetprobleme wird jedoch
nicht vorausgesetzt, dass beliebig große Versuchsreihen innerhalb einer
einzigen Sitzung zuverlässig bleiben. Der Neustart ist daher eine
vorsorgliche Maßnahme und keine empirisch bestimmte harte Laufgrenze.

Praktische Empfehlung:

    - Versuchsblöcke möglichst auf deutlich unter 3000 Läufe je frischer
      Sitzung begrenzen;
    - spätestens bei einem Wechsel von MODE, Rauschbedingung oder
      Seed-Katalog vollständig neu starten;
    - bei wachsender Laufzeit, fehlenden Signalen, Trajektorienfehlern oder
      ungewöhnlichen Wiederholungen den Batch stoppen und die Umgebung neu
      starten.


12. Vorhandene Schutzmechanismen während der Durchführung
---------------------------------------------------------

Die aktuelle Versuchsumgebung enthält mehrere Schutz- und
Plausibilitätsmechanismen:

    - Rücksetzen der aktiven Gelenke auf die normierte Position 0,5 vor
      jedem Optimierungslauf;
    - erneutes Rücksetzen nach jedem vollständigen Lauf;
    - erneutes Laden der Zielrahmen für jeden Problem-Seed;
    - Vorprüfung des empfangenen Anfangssignals gegenüber dem aus dem Seed
      berechneten Erwartungswert;
    - Begrenzung erzeugter Kandidaten auf den normierten Bereich [0,1];
    - verfahrensspezifische Budgetprüfungen;
    - unmittelbarer Abbruch bei Erreichen des Signalschwellwerts;
    - gemeinsame Speicherung von Messwert und zugehöriger vorgegebener
      Gelenkkonfiguration in derselben CSV-Zeile;
    - Speicherung der Initialmessung unter dem Analyseindex 0;
    - getrennte Erfassung von Initialmessung und Optimierungsbewertungen.

Die Schutzmechanismen reduzieren das Risiko fehlerhafter Läufe, ersetzen
aber nicht die Kontrolle der Terminalausgabe und der erzeugten Dateien.


13. Bedeutung der Py-BOBYQA-Historie
------------------------------------

Für Py-BOBYQA gilt:

    CSV-Zeile 0
        Tatsächlich ausgelesene Initialmessung an der Startposition.

    CSV-Zeilen 1 bis n
        Jede weitere physisch ausgeführte Signalmessung. Es werden nicht nur
        Verbesserungen gespeichert. Auch abgelehnte Kandidaten und erneut
        ausgewertete Positionen bleiben enthalten.

Der erste interne Solver-Aufruf am Startpunkt verwendet die bereits
gespeicherte Initialmessung und erzeugt keine zweite Messung. Bei einem
Budget von 80 Optimierungsbewertungen sind daher höchstens 81 CSV-Zeilen
zulässig. Bei Erfolg oder solverinterner Terminierung können es weniger
sein.

summary.txt enthält für Py-BOBYQA das beste beobachtete Ergebnis. Die letzte
CSV-Zeile ist dagegen die letzte tatsächlich bewertete Kandidatenposition
und muss nicht dem Bestwert entsprechen. Für die vergleichende Auswertung
bleibt die CSV-Historie die maßgebliche Datenquelle.


14. Schutzmechanismen der Datensortierung
-----------------------------------------

Der Librarian und der Graph Configurator enthalten ergänzende
Auswertungsschutzmechanismen:

Parametergruppen
    Jeder Satz von Eingabeparametern erhält unabhängig vom Algorithmus eine
    kurze stabile Kennung der Form CFG_XXXXXXXXXX. Zwei Parametersätze, die
    sich beispielsweise nur durch delta = 2,0 und delta = 1,9 unterscheiden,
    erhalten unterschiedliche Kennungen. Die vollständige Zuordnung bleibt
    in parameter_index.json gespeichert.

    Laufabhängige Ergebniswerte wie tatsächlich verwendete Bewertungen,
    Abbruchgrund, Solverstatus, Solvermeldung und Solverzähler werden bewusst
    nicht zur Bildung einer Parametergruppe verwendet. Sie sind Ergebnisse
    eines Laufs und keine vorgegebenen Algorithmusparameter. Automatisch aus
    dem Bewertungsbudget oder der Zahl aktiver Freiheitsgrade abgeleitete
    Größen werden ebenfalls semantisch vereinheitlicht. Insbesondere erzeugen
    maxfun = max_iterations + 1 und die dimensionsabhängige automatische
    Wahl von npt bei Py-BOBYQA keine künstlichen Varianten. Gleichwertige
    Zahlenangaben wie 80, 80.0 und "80" werden identisch behandelt.

Variantenvergleich in Diagrammen
    Enthalten die ausgewählten Daten mehrere Parametersätze desselben
    Algorithmus, verwendet der Graph Configurator automatisch kompakte Namen
    wie Nelder-Mead [V1] und Nelder-Mead [V2]. Die Achsen und Legenden bleiben
    dadurch kurz. Im Gruppenfilter werden keine internen CFG-Kennungen mehr
    angezeigt, sondern dieselben lesbaren Variantennamen. Die Liste enthält
    ausschließlich Parametersätze der im globalen Auswahlfeld aktivierten
    Algorithmen. Gruppe A und Gruppe B können somit denselben Algorithmus mit
    unterschiedlichen Parametern enthalten.

    Jede zu einem Diagramm gespeicherte TXT-Datei enthält den Abschnitt
    VARIANT DEFINITIONS. Dort werden V1, V2 und weitere Kurzbezeichnungen der
    jeweiligen CFG-Kennung und dem vollständigen Eingabeparametersatz
    zugeordnet. Zusätzlich enthält jede Laufzeile sowohl den Anzeigenamen als
    auch den ursprünglichen Algorithmusnamen, die CFG-Kennung und die roh in
    metadata.json gespeicherten Algorithmusparameter.

Vergleichsgruppen und Erfolgsraten
    Bei A/B- oder A/B/C-Vergleichen werden identische Algorithmusvarianten um
    die jeweilige Gruppe ergänzt, damit spezialisierte Diagramme die Gruppen
    nicht versehentlich zusammenfassen. Der Graph Configurator bietet getrennte
    Erfolgsratendiagramme für das post-initiale Bestsignal und für den letzten
    Wert des budgetbereinigten CSV-Verlaufs. Beide beachten die konfigurierte
    Budgetgrenze; sie beantworten jedoch unterschiedliche Fragestellungen.

Exakte Dubletten
    Der Librarian bildet einen Inhaltsfingerabdruck aus metadata.json,
    history.csv und summary.txt. Identische Kopien desselben Laufs werden
    auch dann übersprungen, wenn sie an mehreren Stellen des Quellordners
    vorkommen. Unabhängige Wiederholungsläufe mit unterschiedlichen Inhalten
    bleiben erhalten.

Alte Nelder-Mead-Daten
    Bekannte fehlerhafte Nelder-Mead-Implementierungen werden anhand der
    vorhandenen Informationen klassifiziert und nicht in den regulären
    Vergleich übernommen. Nicht eindeutig klassifizierbare alte
    Nelder-Mead-Läufe werden ebenfalls nicht automatisch als gültig
    behandelt.

Anfangssignal-Ausreißer
    Bei verrauschten Datensätzen vergleicht der Librarian die erste
    CSV-Messung mit der Zielkopplung des Seeds. Läufe außerhalb der für den
    erkannten Rauschmodus verwendeten Toleranz werden getrennt als Flagged
    einsortiert.

Budgetbereinigung
    Der Graph Configurator verwendet vorrangig das in den Metadaten
    gespeicherte Bewertungsbudget. Überzählige CSV-Einträge werden nur im
    eingelesenen Analyse-DataFrame abgeschnitten. history.csv,
    metadata.json und summary.txt bleiben unverändert. Für alte Datensätze
    ohne lesbares Budget wird nur der bekannte Bereich von 81 bis 84
    Optimierungsbewertungen auf 80 begrenzt. Bewusst größere Budgets bleiben
    erhalten.

Fehlgeschlagene Frühabbrüche
    Ein Lauf, der den Erfolgsschwellwert nicht erreicht und trotzdem vor dem
    Budget endet, wird bei der Iterationsbewertung als vollständige
    Budgetausnutzung behandelt. Dadurch erhält ein erfolgloser vorzeitiger
    Solverabbruch keinen künstlichen Geschwindigkeitsvorteil.


15. Kontrolle nach einem Probelauf
----------------------------------

Jeder gültige Run-Ordner sollte mindestens enthalten:

    history.csv
    metadata.json
    summary.txt

Für einen Py-BOBYQA-Lauf sollte gelten:

    Anzahl CSV-Datenzeilen = total_evaluations
    optimization_evaluations = Anzahl CSV-Datenzeilen - 1

Bei max_iterations = 80 darf ein nicht zuvor beendeter Lauf höchstens
81 CSV-Datenzeilen enthalten. Bei einem erfolgreichen Frühabbruch ist eine
kleinere Anzahl korrekt; die Schwellwertmessung muss noch in der CSV stehen.

Vor einer großen Batch-Ausführung sollte außerdem nach folgenden Meldungen
gesucht werden:

    Py-BOBYQA failed:
    Failed to save Py-BOBYQA results:
    Failed to move controller
    Signal could not be verified
    Unknown algorithm

Einzelne identische aufeinanderfolgende Signalwerte sind allein noch kein
Fehlernachweis. Ein Optimierer kann dieselbe oder eine sehr ähnliche Position
mehrfach auswerten, und gerundete Konsolenausgaben können unterschiedliche
Messwerte identisch darstellen. Häufige Trajektorienfehler oder lange
unveränderte Positionsfolgen müssen dagegen geprüft werden.


16. Organisation der Rohdaten
-----------------------------

Rohdaten sollten unmittelbar nach jedem abgeschlossenen Versuchsblock
gesichert und nicht manuell verändert werden. Empfohlen wird eine eindeutige
Ordnerbezeichnung, welche mindestens Modus, Rauschbedingung und Sitzung
erkennen lässt, beispielsweise:

    Session_2D_ROT_NONE_20260909
    Session_2D_ROT_LOW_20260909
    Session_2D_ROT_HIGH_20260909

Die einzelnen Run-Ordner und die Dateien history.csv, metadata.json und
summary.txt sollten nicht umbenannt oder inhaltlich bearbeitet werden, bevor
der Librarian sie eingelesen hat. Der Librarian kopiert die Daten in die
strukturierte Zielbibliothek; er ersetzt keine unveränderte Sicherung der
ursprünglichen Session-Ordner.

Nach der Sortierung sollte kontrolliert werden:

    - erkannter Rauschmodus je Quellordner;
    - Anzahl gefundener und kopierter Läufe;
    - Anzahl übersprungener Dubletten;
    - Anzahl markierter Anfangssignal-Ausreißer;
    - Anzahl ausgesonderter Nelder-Mead-Läufe;
    - erwartete Kombinationen aus Modus, Seed, Leistung und Algorithmus.


17. Kurzcheck vor dem vollständigen Batch
-----------------------------------------

[ ] Richtiger MODE gewählt
[ ] MODE ist nicht None
[ ] 3D_ROT wird nicht als echter 3D-Rotationsmodus verwendet
[ ] Richtige Leistungsgruppen und Seed-Anzahl eingestellt
[ ] Überschreiben von seed_poses.json bewusst erlaubt
[ ] Master_Test_Makyr.py ausgeführt und Ausgabe kontrolliert
[ ] batch_config.json stimmt mit seed_poses.json überein
[ ] Nur gewünschte Algorithmen besitzen iterations > 0
[ ] Algorithmusparameter und Bewertungsbudget geprüft
[ ] Rauschmodus none, low oder high korrekt eingestellt
[ ] Neuer Session-Ordner für diesen Rauschmodus vorgesehen
[ ] Alle Dateien in Visual Studio Code gespeichert
[ ] active_alignment_skills erfolgreich gebaut
[ ] Workspace neu gesourct
[ ] Simulationsumgebung frisch gestartet
[ ] Probelauf ohne relevante Fehlermeldungen abgeschlossen
[ ] CSV- und Metadatenzählung des Probelaufs plausibel
[ ] Erst danach vollständige Wiederholungszahl aktiviert


18. Zusammenfassung der wichtigsten Regeln
-------------------------------------------

1. Änderungen immer speichern, bevor Colcon ausgeführt wird.
2. Nach Code- oder Rauschänderungen neu bauen und alle betroffenen Nodes neu
   starten.
3. Nach einem Moduswechsel neue, zum Freiheitsgrad passende Seeds erzeugen.
4. Keine unterschiedlichen Rauschmodelle in denselben unmittelbaren
   Session-Ordner schreiben.
5. Neue Versuchsblöcke vorsorglich in einer frisch gestarteten
   Simulationsumgebung ausführen.
6. Zuerst einen Probelauf, danach erst die vollständige Batch-Anzahl starten.
7. Rohdaten unverändert aufbewahren; Bereinigungen nur in der Analyse
   durchführen.
8. Konsolenmeldungen und erwartete Laufanzahl nach jedem Batch kontrollieren.
