# PM Active Alignment

Ein ROS2-basiertes System zur automatischen Ausrichtung von Robotergelenken mittels iterativer Optimierungsalgorithmen. Das Projekt ermöglicht es, Roboterarme durch Variationen von Gelenkpositionen optimal auszurichten und dabei Evaluierungssignale zu minimieren oder zu maximieren.

## 📋 Überblick

Das Projekt besteht aus zwei ROS2-Paketen:

1. **active_alignment_interfaces**: Definiert die Schnittstellen (Actions, Messages) für die Kommunikation
2. **active_alignment_skills**: Implementiert den Optimierungsserver und die Algorithmen

### Hauptmerkmale

- **Mehrere Optimierungsalgorithmen**: Hill-Climbing und Nelder-Mead
- **Multi-Controller-Support**: Arbeitet mit mehreren Joint-Trajectory-Controllern
- **Echtzeitausführung**: Asynchrone Architektur für ROS2
- **Flexible Optimierungsziele**: Minimierung oder Maximierung von beliebigen Topics
- **Adaptive Step-Sizes**: Automatische Anpassung der Schrittweiten

---

## 📦 Paketstruktur

```
pm_active_alignment/
├── active_alignment_interfaces/          # Interface-Definitionen
│   ├── action/
│   │   ├── ActiveAlign.action           # Haupt-Action für Ausrichtungsoperation
│   │   └── ActiveAlignment.action       # Alternative Action
│   ├── msg/
│   │   ├── AlignTopic.msg               # Optimierungsziel-Definition
│   │   ├── JointAlign.msg               # Gelenk-Spezifikation
│   │   └── JointAlignmentResult.msg     # Optimierungsergebnis
│   ├── CMakeLists.txt
│   └── package.xml
│
└── active_alignment_skills/               # Implementierung
    ├── active_alignment_skills/
    │   ├── active_alignment_skill_node.py  # Hauptserver-Node
    │   ├── dummy_value_publisher.py        # Test-Publisher
    │   └── py_modules/
    │       ├── AlignmentControllerManager.py   # Controller/Gelenk-Verwaltung
    │       ├── JointLimitFetcher.py           # Gelenklimit-Abfrage
    │       └── OptimizationValueManager.py     # Evaluierungswert-Verwaltung
    ├── doc/
    │   ├── spawn_test_frames.json
    │   └── test_active_alignment.rsap.json
    ├── test/
    │   ├── test_copyright.py
    │   ├── test_flake8.py
    │   └── test_pep257.py
    ├── setup.py
    ├── setup.cfg
    ├── package.xml
    └── LICENSE
```

---

## 🔧 Installation

### Voraussetzungen

- **ROS2** (getestet auf Humble/Iron)
- **Python 3.8+**
- **Abhängigkeiten**:
  ```bash
  pip install numpy scipy matplotlib urdf_parser_py
  ```

### Aufbau

```bash
# Im ROS2 Workspace
cd ~/ros2_ws

# Dependencies installieren
rosdep install --from-paths src --ignore-src -r -y

# Workspace bauen
colcon build --packages-select active_alignment_interfaces active_alignment_skills
```

---

## 📚 Schnittstellen-Definitionen

### Action: `ActiveAlign`

**Goal:**
```
JointAlign[] active_joints          # Zu optimierende Gelenke
AlignTopic[] alignment_topics        # Evaluierungssignale
---
Result
---
bool active                          # Feedback: Lauf aktiv?
```

**Result:**
```
bool success                         # Erfolgreich optimiert?
int32 num_of_iterations              # Anzahl Iterationen
float32 final_eval_value             # Finaler Evaluierungswert
JointAlignmentResult[] joint_results  # Optimale Gelenkpositionen
```

### Message: `AlignTopic`

Definiert ein Optimierungsziel (z. B. ein ROS-Topic als Feedback):

```
string topic_name                   # Name des zu überwachenden Topics
bool minimize_not_maximize = true   # true: minimieren, false: maximieren
float64 weight_factor = 1.0         # Gewichtung für Mehrfach-Ziele
```

### Message: `JointAlign`

Spezifizierung eines Gelenks für die Optimierung:

```
string joint_name                   # Name des Gelenks (z. B. "joint_1")
bool has_constraint                 # Hat das Gelenk Limits?
float64 upper_limit                 # Oberes Limit (rad)
float64 lower_limit                 # Unteres Limit (rad)
```

### Message: `JointAlignmentResult`

Ergebnis für ein einzelnes Gelenk:

```
string joint_name
float64 joint_value                 # Optimierte Position
```

---

## 🚀 Verwendung

### Server starten

```bash
ros2 run active_alignment_skills active_alignment_skill_node
```

Der Server startet zwei Action-Server:
- `/active_alignment_server/exec_active_alignment_hill_climb`
- `/active_alignment_server/exec_active_alignment_nelder_mead`


---

## 🧠 Optimierungsalgorithmen

### Hill-Climbing

**Beschreibung:**
- Einfacher, stochastischer Algorithmus
- Zufällige Perturbation der Gelenkpositionen im normalisierten Raum [0, 1]
- Akzeptiert nur Schritte, die den Evaluierungswert verbessern

**Parameter:**
```python
max_iterations: int = 20           # Maximale Iterationen
step_size: float = 0.1             # Schrittweite (0-1, normalisiert)
adaptive_step_size: bool = False   # Adaptive Anpassung aktivieren
movement_time: float = 0.1         # Bewegungszeit zwischen Schritten (s)
```

**Algorithmus:**
1. Aktuelle Gelenkpositionen auslesen
2. Evaluierungswert messen
3. Zufällige neue Gelenkpositionen generieren
4. Wenn Evaluierungswert besser: Diese akzeptieren
5. Sonst: Zu den besten Positionen zurückkehren
6. Repeat bis max_iterations oder Konvergenz

### Nelder-Mead

**Beschreibung:**
- Simplexbasierter Optimierungsalgorithmus
- Bessere Konvergenz als Hill-Climbing
- Keine Gradienten notwendig

**Parameter:**
```python
maxiter: int = 200                 # Maximale Iterationen
xatol: float = 1e-5                # Positionstoleraz Konvergenz
fatol: float = 1e-5                # Evaluierungswert-Toleranz
```

---

## 📦 Modul-Beschreibungen

### `ActiveAlignmentServer` (Hauptnode)

**Hauptfunktionen:**
- `execute_callback()`: Verwaltet Action-Austausch und Optimierungsprozess
- `hill_climb_optimization()`: Hill-Climbing Implementierung
- `nelder_mead_optimization()`: Nelder-Mead Implementierung
- `init_controller_manager()`: Setzt Controller und Gelenke auf
- `send_trajectory_from_dict()`: Sendet Gelenktrajectories an Controller

**Wichtige Attribute:**
```python
_action_clients: dict[str, ActionClient]           # Gecachte Action-Clients
_current_joint_state_positions: dict[str, float]   # Aktuelle Gelenkpositionen
optimization_value_manager: OptimizationValueManager
_joint_limit_fetcher: JointLimitFetcher
```

### `AlignmentControllerManager`

**Zweck:** Verwaltung von Controllern und deren Gelenken

**Hauptklassen:**
- `ControllerJoint`: Repräsentation eines einzelnen Gelenks
- `AlignmentController`: Verwaltung eines Controllers mit seinen Gelenken
- `AlignmentControllerManager`: Verwaltung mehrerer Controller

**Wichtige Methoden:**
```python
add_joint_to_controller(controller_name, joint)
get_all_joint_states_as_mapped()          # Normalisiert [0, 1]
set_joint_states_from_mapped(dict)        # Setzt aus normalisiertem Raum
get_controllers() -> list[AlignmentController]
```

### `JointLimitFetcher`

**Zweck:** Abrufen von Gelenkparametern aus dem System

**Methoden:**
```python
get_joint_info(joint_name) -> dict       # Liefert Limits, Controller, Type
get_joints_for_controller(controller_name) -> list[str]
```

### `OptimizationValueManager`

**Zweck:** Verwaltet Evaluierungswerte von mehreren Topics und berechnet das gewichtete Fehlersignal

**Fehlersignal-Berechnung:**

Der Manager aggregiert Werte von mehreren Topics zu einem einzigen Evaluierungswert:

$$\text{eval\_value} = \frac{\sum_{i} w_i \cdot v_i}{\sum_{i} w_i}$$

Wobei:
- $w_i$ = `weight_factor` des Topics $i$
- $v_i$ = aktueller Wert des Topics $i$ (ggf. negiert)
- Negation: Falls `minimize_not_maximize=false`, wird der Wert mit $-1$ multipliziert

**Beispiel:**
```python
# Topic 1: distance (minimieren)
align_topic_1 = AlignTopic()
align_topic_1.topic_name = "/distance"
align_topic_1.minimize_not_maximize = True
align_topic_1.weight_factor = 2.0

# Topic 2: angle (minimieren)
align_topic_2 = AlignTopic()
align_topic_2.topic_name = "/angle"
align_topic_2.minimize_not_maximize = True
align_topic_2.weight_factor = 1.0

# Bei Werten: distance=0.5, angle=0.2
# eval_value = (2.0*0.5 + 1.0*0.2) / (2.0 + 1.0) = 1.2 / 3.0 = 0.4
```

**Methoden:**
```python
init(alignment_topics)              # Initialisierung mit Topics
get_eval_value(wait_for_update_sec, check_for_new_values)  # Berechnet gewichtete Summe
clear()                             # Cleanup
_calc_output_signal()               # Interne Berechnung des Fehlersignals
_topics_have_new_data()             # Prüft auf neue Topic-Updates
```

**Interne Funktionsweise:**
1. Subscribed sich zu allen Topics in `AlignTopic[]`
2. Speichert aktuellste Werte mit Timestamps
3. Bei Abruf: Berechnet gewichtete Summe aller verfügbaren Werte
4. Publiziert das resultierende Fehlersignal auf `/active_alignment/eval_value`

**Besonderheiten:**
- **Neue-Daten-Check**: Wenn `check_for_new_values=True`, wartet auf mindestens einen neuen Messwert
- **Timeout-Handling**: Wartet max `wait_for_update_sec` Sekunden auf gültige Werte
- **Gewichtung**: Höhere `weight_factor` bedeuten größerer Einfluss auf Optimierung
- **Minimierung vs. Maximierung**: 
  - `minimize_not_maximize=True`: Wert wird direkt verwendet (Ziel: minimieren)
  - `minimize_not_maximize=False`: Wert wird negiert (Ziel: maximieren)

---

## 🔄 Workflow-Beispiel

```
1. Client sendet ActiveAlign.Goal
   ├─ active_joints: [joint_1, joint_2]
   └─ alignment_topics: [/feedback_1, /feedback_2]

2. Server initialisiert Controller/Gelenke
   ├─ Lädt Gelenklimits
   ├─ Verbindet sich mit Controllern
   └─ Setzt initiale Positionen

3. Optimierungsschleife startet
   ├─ Iteration 1: Perturbe Positionen → Messe Feedback
   ├─ Iteration 2: Perturbe Positionen → Messe Feedback
   ├─ ...
   └─ Iteration N: Konvergiert oder max erreicht

4. Ergebnis an Client zurück
   ├─ success: true/false
   ├─ num_of_iterations: 42
   ├─ final_eval_value: 0.0234
   └─ joint_results: [{joint_1: 1.234}, {joint_2: -0.567}]
```

---

## 🧪 Testing

### Frame Deviation Publisher für Tests/Echte Ausführung

Der `dummy_value_publisher.py` (eigentlich `MultiFrameDeviationNode`) berechnet **Abweichungen zwischen TensorFlow-Frames** und publiziert diese als Evaluierungssignale.

**Starten:**
```bash
ros2 run active_alignment_skills dummy_value_publisher
```

**Funktionsweise:**

Der Node vergleicht konfigurierte Frame-Paare und berechnet:

1. **Translatorischer Abstand:**
   $$d = \sqrt{x^2 + y^2 + z^2}$$
   - Euklidischer Abstand zwischen zwei Frames
   - Skaliert mit Faktor 10 für bessere Gewichtung gegen Rotationen: $d_{scaled} = 10 \cdot d$
   - Veröffentlicht auf: `/{pair_name}/distance`

2. **Rotationsabweichung (Angle Deviation):**
   $$\theta = |\omega|$$
   - Magnitude des Rotationsvektors (aus Quaternion berechnet)
   - Gibt die Rotation in Radianten an
   - Veröffentlicht auf: `/{pair_name}/angle`

**Konfiguration:**

Die Frame-Paare sind in Zeile ~30 hardcodiert:
```python
self.frame_pairs: List[FramePair] = [
    FramePair("Test1", "AL_Test_Frames_Align_1", "AL_Test_Frames_Target_1"),
    FramePair("Test2", "AL_Test_Frames_Align_2", "AL_Test_Frames_Target_2"),
]
```

Zum Hinzufügen neuer Paare:
```python
FramePair("YourName", "is_frame", "target_frame")
```

**Ausgabe (10 Hz):**
```
[INFO] Test1: distance=5.342 mm, angle_dev=0.03421 deg
[INFO] Test2: distance=12.654 mm, angle_dev=0.08756 deg
```

**Topics:**
- `/Test1/distance` (Float32): Translationale Abweichung
- `/Test1/angle` (Float32): Rotationsabweichung
- `/Test2/distance` (Float32): ...
- `/Test2/angle` (Float32): ...
- (weitere Paare analog)

**Integration mit OptimizationValueManager:**

In der Goal können diese Topics als Optimierungsziele verwendet werden:
```python
align_topic1 = AlignTopic()
align_topic1.topic_name = "/Test1/distance"
align_topic1.minimize_not_maximize = True    # Minimiere Abstand
align_topic1.weight_factor = 2.0

align_topic2 = AlignTopic()
align_topic2.topic_name = "/Test1/angle"
align_topic2.minimize_not_maximize = True    # Minimiere Rotation
align_topic2.weight_factor = 1.0

goal.alignment_topics = [align_topic1, align_topic2]
```

Der Manager berechnet dann:
$$\text{eval} = \frac{2.0 \cdot d + 1.0 \cdot \theta}{2.0 + 1.0}$$

### Unittest durchführen

```bash
# Linting
colcon test --packages-select active_alignment_skills

# Detaillierte Tests
ros2 test active_alignment_skills
```

---

## ⚙️ Konfiguration

### Adaptive Step-Size aktivieren

In `active_alignment_skill_node.py`, Zeile ~350, `hill_climb_optimization()`:

```python
adaptive_step_size = True  # Aktiviere adaptive Schrittweiten
```

Bei Verbesserung: `step_size *= 1.2` (explorativer)
Bei Stagnation: `step_size *= 0.5` (feiner-tuning)

### Optimierungszeiträume anpassen

```python
movement_time = 0.1        # Bewegungsdauer zwischen Schritten
time.sleep(0.5)            # Wartezeit nach Bewegung für Evaluierung
wait_for_update_sec = 5    # Max Wartezeit auf neuen Evaluierungswert
```


## 📊 Ausgabe und Logging

### Optimization History Plots

Der Server speichert nach jeder Optimierung einen Plot:

```
~/.ros/optimization_history_active_alignment_optimization_20240512_143022.png
```

Zeigt den Verlauf des Evaluierungswerts über Iterationen.

### Robot Optimization Log

Bei Nelder-Mead wird zusätzlich eine CSV geschrieben:

```
~/.ros/robot_optimization_log_20240512_143022.csv
```

Format: `[x0, x1, ..., xN, eval_value]`

---

## 📝 Lizenz

- `active_alignment_skills`: Apache-2.0
- `active_alignment_interfaces`: Siehe package.xml

---

## 👤 Autoren & Kontakt

**Maintainer:** 
- `active_alignment_skills`: mll (terei@match.uni-hannover.de)
- `active_alignment_interfaces`: pmlab

---

## 🔗 Zusätzliche Ressourcen

- **ROS2 Dokumentation**: https://docs.ros.org/en/humble/
- **scipy.optimize.minimize**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
- **Joint Trajectory Controller**: https://github.com/ros-controls/ros2_controllers
