# Stochastische Prozesse

Ein stochastischer Prozess beschreibt die Zustände eines Systems zu einem bestimmten Zeitpunkt welche vom Zufall beeinflusst sind.

Für jeden Zeitpunkt $t \in T \subset \mathbb{R}$ beschreibt die Zufallsvariable $X_t: \Omega \mapsto I$ den Zustand eines Systems zum Zeitpunkt $t$. Dann heisst $(X_t, t\in T)$ oder kurz $(X_t)$ **stochastischer Prozess mit Zustandsraum** $I$. Wir beschränken uns hier auf Prozesse mit diskreter Zeit und diskretem Zustandsraum, also sind beide endlich abzählbar ($\mathbb{N}, \mathbb{Z}$).

## Markow-Kette

Bei vielen Systemen hängt der Folgezustand nur vom aktuellen Zustand ab, und nicht noch von allen Zuständen davor. Solche Prozesse nennt man Markow-Ketten. Mathematische ausgedrückt sieht das dann so aus für $(X_n,n\in\mathbb{N})$

$$P(X_n=i_n | X_{n-1}=i_{n-1},...,X_0=i_o)=P(X_n=i_n | X_{n-1}=i_{n-1})$$

hierbei sind alle $i_0,...,i_n \in I$. Ein Beispiel hier wäre, dass wenn ein Server ohne Probleme läuft, dann hat er mit 0.9 Wahrscheinlichkeit am nächsten Tag auch kein Problem. Hat er jedoch ein Problem dann hat er mit 0.5 Wahrscheinlichkeit am nächsten Tag immer noch ein Problem.

### Homogene Markow-Kette, HMK

Wenn nicht nur die ganze Historie sondern auch der Tag keine Rolle spielt dann ist es eine **homogene Markow-Kette** (HMK). Dies wäre nicht der Fall wenn die Wahrscheinlichkeit, dass am Sonntag der Server noch Probleme hat, wenn er am Samstag schon ein Problem hat anders ist also alle andere Tage (Obwohl dies sehr wahrscheinlich der Fall ist weil niemand am Sonntag arbeitet um den Server zu reparieren). Eine Markow-Kette ist also homogen, wenn die Übergangswahrscheinlichkeiten für alle $n$ gleich sind.

$$P(X_n=j|X_{n-1}=i), (i,j\in I)$$

### Übergangs-Matrix / Graph

In diesem Fall heisst die Matrix $P=(p_{ij})$ eine Übergangsmatrix wobei $P_{ij}=P(X_n=j|X_{n-1}=i)$ (Zeile=$i$ und Spalte=$j$). Wichtig hierbei ist, dass die Summe der Reihen 1 ergeben weil die Wahrscheinlichkeiten normalisiert sind.

![markowKetteUebergangsmatrix](/maths/markowKetteUebergangsmatrix.png)

Oftmals werden Markow-Ketten auch mit Hilfe von einem Übergangsgraph dargestellt.

![markowKetteUebergangsgraph](/maths/markowKetteUebergangsgraph.png)

### Zustände in der Zukunft

Mit der Übergangsmatrix können wir Zustände des Systems in der Zukunft berechnen solange wir den Startwert kennen in dem wir folgendes machen

$$P(X_{n+m}=j|X-n=i) \text{ ist der ij-te Eintrag der Matrix } P^m$$

Jedoch ist der Startwert nicht immer bekannt dieser kann auch zufällig sein. Zum Beispiel kann ein Server mit 1% Wahrscheinlichkeit beim liefern schon kaputt gehen. Man hat also eine Startverteilung $P(X_0=i)$ für alle $i \in I$. Einen nicht zufälligen also festen Startwert $X_0=s$ kann man auch so modellieren $P(X_0=s)=1$ für all andere $i\neq s,$ $P(X_0=i)=0$.

Der Vektor mit den Einträgen $P(X_0=i)$ für alle $i \in I$ bezeichnet man meistens mit $\pi_0$. Mit $\pi_n$ bezeichnet man den Vektor mit den Einträgen für $P(X_n=i)$ für alle $i \in I$. Daraus folgt dann

$$\pi_n=\pi_0 \cdot P^n$$

Für unseres vorherige Beispiel erhalten wir also die Startverteilung $\pi_0=(0.99,0.01)$ daraus können wir dann berechnen was die Verteilung am ersten Tag, am vierten etc ist.

- $\pi_1=\pi_0 \cdot P = (0.99 \cdot 0.9 + 0.01 \cdot 0.5, 0.99 \cdot 0.1 + 0.01 \cdot 0.5)=(0.896, 0.104)$
- $\pi_4=\pi_0 \cdot P^4=(0.8434, 0.1566)$
- $\pi_364=\pi_0 \cdot P^{364}=(0.8333, 0.1667)$

### Reguläre HMK

Die Frage, ob der Server an einem konkreten Tag Probleme hat, ist aber eigentlich gar nicht so wichtig. Viel wichtiger ist die Frage, an wie vielen Tagen der Server langfristig Probleme hat. Dazu schauen wir uns das Verhalten von $\pi_n$ über die Zeit an.

![markowKetteTimeGraph](/maths/markowKetteTimeGraph.png)

Wir sehen also das nach einer Phase beträgt die Wahrscheinlichkeit, das der Server an einem beliebigen Tag ein Problem hat, 16.667%. Mit einem anderen Startwert erhalten wir.

![markowKetteTimeGraph2](/maths/markowKetteTimeGraph2.png)

Die Konvergenz, unabhängig von der Startverteilung, ist kein Zufall, dies gilt für jede **reguläre** HMK. Eine HMK mit Übergangsmatrix $P$ heisst regulär wenn ein $n$ existiert, so dass all Einträge von $P^n$ grösser als 0 sind.

### Grenzverteilung

Zu jeder regulären HMK existiert eine sogenannte Grenzverteilung(auch stationäre oder Gleichgewichtsverteilung) $\pi^*= (\pi_1^*,...,\pi_m^*)$. Diese Grenzverteilung hat die folgende Eigenschaften

- Für jede Startverteilung $\pi_0$ gilt $\lim_{n \to \infty}{\pi_0\cdot P^n=\pi^*}$
- $\pi^*\cdot P = \pi^*$
- Die Zeilen der Matrix $P^n$ konvergieren gegen die Grenzverteilung $\pi^*$

Um die Gleichgewichtsverteilung zu bestimmen muss man das folgende Gleichungssystem lösen:

- Normierung Bedingung (N): $\sum_{i\in I}{\pi_i^*}=1$
- Gleichgewicht Bedingung (G): $\pi^*\cdot P = \pi^*$
