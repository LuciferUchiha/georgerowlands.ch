# Bediensysteme

Oftmals wird dieses Thema auch als Warteschlangentheorie bezeichnet. In der klassischen Warteschlangentheorie geht man davon aus, dass sich im System eine Grenzverteilung eingestellt hat wofür wir bestimmte Kenngrössen bestimmen.

1. Verteilung der Anzahl Kunden im System.
2. Länge der Warteschlange $N_Q$.
3. Anzahl der Kunden die gerade bearbeitet werden $N_s$.
4. Erwartete Wartezeit eines Kunden in der Schlange $W_q$.
5. Verweilzeit eines Kunden im System $D$ (Warten + Abarbeitung).

## Kendall-Notation

Um Bediensysteme zu beschreiben verwenden wir die Kendall-Notation. Dabei werden die Charakterisierungen in eines bestimmte Reihenfolge geschrieben.

$$A|B|s|c|R$$

### A - Art des Ankunftprozesses

Hier gibt es 3 beliebte Optionen:

- $D$ für deterministisch, also das die Ankünfte der Kunden zu festen Zeitpunkten stattfinden.
- $G$ für Generelle Annahmen, also das nicht bekannt ist über die Ankünfte der Kunden.
- $M$ für Markov-Eigenschaft, also ist die Wartezeit auf den nächsten Kunden exponentialverteilt weil es das no-memory-property besitzt so wie Markow-Ketten.

### B - Art des Bedienvorgangs

Auch hier gibt es die 3 beliebten Optionen:

- $D$ für deterministisch, also die Dauer zur Abarbeitung eines Kunden ist nicht zufällig.
- $G$ für Generelle Annahmen, also das nicht bekannt ist die Dauer der Abarbeitung der Kunden.
- $M$ für Markov-Eigenschaft, also das die Dauer der Abarbeitung eines Kunden ist exponentialverteilt.

### S - Anzahl Server

Ist entweder eine ganze Zahl grösser gleich 1 oder $\infty$.

### C - Capacity

Die grösse  des Warteraums welche entweder eine ganze Zahl grösser gleich 0 ist oder $\infty$.

### R - Reihenfolge der Bedienung

Hier gibt es auch wieder 3 beliebte Optionen:

- **FIFO** first in first out
- **LIFO** last in first out
- **SIRO** service in random order

### Spezialfälle

Der Standardfall ist FIFO welcher oft weggelassen wird, ebenso bei $c=\infty$.  Wenn $s=\infty$ entfällt die Angabe von C, weil kein Warteraum benötigt wird.

## M|M|s|c-Systeme

Wir beginnen mit der Grenzverteilung von einem M|M|1|0-System und leiten dann daraus die Formeln für M|M|s|c-Systeme.

Wir gehen davon aus das die Wartezeit exponentialverteilt ist mit dem Parameter $\lambda$(Ankunfts-rate) und das die Dauer zur Abarbeitung eines Kunden auch exponentialverteilt ist mit dem Parameter $\mu$(Bedien-rate). Als Einheit nehmen wir Stunden, also kommen pro Stunde im Schnitt $\lambda$ Kunden.

Wir haben einen Server und keinen platz im Warteraum. Das heisst, dass das System sich in zwei verschiedene Zustände befinden kann (1: Kund wird bearbeitet oder 0: Kein Kunde wird bearbeitet). Wird ein Kunde gerade bearbeitet, so werden weitere Kunden abgewiesen.

Wir wählen dann ein Zeitintervall $h$ in Stunden, welche so klein ist (limes gegen 0), dass in diesem Zeitintervall höchstens ein Kunde ankommt und höchstens ein Kunde abgearbeitet wird. Da im Zeitintervall $h$ also entweder ein Kunde kommt oder nicht muss die Wahrscheinlichkeit für das Ankommen eines Kunden $\lambda \cdot h$ sein damit immer noch $\lambda$ Kunden pro Stunde kommen. Analog ist die Wahrscheinlichkeit, dafür dass ein Kunde welcher gerade  bearbeitet wird fertiggestellt wird$\mu \cdot h$. Daraus erhalten wir dann

![bediensystemeGraph](/maths/bediensystemeGraph.png)

Daraus erhalten wir dann die folgende Formeln für die Elemente der Grenzverteilung $\pi^*=(p_0,p_1)$

$$p_0=\frac{(1-\lambda h)\cdot \mu}{(1-\lambda h)\cdot \mu + \lambda} \text{ mit } {h\to 0} = \frac{\mu}{\mu + \lambda}$$

$$p_1=1-p_0=\frac{\lambda}{\mu + \lambda}$$

Die Wahrscheinlichkeit, dass der Server belegt ist, beträgt also $\frac{\lambda}{\lambda + \mu}$

In einem M|M|s|c-System sieht das dann so aus

![bediensystemeFormeln](/maths/bediensystemeFormeln.png)

### Kennzahlen

#### Erwartete Warteschlangenlänge

Für die Erwartete Warteschlangenlänge wobei die Warteschlange die Länge $j-s$ hat, wenn $j$ Kunden im System sind und dafür beträgt die Wahrscheinlichkeit $p_j$ ergibt sich:

$$E(N_Q)=\sum_{j=s+1}^{s+c}{(j-s)p_j}$$

Dies lässt sich dann umformulieren zu

$$E(N_Q)=\begin{cases}
\frac{p_0\cdot(\frac{\lambda}{\mu})^s \cdot \frac{\lambda}{s\mu}\cdot (1+c)\cdot c}{2\cdot s!} \quad & if \lambda=s\mu\\
\frac{p_0\cdot(\frac{\lambda}{\mu})^s \cdot \frac{\lambda}{s\mu}[1-(\frac{\lambda}{s\mu})^{c+1} - (1-\frac{\lambda}{s\mu})\cdot(c+1)\cdot(\frac{\lambda}{s\mu})^c]}{s!\cdot (1-\frac{\lambda}{s\mu})^2} \quad & if \lambda \neq s\mu
\end{cases}$$

#### Erwartete bediente Kunden

Wenn $0\leq j < s$ Kunden im System sind mit Wahrscheinlichkeit $p_j$, werden $j$ Kunden bedient. Wenn mindestens $s$ Kunden im System sind mit der Wahrscheinlichkeit $\sum_{j=s}^{s+c}{p_j}$ werden $s$ Kunden bedient.

Vereinfachen kann man es zu

$$E(N_s)=\frac{\lambda}{\mu}\cdot (1-p_{s+c})$$

#### Erwartete Wartezeit

Aus der Formel von Little, welche besagt, dass die erwartete Anzahl von Kunden in einem System gleich dem Produkt ihrere durchschnittlichen Ankunfts-rate und ihre erwartete Verweildauer im System ist.

$$E(W_Q)=\frac{E(N_Q)}{\lambda \cdot (1-p_{s+c})}$$

#### Erwartete Verweilzeit

Für die erwartete Verwilzeit eines Kunden im System ergibt sich

$$E(D)=E(W_Q)+\frac{1}{\mu}$$

### Kennzahlen mit unendlich Warteraum

Wenn $\lambda \geq s\mu$ existiert keine Gleichverteilung weil mehr Anfragen rein kommen also von den Servern bearbeitet werden können und somit geht die Wartezeit gegen unendlich. Sonst gibt es die folgende Grenzverteilung

![bediensystemeFormelnUnendlich](/maths/bediensystemeFormelnUnendlich.png)
