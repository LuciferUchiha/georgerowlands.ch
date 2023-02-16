# Kombinatorik

## Fakultät

Die **Fakultät** einer nicht negativen ganzen Zahl $n$, definiert als $n!$, ist das Produkt aller positiven ganzen Zahl kleiner gleich $n$.

$$n! = n \cdot (n-1) \cdot (n-2)\cdot ... \cdot 2 \cdot 1$$

Zum Beispiel ist $5!=5\cdot 4 \cdot 3 \cdot 2 \cdot 1 = 120$
Die einzige Ausnahme ist $0! = 1$.

## Binomialkoeffizient

$$\binom{n}{k}=\frac{n!}{k!(n-k)!}$$

### Für Wahrscheinlichkeiten

Der Binomialkoeffizient ist eine mathematische Funktion, welche angibt, auf wie viele verschiedene Arten man $k$ Objekte aus einer Menge von $n$ verschiedenen Objekten auswählen kann (ohne Zurücklegen und ohne die Reihenfolge zu beachten). Der Binomialkoeffizient ist also die Anzahl der $k$-elementigen Teilmengen einer $n$-elementigen Menge.

Die englische Abkürzung **nCr**, welche für **n choose r**  steht findet man oft auf Taschenrechnern.

### Für Binomische Terme

Bei einem binomischen Term der Form $(x + y)^n , n \in \mathbb{N}$ geben die Binomialkoeffizienten die Koeffizienten der Ausmultiplikation an. Dazu verwendet man die folgende Formel

$$(x + y)^n = \sum_{k=0}^{n}{\binom{n}{k}x^{n-k}y^k}$$

:::note Beispiel Binom Koeffizienten mit Binomialkoeffizienten

$$(x+y)^3=\binom{3}{0} x^{3} + \binom{3}{1} x^{2}y + \binom{3}{2} xy^{2} + \binom{3}{3} y^{3}=x^3+3x^2y+3xy^2+y^3$$

:::

### Pascalsches Dreieck

Das **Pascalsche Dreieck** ist eine grafische Darstellung der Binomialkoeffizienten, welches auch eine einfache Berechnung erlaubt. Dabei zeichnet man ein Dreieck wo die variable $n$ dem Zeilenindex und $k$ dem Spaltenindex entspricht.

![PascalsTriangle](/maths/pascalsTriangle.png)

Die Herleitung kommt von der Gleichung

$$\binom{n+1}{k+1}=\binom{n}{k} + \binom{n}{k+1}$$

:::note Beispiel für Pascalsches Dreieck Herleitung

 Wir nehmen den Wert in der dritten Reihe in der Mitte $\binom{2}{1}$

 $$\begin{align}
   \binom{2}{1} &= \binom{1}{0} + \binom{1}{1} \\
   \frac{2!}{1!(2-1)!} &= \frac{1!}{0!(1-0)!} +
   \frac{1!}{1!(1-1)!} \\
   \frac{2}{1!1!} &= \frac{1}{1!} + \frac{1}{1!0!} \\
   \frac{2}{1} &= \frac{1}{1} + \frac{1}{1} \\
   2 &= 1 + 1
   \end{align}$$

:::

Diese Herleitung führt dazu, dass jede Zahl die Summe der Zahlen genau über sich sind. Dabei setzt man zuerst für alle Elemente am Rand des Dreiecks eine 1 ein, weil ein Binom mit $0$ oder $n$ im unteren Teil lässt sich auflösen zu 1.

![PascalsTriangleAnimated](/maths/pascalsTriangleAnimated.gif)

## Urnenmodel

Das Urnenmodel ist eine verbreite Veranschaulichung um viele Probleme der Wahrscheinlickeitsrechnung zu lösen. Dafür stellen wir uns vor wir haben eine Urne mit $n$ **verschiedene** Kugeln, die sich z.B. in ihrer Farbe oder einer Beschriftung unterscheiden und ziehen dann davon $k$ Kugeln. Dabei wird noch unterschieden ob wir nach dem Ziehen einer Kugel, die Kugel wieder zurücklegen oder nicht.

Oftmals wird auch das zufällige ziehen der Kugeln eine **Stichprobe** genannt.

![urnenModell](/maths/urnenModell.png)

## Permutation

Jede mögliche **Anordnung** von $n$ Kugeln heisst eine **Permutation** der $n$ Kugeln. Die Anzahl der Permutationen hängt noch davon ab, ob alle Kugeln verschieden sind (Ohne Wiederholung), oder ob es gewisse Kugeln mehrmals hat (Mit Wiederholung). Diese Unterscheidung macht man, weil die Permutationen nicht verschieden sind wenn man die Kugeln die gleich sind vertauscht.

![permutationMitWiederholung](/maths/permutationMitWiederholung.png)

### Ohne Wiederholung

Sind alle Kugeln verschieden, also gibt es keine Wiederholungen, dann ist die Anzahl der verschiedenen möglichen Permutationen, und somit auch Anordnungen

$$Per(n)=n!$$

Wir ziehen also aus einer Urne mit $n$ Kugeln $n$-mal eine Kugel daraus. Wir können uns also vorstellen das wir für den ersten Zug $n$ mögliche Kugeln, weil sich noch alle in der Urne befinden. Danach für den zweiten Zug gibt es nurnoch $n-1$ mögliche Kugeln die wir aus der Urne ziehen können, weil wir eine schone herausgenommen haben. Das geht weiter so bis keine Kugel mehr in der Urne sind. Am Schluss wird dann alles multipliziert was dann $n \cdot (n-1) \cdot (n-2)\cdot ... \cdot 2 \cdot 1 = n!$ ergibt.

Eine gute Video Erklärung dazu gibt es auch [hier](https://studyflix.de/statistik/permutation-ohne-wiederholung-1071)

:::note Beispiel Permutation ohne Wiederholung

 Auf einem Regal sollen 3 verschiedene Bücher($b_1,b_2,b_3$) angeordnet werden.

 Es gibt dann $3!=6$ verschiedene Anordnungen.
 $(b_1,b_2,b_3),(b_1,b_3,b_2),(b_2,b_1,b_3),(b_2,b_3,b_1),(b_3,b_1,b_2),(b_3,b_2,b_1)$

:::

### Mit Wiederholung

Wenn sich unter den Kugeln $n_1,n_2,...,n_k$ gleiche Kugeln befinden, also es Wiederholungen hat, so ist die Anzahl der verschiedenen Anordnungsmöglichkeiten

$$Per(n;n_1,n_2,...,n_k) = \frac{n!}{n_1!n_2!...n_k!}$$

Eine gute Video Erklärung dazu gibt es auch [hier](https://studyflix.de/statistik/permutation-mit-wiederholung-1070)

:::note Beispiel Permutation mit Wiederholung

 In einer Urne befinden sich 6 Kugeln, darunter sind 3 weisse, 2 graue und 1 schwarz. Die gleichfarbigen Kugeln sind nicht von einander unterscheidbar.

 Es gibt dann $\frac{6!}{3!2!1!} = \frac{6!}{3!2!} = 60$ verschieden Anordnungsmöglichkeiten.

:::

## Kombination

Bei der **Kombination** werden nacheinander $k$ Kugeln aus einer Urne mit $n$ Kugeln gezogen. Dabei schauen wir nicht auf die Reihenfolge in der wir die Kugeln ziehen, dass heisst wenn wir 2 Kugeln ziehen und wir zuerst eine Schwarze und dann eine Weisse ziehen zählen wir als das selbe Resultate wie wenn wir zuerst eine Weisse und dann eine Schwarze gezogen hätten.

Hier unterscheiden wir auch wieder zwischen 2 Fälle, ob wir nach dem Ziehen die Kugel wieder in die Urne zurücklegen oder nicht.

### Ohne Zurücklegen

Hier lautet also die genaue Fragestellung "Auf weiviele verschieden Arten können wir $k$ Kugeln aus einer Urne mit $n$ verschiedenen Kugeln ziehen, ohne sie nach dem Ziehen zurückzulegen und ohne die Reihenfolge in der wir die Kugeln ziehen, zu beachten."

Dieses Problem lässt sich ziemlich gut zu einer Permutation umwandeln. Wir können uns vorstellen das wir jede gezogene Kugel als 1 markieren und die anderen als 0. So bekommen wir zum Schluss eine Binärezahl mit $k$ mal eine 0 und $n-k$ mal eine 1.
Wir können uns nun die Frage stellen, wieviele verschiedene Anordnungsmöglichkeiten gibt es für die $n$ Zahlen mit $k$ und $n-k$ gleiche Zahlen.

$$C(n;k) = Per(n;k,(n-k)) = \frac{n!}{k!(n-k)!} = \binom{n}{k}$$

Was genau dem Binomialkoeffizient entspricht.

Eine gute Video Erklärung dazu gibt es auch [hier](https://studyflix.de/statistik/ziehen-ohne-zuruecklegen-1077)

:::note Beispiel Kombination ohne zurücklegen

 Im Lotto gibt es 49 Zahlen, davon werden 6 ohne wiederholung gezogen und die Reihenfolge der Zahlen wird nicht beachtet.

 So gibt es $\binom{49}{6}=13'983'816$ verschiedene Kombinationen

:::

:::note Enigma Beispiel Kombination ohne zurücklegen

 Um eine Enigma maschine zu betätigen müssen 3 Rotoren von 5 ausgewählt weden. Das Militär hat sogar 8 Rotoren zur Auswahl.

 $\binom{5}{3}=10$ verschiedene Kombinationen

 $\binom{8}{3}=56$ verschiedene Kombinationen

 Wie man sieht Steigt die Zahl drastisch wenn man mehr Rotoren zur Auswahl hat.

:::

### Mit Zurücklegen

Wenn wir nach dem ziehen die Kugeln wieder zurücklegen, dann kann es sein, dass eine Kugel mehrmals verwendet wird. Dabei kommen wir auf

$$C_w(n;k) = \frac{(n+k-1)!}{k!(n-k)!} = \binom{n+k-1}{k}$$

Eine gute Video Erklärung dazu gibt es auch [hier](https://studyflix.de/statistik/ziehen-mit-zurucklegen-ohne-reihenfolge-1074)

:::note Beispiel Kombination mit zurücklegen

 Wie viele Kombinationsmöglichkeiten gibt es beim dreimaligen Würfeln?

 Vergleicht man die drei Würfe mit der Anzahl der zu ziehenden Kugeln $k$ und die sechs möglichen Ergebnisse, nämlich die Würfelaugen 1 bis 6, mit der Gesamtzahl der Kugeln $n$, erhält man folgende Anzahl möglicher Ergebnisse:

 $$\binom{6+3-1}{3}=\binom{8}{3}=56$$

:::

:::note Gummibärchen-Orakel Beispiel Kombination mit zurücklegen

 Beim sogenannten Gummibärchen-Orakel haben wir eine Tüte mit Gummibärchen. Wir wissen nicht die Anzahl der Gummibärchen aber das es sie in 5 verschiedene Farben gibt. Wir nehmen aus der Tüte 5 Gummibärchen. Die Frage ist demnach wieviele Farbkombinationen kann man ziehen.

 $$\binom{5+5-1}{5}=\binom{9}{5}=126$$

:::

## Variation

Bei der **Variation** haben wir genau die gleiche Überlegungen wie bei der Kombination, nur berücksichtigen wir jetzt die Reihenfolge in der wir die Kugeln ziehen.

### Ohne Zurücklegen

Da wir $k$-mal aus einer Urne mit $n$ Kugeln ziehen und sie nicht zurücklegen haben wir eigentlich eine Kombination. Nur berücksichtigen wir jetzt die Reihenfolge. Also kommt die Frage noch auf wieviele Arten können wit $k$ Kugeln anordenen, eine Permutation mit $k!$  verschiedene Anordnungen. Somit kommen wir auf

$$V(n;k) = k! \cdot C(n;k) = k! \cdot \frac{n!}{k!(n-k)!} = \frac{n!}{(n-k)!}$$

Eine gute Video Erklärung dazu gibt es auch [hier](https://studyflix.de/statistik/ziehen-ohne-zurucklegen-mit-reihenfolge-1073)

:::note Beispiel Variation ohne zurücklegen

 Beim Pferdewetten muss in der sogenannten "Dreierwette" die Reihenfolge der ersten 3 Pferde die ins Ziel laufen korrekt angegeben werden. Die Frage ist nun wieviele Dreierwetten gibt es wenn das Rennen 10 Pferde hat.

 Es gitb also $\frac{10!}{(10-3)!} = \frac{10!}{7!} = 8 \cdot 9 \cdot 10 = 720$ verschiedene Dreierwetten.

:::

### Mit Zurücklegen

Da wir $k$-mal ziehen können wir uns vorstellen, dass wir $k$ Stellen haben und weil wir nach jeder Ziehung die Kugel wieder in die Urne legen haben wie bei jeder Ziehung $n$ mögliche Kugeln die wir ziehen könnten. Daraus lässt sich folgen

$${V_w(n;k)} = {n \cdot n ... \cdot n = n^k}$$

Eine gute Video Erklärung dazu gibt es auch [hier](https://studyflix.de/statistik/ziehen-mit-zurucklegen-mit-reihenfolge-1072)

:::note Beispiel Variation mit zurücklegen

 Bei einem Zahlenschloss hat es vier Ringe die je Zehn Ziffern haben. So gibt es $10^4=10000$ verschiedene Variationen, die Zahlen 0000 bis 9999.

:::

## Zusammenfassung

![kombinatorikUebersicht](/maths/kombinatorikUebersicht.png)
