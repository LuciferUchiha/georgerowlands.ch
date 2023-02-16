# Diskrete Zufallsvariablen

## Zufallsvariable

Eine Zufallsvariable $X$ ist eine Funktion welche jedem Elementarereignis $\omega \in \Omega$ genau eine reele Zahl zuordnet. Wir unterscheiden noch dabei ob eine Zufallsvariable **diskret** oder **stetig** ist.  Hier werden wir diskrete Zufallsvariablen anschauen später werden wir noch stetige dazu nehmen.

### Diskrete Zufallsvariable

Eine diskrete Zufallsvariable kann endlich viele oder abzählbar unendliche viele Werte annehmen.
Gute Beispiel dafür sind Ergebnisse beim Würfeln, Anzahl Münzwurfe bis zum ersten Mal Kopf etc.
Eine gute Video Erklärung dazu gibt es auch [hier](https://studyflix.de/statistik/diskrete-zufallsvariablen-1114)

## Dichte-/Wahrscheinlichkeits-funktion

Bei einer diskreten Zufallsvariable gehört zu jedem Wert $x_i$ eine bestimmte Wahrscheinlichkeit $P(X=x_i)$. Dies Beziehung lässt sich gut mit einer sogenannten Verteilungstabelle oder einem Stabdiagram (Wahrscheinlichkeitsdiagram) visualisieren, dabei gilt

$$f(x_i)=p_i \text{ wobei } p_i \in [0,1]$$

Eine Dichtefunktion ist auch normiert was so viel heisst wie, dass alle Wahrscheinlichkeiten der verschiedenen Werten $x_i$ der Zufallsvariable $X$ zusammen 1 ergeben.
$$\sum_{x_i \in X}{f(x_i)}=1$$

Verteilungstabelle:

![verteilungsTabelle](/maths/verteilungsTabelle.png)

Stabdiagram:

![stabdiagram](/maths/stabdiagram.png)

mehr dazu findest du [hier](https://studyflix.de/statistik/stetige-dichtefunktion-und-verteilungsfunktion-1080)

## Verteilungsfunktion

Die Verteilungsfunktion $F(x)$ einer Zufallsvariable $X$ ist die Wahrscheinlichkeit dafür, dass die Zufallsvariable $X$ einen Wert, der kleiner oder gleich $x$ annimmt.

$$F(x)=P(X \leq x) = \sum_{x_i \leq x}{f(x_i)}$$

Die Wahrscheinlichkeit bei einer diskreten Zufallsvariable $P(a < X \leq b)$ ist gegeben durch $F(b)-F(a)$

![verteilungsFunktion](/maths/verteilungsFunktion.png)

mehr dazu findest du [hier](https://studyflix.de/statistik/diskrete-dichte-und-verteilungsfunktion-1115)

## Erwartungswert

Der Erwartungwert ist der Wert den wir im Durchschnitt erwarten können und ist definiert als

$$E(X)=\sum_{x_i \in X}{x_i\cdot f(x_i)}=\sum_{x_i \in X}{x_i \cdot P(X=x_i)}$$

:::note Beispiel Erwartungswert würfeln

Beim Wurf eines Würfels mit $X=$Augenzahl ist

$$E(X)=3.5$$

:::

## Varianz

Wenn wir den Erwartungswert von $X$ als $\mu$ beschreiben, dann ist die Varianz gegeben durch

$$V(X)=\sigma^2(X)=\sum_{x_i \in X}{(x_i - \mu)^2 \cdot f(x_i)}=\sum_{x_i \in X}{(x_i - \mu)^2 \cdot P(X=x_i)}$$

oder auch in kurz
$$V(X) = \sigma^2(X) = E(X^2) - E(X)^2$$

## Standardabweichung

Die Standardabweichung ist gegeben durch

$$\sigma(X)=\sqrt{V(X)}$$

## Diskrete Verteilungen

Nun schauen wir uns ein paar Verteilungen an die häufig vorkommen wenn man mit diskreten Zufallsvariablen arbeitet.

### Binomialverteilung

Die Binomialvereteilung der Zufallsvariable $X$ ist die Anzahl Treffer bei der $n$-maligen unabhängigen Durchführung eines Experiments mit 2 Elementarereignisse, Treffer und kein Treffer wobei $p$ die Wahrscheinlichkeit für ein Treffer ist.

- Wir schreiben  dann $X \sim Bin(n,p)$
- Die Dichtefunktion von $X$ ist $f(k)=\binom{n}{k}p^k(1-p)^{n-k}$ wobei $k$ die Anzahl benötigter treffer ist.
- $E(X) = n \cdot p$
- $V(X) = n \cdot p \cdot (1-p)$

In Matlab haben wir die Funktionen:

- Dichtefunktion $binopdf(k,n,p)$ wobei $pdf$ English ist und für "probability density function" steht
- Verteilungsfunktion $binocdf(k,n,p)$ wobei $cdf$ English ist und für "cumalitive distribution function" steht

mehr dazu findest du [hier](https://studyflix.de/statistik/binomialverteilung-1118)

:::note Beispiel Binomialverteilung

Ein Multiple Choice Test besteht aus 12 Fragen mit je 4 möglichen Antworten wovon genau 1 richtig ist. Der Test wird durch Erraten ausgefüllt. Wie gross ist die Wahrscheinlichkeit für mehr als 8 richtige Antworten?

$X \sim Bin(12,1/4)$

Mit $P(X \geq 9) = P(X=9)+...+P(X=12)=1 - P(x \leq 8)$

Und somit dann $1-binocdf(8,12,1/4)$

:::

### Bernoulli-Verteilung

Die Bernoulli-Verteilung ist eine spezielle Form der Binomialverteilung wobei $n=1$. Wir können dann alles  ein wenig vereinfachen.

- Wir schreiben dann $X \sim B(p)$
- Die Dichtefunktion von $X$ ist $f(0)= 1-p, f(1)=p$
- $E(X) = p$
- $V(X) = p \cdot (1-p)$

mehr dazu findest du [hier](https://studyflix.de/statistik/bernoulliverteilung-1117)

### Geometrische Verteilung

Die Geometrische Verteilung der Zufallsvariable $X$ ist die Anzahl der Versuche bis zum ersten Treffer bei der wiederholten unabhängigen Durchführung eines Experiments mit 2 Elementarereignisse, Treffer und kein Treffer wobei $p$ die Wahrscheinlichkeit für ein Treffer ist.

- Wir schreiben dann $X \sim Geo(p)$
- Die Dichtefunktion von $X$ ist $f(k)=\binom{n}{k}p\cdot (1-p)^{k-1}$ wobei $k$ bedeutet, dass die ersten $k-1$ Versuche kein Treffer waren aber der $k$-te Versuch ein Treffer ist.
- $E(X) = \frac{1}{p}$
- $V(X) = \frac{1-p}{p^2}$

In Matlab haben wir die Funktionen:

- Dichtefunktion: $geopdf(k-1,p)$
- Verteilungsfunktion $geoocdf(k-1,p)$

mehr dazu findest du [hier](https://studyflix.de/statistik/geometrische-verteilung-1120)

:::note Beispiel Geometrische Verteilung

Wir würfeln solange bis eine sechs kommt. Wie hoch ist die Wahrscheinlichkeit, dass dies im zehnten Versuch passiert?

$X \sim Geo(1/6)$
Mit $P(X = 10) = (\frac{5}{6})^9 \cdot \frac{1}{6} = geopdf(9,1/6) \approx 3.2$%

:::

### Hypergeometrische Verteilung

Die Hypergeometrische Verteilung der Zufallsvariable $X$ ist die Verteilung, die beim $n$-maligen Ziehen ohne Zurücklegen und ohne Reihenfolge aus einer Urne mit $N$ Kugeln, von denen $M$ eine spezielle Eigenschaft haben und wo die Anzahl der gezogenen Kugeln mit dieser speziellen Eigenschaft gezählt werden.

- Wir schreiben dann $X \sim Hyp(N,M,n)$
- Die Dichtefunktion von $X$ ist $f(k)=\binom{M}{k} \cdot \frac{\binom{N-M}{n-k}}{ \binom{N}{n}}$ wobei $N$ die Gesamtanzahl der Kugeln ist, $M$ die Anzahl mit der speziellen Eigenschaft. $n$ ist dann der Umfang der Stichprobe also die Anzahl der entnommenen Kugeln und $k$ die Anzahl angestrebte Kugeln mit der speziellen Eigenschaft.
- $E(X) = n \cdot \frac{M}{N}$
- $V(X) = n \cdot \frac{M}{N} \cdot (1 - \frac{M}{N}) \cdot \frac{N-n}{N-1}$

In Matlab haben wir die Funktionen:

- Dichtefunktion: $hygepdf(k,N,M,n)$
- Verteilungsfunktion $hygecdf(k,N,M,n)$

mehr dazu findest du [hier](https://www.youtube.com/watch?v=BoPYslAe8sg)

:::note Beispiel Hypergeometrische Verteilung

Das perfekte Beiepiel dafür ist Lotto, wobei wir 49 nummerierte Kugeln haben, 6 davon werden gezogen, welche in diesem Falle unsere spezielle Kugeln sind. Wir dürfen 6 Zahlen aufschreiben, also sind das unsere Kugeln die wir herausnehmen ohne zurücklegen oder die Reihenfolge zu beachten. Was ist nun die Wahrscheinlichkeit das wir 4 von den 6 richtig haben?

$X \sim Hyp(49,6,6)$
$hygepdf(4,49,6,6) = \frac{645}{665896} \approx 0.09686$%

:::

### Poisson-Verteilung

Die Poisson-Verteilung kommt bei Zufallsvariablen zum Einsatz, welche die Anzahl der Ereignisse einer bestimmten Art in einem Zeit- und/oder Ortsintervall beschreiben die Anzahl dieses Ereignisses entspricht $\lambda$. Diese Ereignisse sind oftmals "seltene" Ereignisse z.B.

- $X$ Anzahl Druckfehler auf einer Seite eines Buchs
- $X$ Anzahl Unfälle an einem Wochenende in einem Skigebiet
- $X$ Anzahl falsch gewählter Telefon-Nummern an einem Tag
- $X$ Anzahl Erdbeben in einem Jahr in einer bestimmten Region.

Dies sind nur ein paar Beispiele der Poisson-Verteilung, sie ist einer der wichtigsten Verteilungen die wir kennen.

- Wir schreiben dann $X \sim Poi(\lambda)$
- Die Dichtefunktion von $X$ ist$f(k)=\frac{\lambda^k}{k!} \cdot e^{-\lambda}$
- $E(X) = \lambda$
- $V(X) =\lambda$

In Matlab haben wir die Funktionen:

- Dichtefunktion: poisspdf(k,$\lambda$)
- Verteilungsfunktion poisscdf(k,$\lambda$)

mehr dazu findest du [hier](https://studyflix.de/statistik/poissonverteilung-1121)

:::note Beispiel Poisson-Verteilung

Der Druchschnitt der Anzahl Druckfehler pro Seite ist 0.4.
Dann ist $X \sim Poi(0.4)$ ein gutes Modell.
Damit erhalten wir:

- $P(X=0)=poisspdf(0,0.4)=67.03$%
- $P(X=2)=poisspdf(2,0.4)=5.36$%

:::
