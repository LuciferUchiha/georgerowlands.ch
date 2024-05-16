# Grenzwertsätze

## Ungleichung von Tschebyscheff

Wenn $X$ eine Zufallsvariable ist mit einem Erwartungswert von $\mu$ und eine Varianz $\sigma^2$ dann kann man die Wahrscheinlichkeit die folgende Wahrscheinlichkeit abschätzen für jede mögliche Verteilung von $X$

$$P(|X-\mu|\geq k) \leq \frac{\sigma^2}{k^2}, \text{ für k > 0}$$

daraus folgt dann auch

$$P(|X-\mu| < k) \leq 1 - \frac{\sigma^2}{k^2}$$

:::note Beispiel Ungleichung von Tschebyscheff

 Für die Grösse einer erwachsenen Personen haben wir einen Erwartungswert von 175cm und eine Standardabweichung von 10cm. Was ist die Wahrscheinlichkeit, dass eine Person kleiner als 160cm oder grösser als 190cm ist.

 Abschätzung mit Tschebyscheff:

 $$P(|X -175cm| \geq 15) \leq \frac{100}{15^2} \approx 44.4 \%$$

 Tatsächlich mit Normalverteilung:

 $$P(|X -175cm| \geq 15) = 1- (normcdf(190,175,10) - normcdf(160,175,10)) \approx 13.4 \%$$

:::

Mehr dazu findest du [hier](https://studyflix.de/statistik/tschebyscheff-ungleichung-1546)

## Gesetz der grossen Zahlen

Wir wissen wenn wir eine Zufallsvariable $X_i$ mit einer Bernoulli Verteilung haben also $X_i \sim B(p)$ dann ist $\sum_{i=1}^{n}{X_i} \sim Bin(n,p)$. Die relative Häufigkeit ist wie wir wissen die Anzahl des Eintreffen eines Ereignis $X$ durch die anzahl unabhängige Ausführungen des Experiments $n$ dann gilt folgendes

$$\lim_{n \to \infty}{P(|\frac{X}{n} - p| \geq e)} = 0$$

Was so viel heisst wie wenn desto mehr unabhängige Experimente wir ausführen desto besser stabilisiert sich die relative Häufigkeit um den Erwartungswert.

Mehr dazu findest du [hier](https://studyflix.de/statistik/gesetz-der-grosen-zahlen-2053)

## Zentraler Grenzwertsatz

Der zentrale Grenzwertsatz liefert die Begründung für das Phänomen, dass sich bei der additiven Überlagerung vieler kleiner unabhängiger Zufallsexperiment approximativ zu einer Normalverteilung wird.

Wenn wir also eine Folge von unabhängigen Zufallsvariablen $X_1,X_2,...$vom gleichen Wahrscheinlichkeitsraum haben welche alle dieselbe Verteilung mit Erwartungswert $\mu$ und Varianz $\sigma^2$ haben dann gilt für $n$ die Anzahl Zufallsvariablen und die Summe $S_n=X_1+...+X_n$. Dann hat die Summe approximative die Normalverteilung $N(\mu n, \sigma n)$ wobei $\mu n = n \cdot \mu$ und $\sigma n = \sqrt{n} \cdot \sigma$.

Diese Verteilung kann dann natürlich auch noch standardisiert werden.

$$\frac{S_n -\mu n}{\sqrt{n}\cdot \sigma}\sim N(0,1)$$

## Satz von Moivre-Laplace

Weil eine Binomialverteilte Zufallsvariable $X \sim Bin(n,p)$ als Summe von $n$ Bernoulliverteilte Zufallsvariablen interpretiert werden kann und wir sie mit der Normalverteilung annähern können mit $N(np, \sqrt{np(1-p)})$ können wir ein paar Approximationen machen. Wobei $normcdf(x)$ die Verteilungsfunktion der standardisierten Normalverteilung ist.

Mit dem Satz können wir für $n > \frac{9}{p(1-p)}$ folgendes gut approximieren

$$P(a \leq X \leq b) \approx normcdf(\frac{b-np}{\sqrt{np(1-p)}})-normcdf(\frac{a-np}{\sqrt{np(1-p)}})$$

Genauer wird es dann mit der Stetigkeitskorrektur

$$P(a \leq X \leq b) \approx normcdf(\frac{b+\frac{1}{2}-np}{\sqrt{np(1-p)}})-normcdf(\frac{a-\frac{1}{2}-np}{\sqrt{np(1-p)}})$$

:::note Beispiel Satz von Moivre-Laplace

 Ein fairer Würfel wirf 1000 mal geworfen. Wie hoch ist die Wahrscheinlichkeit, dass wir zwischen 150 und 200 sechs würfeln?

 Genau:

 $$binocdf(200,1000,1/6)-bincdf(149,1000,1/6)=0.9265$$

 Mit Satz von Moivre-Laplace:

 $$normcdf(\frac{200+\frac{1}{2}-\frac{1000}{6}}{\sqrt{1000\cdot \frac{5}{36}}}) - normcdf(\frac{150-\frac{1}{2}-\frac{1000}{6}}{\sqrt{1000\cdot \frac{5}{36}}})=0.9253$$

:::

## Simulation von Zufallsvariablen

Manchmal ist es nur mit grossem Aufwand Wahrscheinlichkeiten zu exakt berechnen. Eine Lösung für dieses Problem ist die Zufallsvariable zu simulieren, also Zahlen zu erzeugen die korrekt verteilt sind und dann schauen, welcher Prozentsatz dieser Zahlen im gesuchten Ereignis liegen. Dank dem Gesetz der grossen Zahlen wird diese Wahrscheinlichkeit genauer mit wachsender Anzahl an Wiederholungen.

In den meisten Programmiersprachen ist der sogenannte lineare Kongruenzgenerator eingebaut welcher ein Pseudozufallszahlengenerator ist. Das heisst er erzeugt die Zahlen nicht wirklich zufällig sondern berechnet sie anhand eines Startwerts, der sogenannte **Seed** (oftmals die Systemzeit).

Für kryptographische Zwecke wie Schlüsselerzeugung sind Pseudozufallszahlengenerator nicht geeinigt, weil mit wenigen Werten kann man die verwendete Parameter berechnen und kann dann die Zufallsvariablen voraus sehen.

### Linearer Kongruenzgenerator

Beim linearen Kongruenzgenerator haben wir ein sogenanntes modul $m >1$ und  einen Anfangswert (der Seed) $x_0$ und zwei weitere Werte $a$ und $b$. Wichtig dabei ist, dass $a,b,x_0 \in \{0,1,...,m-1\}$ sind. Dann können wir eine Zufallszahl wie gefolgt berechnen

$$x_{n+1}=(a\cdot x_n +b) \text{ mod } m$$

Wir erhalten dann einen Wert aus dem endlichen Bereich ${0,1,...m-1}$. Weil der Wertebereich endlich ist gibt es eine Periode. Wenn z.B. $m=12,a=4,b=1,x_0=1$ dann wiederholt sich die Folge schon nach dem dritten Wert.

Der Satz von Knuth besagt, damit die Periodenlänge maximal ist, also $m$ muss folgendes gelten:

- $b$ ist zum Modul $m$ teilerfremd, also $ggt(b,m)=1$.
- Jeder Primfaktor von $m$ teilt $a-1$.
- Wenn $m$ durch 4 teilbar ist, dann muss auch $a-1$ durch 4 teilbar sein.

Durch eine Transformation bekommen wir auch nur noch Werte im Intervall $[a,b]$.

$$z_n=a+(b-1)\frac{x_n}{m}$$

Die Werte $U$ die wir erhalten sind im Intervall $[a,b]$ gleich verteilt.

### Inversionsmethode

Wir können nun unsere Zufallszahlen auf eine bestimmte Verteilung abbilden mit der Inversionsmethode.

Es sei $F$ eine streng monoton steigende Verteilungsfunktion und $U$ eine im Intervall $[0,1]$ gleichverteilte Zufallsvariable dann ist Zufallsvariable $F^{-1}(U)$ verteilt mit der Verteilungsfunktion $F$.
