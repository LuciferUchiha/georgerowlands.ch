# Stetige Zufallsvariablen

Eine stetige Zufallsvariable kann unendlich viele nicht Abzählbare Werte annehmen. Stetige Zufallsvariablen entstehen meist durch einen Messvorgang. Unabhängig von der Messgenauigkeit kann eine stetige Zufallsvariable innerhalb eines Intervalls unendlich viele Werte annehmen aber einen genauen Wert zu messen ist nicht wirklich möglich.

Die Wahrscheinlichkeit, dass eine stetige Zufallsvariable $X$ einen exakten Wert $x_i$ annimmt ist gleich 0. Z.B. ist die Wahrscheinlichkeit das eine Person 180cm gross ist 0, weil er könnte auch 180.000000000000001cm gross sein. Weshalb es mehr Sin ergibt, dass $X$ einen Wert in einem Interval $[a,b]$ annimmt Z.B. $[179.5,180.5]$

mehr dazu findest du [hier](https://www.youtube.com/watch?v=g9acHn8zSr8)

## Dichtefunktion

Die Wahrscheinlichkeiten von Stetigen Zufallsvariablen werden durch die Fläche unter der Dichtefunktion $f(x)$ für alle $x$ der Zufallsvariable $X$ beschrieben

$$P(a \leq X \leq b) = \int_{a}^{b}{f(x) dx}$$

![stetigeDichtefunktion](/maths/stetigeDichtefunktion.png)

Die Gesamtfläche unter der Dichtefunktion muss gleich 1 sein sonst ist sie nicht normalisiert

$$P(-\infty < X < \infty)=\int_{-\infty}^{\infty}{f(x) dx}=1$$

Weil die Wahrscheinlichkeit von einem genauen Wert 0 ist haben abgeschlossene und offene Intervalle dieselben Wahrscheinlichkeiten

$$P(a \leq X \leq b)=P(a < X \leq b)=P(a \leq X < b)=P(a < X < b)$$

## Erwartungswert

$$E(X)=\int_{-\infty}^{\infty}{x \cdot f(x) dx}$$

## Varianz

$$E(X)=\int_{-\infty}^{\infty}{(x-E(X))^2 \cdot f(x) dx}$$
Oder Kurz geschrieben genau gleich wie bei diskreten Zufallsvariablen
$$E(X^2)-E(X)^2$$

## Standardabweichung

Genau gleich wie bei diskreten Zufallsvariablen

$$\sigma(X)=\sqrt{V(X)}$$

## Verteilungsfunktion

Die Verteilungsfunktion ist

$$F(x)=P(X \leq x)=\int_{-\infty}^{x}{f(y) dy}$$

Und hat die follgende Eigneschaften

- $P(a \leq X \leq b)=F(b)-F(a)$
- $F'(x)=f(x)$ wenn $f$ stetig ist.
- $\lim_{x\to\infty} F(x)=1$
- $\lim_{x\to - \infty} F(x)=0$

## Stetige Verteilungen

Nun schauen wir uns ein paar Verteilungen an die häufig vorkommen wenn man mit stetigen Zufallsvariablen arbeitet.

### Stetige Gleichverteilung

Die **Stetige Gleichverteilung** wird auch oft **Uniformverteilung** genannt. Sie hat auf dem Intervall $[a,b]$ eine konstante Wahrscheinlichkeitsdichte, dass heisst das alle Teilintervalle gleicher Länge dieselbe Wahrscheinlichkeit besitzen.

- Wir schreiben dann $X \sim U[a,b]$
- Die Dichte von $X$ ist $f(x)=\frac{1}{b-a}$ wobei $a\leq x\leq b$ Dies kommt davon weil die Dichte normalisiert ist und die Fläche 1 ergeben muss.
- $E(X) = \frac{a+b}{2}$
- $V(X) = \frac{1}{12}(b-a)^2$

![Stetige_Gleichverteilung_Dichte](/maths/stetigeGleichverteilungDichte.png)

In Matlab haben wir die Funktionen:

- Dichte: $unifpdf(x,a,b)$
- Verteilungsfunktion $unifcdf(x,a,b)$

mehr dazu findest du [hier](https://www.youtube.com/watch?v=SzvXAJmVPdI) und [hier](https://studyflix.de/mathematik/stetige-gleichverteilung-1081)

:::note Beispiel stetige Gleichverteilung

 Eine Person kommt zu einem zufälligen Zeitpunkt zum Bahnhof. Der Zug fährt einmal pro Stunde. Wie hoch ist die Wahrscheinlichkeit, dass man höchstens 10 Minuten warten muss? Was ist die erwartete Wartezeit im Durchschnitt?

 $X$: Wartezeit, Dann ist $X \sim U[0,60]$

 Wir erhalten also: $f(x)=\frac{1}{b-a}=\frac{1}{60}$ und somit dann $F(X)=\frac{1}{60}x=\frac{x}{60}$

 $$P(X \leq 10)=\int_{0}^{10}{f(x) dx}=\int_{0}^{10}{\frac{1}{60} dx} = \frac{x}{60} \Big|_{0}^{10}=\frac{10}{60}-\frac{0}{60}=\frac{1}{6}$$

 Und die erwartete Wartezeit ist $E(X)=\frac{a+b}{2}=30$ Minuten

:::

### Normalverteilung

Die Normalverteilung oder auch oft Gauss-verteilung oder Glockenkurve genannt, ist eines der wichtigsten stetigen Verteilungen.

Die Normalverteilung besteht aus 2 Parametern, der Erwartungswert $\mu$ und die Standardabweichung $\sigma$. Desto Kleiner $\sigma$ desto enger ist die Glockenkurve.

- Wir schreiben dann $X \sim N[\mu,\sigma]$
- Die Dichte von $X$ ist $f_{\mu,\sigma}(x)= \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ wobei $a\leq x\leq b$
- $E(X) = \mu$
- $V(X) = \sigma^2$

![normalVerteilungGraph](/maths/normalVerteilungGraph.png)

In Matlab haben wir die Funktionen:

- Dichte: $normpdf(x,\mu,\sigma)$
- Verteilungsfunktion $normcdf(x,\mu,\sigma)$

mehr dazu findest du [hier](https://studyflix.de/mathematik/normalverteilung-1089)

:::note Beispiel Normalverteilung

 Der Intelligenzquotient (IQ) ist normalverteilt und so festgelegt, dass $\mu=100$ und $\sigma=15$. Wie hoch ist die Wahrscheinlichkeit, dass eine zufällig ausgewählte Person einen IQ zwischen 90 und 110 hat oder grösser als 150.

 Es sei X der IQ der Person dann

 $X \sim N(100,15)$

 $P(90 \leq X \leq 110)=P(X \leq 110) - P(X \leq 90) = normcdf(110,100,15)-normcdf(90,100,15) \approx 50$%

 $P(X \geq 150)= 1 - P(X < 150) = 1 - normcdf(150,100,15) \approx 0.04$%

:::

#### Standardisierung der Normalverteilung

Mit Standardisierung bezeichnen wir die transformation einer Normalverteilten Zufallsvariable $X$, zu einer Zufallsvariable $Z$ welches den Erwartungswert $E(Z)=0$ und die Varianz $V(Z)=1$ besitzt. Dies machen wir damit wir verschiedene Zufallsvariablen besser vergleichen können und damit wir auch schneller rechnen können.

Zuerst zentrieren wir die Zufallsvariable, dies machen wir indem wir von allen ihre Elementarereignisse den Erwartungswert $\mu$ abziehen. Mit Zentrieren ist hier gemeint das wir den Gipfel der funktion $f(x)$ bei der Nullstelle der x-Achse wollen.

Danach Dividieren wir die Differenz $X - \mu(X)$ durch die Standardabweichung $\sigma (X)$.

$$Z=\frac{X-\mu(X)}{\sigma(X)}$$

Z.B. können wir nun $X \sim N(\mu,\sigma)$ zu $Z \sim N(0,1)$ umwandeln mit $Z = \frac{X-\mu(X)}{\sigma(X)}$

mehr dazu findest du [hier](https://www.youtube.com/watch?v=YmEGmn5C4tg&t=11s)

#### Quantile

Oftmals haben wir einen Wert $\alpha \in [0,1]$ gegeben. Und wir suchen nun den Wert $z_\alpha$ wofür $P(X \leq z_\alpha)=\alpha$.

- Wenn $\alpha = 0.5$ dann reden wir vom Median, auch Zentralwert, **Erwartungswert** und der Modus. Eine Kennzahl dafür, wo sich die "Mitte" einer Wahrscheinlichkeitsverteilung befindet.
- Wenn $\alpha = 0.25 \text{oder} 0.75$ reden wir von einer Quartile
- Mit dem Perzentil schneiden wir $[0,1]$ in 100 Teile was equivalent ist zu den Prozentanzahlen.

In Matlab haben wir die Funktionen:

- $norminv(\alpha,\mu,\sigma)$ wobei $norminv$ English ist und für "normal inverse" steht

![quantileGraph](/maths/quantileGraph.png)

mehr dazu findest du [hier](https://www.youtube.com/watch?v=KdQLNiCOa0U)

:::note Beispiel Quantile

 Der Intelligenzquotient (IQ) ist normalverteilt und so festgelegt, dass $\mu=100$ und $\sigma=15$. Eine gewisse Schulform ist für die tiefsten 5% gedacht. Ab welchem IQ sollte man an diese Schule gehen?

 $X \sim N(100,16)$

 Wir suchen also $z_\alpha$ mit $P(X \leq z_\alpha) = 0.05$

 Dies bekommen wir mit der Matlab funktion $norminv(0.05,100,15)=75.33$

:::

#### Sigma-Regeln

Für $X \sim N(\mu,\sigma)$ gilt

1. $P(|X-\mu| \leq \sigma) \approx 68.3$%
2. $P(|X-\mu| \leq 2\sigma) \approx 95.5$%
3. $P(|X-\mu| \leq 3\sigma) \approx 99.7$%

Was bedeutet, dass ein Wert einer normalverteilten Zufallsvariable mit der Wahrscheinlichkeit 68% maximal um $\pm \sigma$ vom Erwartungswert $\mu$ abweicht.

![sigmaRegelnGraph](/maths/sigmaRegelnGraph.png)

mehr dazu findest du [hier](https://www.youtube.com/watch?v=OmmODKdYLSI)

### Exponentialverteilung

Die Exponentialverteilung beschreibt zufällige Lebensdauern von Geräten oder Wartezeiten auf zufällige Ereignisse.

- Lebensdauer einer Glühbirne
- Wartezeit auf nächstes Erdbeben

Die Exponentialverteilung und Poisson-Verteilung haben eine enge Beziehung mit einander.

$$Anzahl \sim Poi(\lambda) \leftrightarrow Zwischenankunftszeit \sim Exp(\lambda)$$

- Wir schreiben dann $X \sim Exp(\lambda)$
- Die Dichte von $X$ ist $F'(x)= \lambda e^{-\lambda x}$
- Die Verteilung von $X$ ist $F(x)= 1 - e^{-\lambda x}$
- $E(X) = \frac{1}{\lambda}$
- $V(X) = \frac{1}{\lambda^2}$

In Matlab haben wir die Funktionen:

- Dichte: $exppdf(x,1/\lambda)$
- Verteilungsfunktion $expcdf(x,1/\lambda)$

mehr dazu findest du [hier](https://studyflix.de/mathematik/exponentialverteilung-1088)

:::note Beispiel Exponential-Verteilung

 In einem Geschäft kommen im Schnitt 20 Kunden pro Stunde.

 1. Wie hoch ist die Wahrscheinlichkeit, dass mehr als 30 Kunden in einer Stunde kommen? $X \sim Poi(20),\,P(X > 30)=1-poisscdf(30,20)$

 2. Wie hoch ist die Wahrscheinlichkeit, dass man weniger als 5 Minuten auf den ersten Kunden warten muss? $T \sim Exp(20),\,P(T \leq \frac{1}{12})=expcdf(1/12,1/20)$

:::

#### Gedächtnislosigkeit

Die Exponential-Verteilung hat die spezielle Eigenschaft, dass sie kein Gedächtnis hat.

Was so viel heisst, wie wenn ein Gerät mit einer exponential verteilten Lebensdauer $X$ während $t$ Stunden gelaufen ist, so ist die Wahrscheinlichkeit, dass es weitere $h$ Stunden läuft gleich gross, wie wenn ein neues Gerät $h$ Stunden läuft. Dies können wir mit ein wenig Mathe und bedingte Wahrscheinlichkeiten auch beweisen

$$P(X \geq t + h | X \geq t) = \frac{P(X \geq t + h \cup X \geq t)}{X \geq t} = \frac{P(X \geq t + h)}{X \geq t}$$
$$=P(X \geq h)=\frac{e^{-\lambda (t+h)}}{e^{-\lambda t}}=e^{-\lambda h}=P(X \geq h)$$
