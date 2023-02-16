# Numerische Integrationsmethoden

Die numerische Integration wird genutzt, wenn sich eine Stammfunktion nicht durch elementare Funktionen ausdrücken lässt, die numerische Auswertung der Stammfunktion zu komplex oder zeitaufwendig ist. In diesen Fällen verwendet man spezielle Näherungsverfahren, die sog. numerische Integration.

## Trapezformel

Wir setzen voraus, dass die stetige Funktion $f(x)$ im Integrationsintervall $a\leq x \leq b$ oberhalb der x-Achse verläuft damit wir es uns besser vorstellen können jedoch muss dies nicht der Fall sein um die Trapezformel zu verwenden.

Wir zerlegen zuerst das Integrationsintervall in $n$ Teilintervalle gleicher Länge

$$h = \frac{b-a}{n}$$

Die Punkte der Teilintervalle auf der x-Achse nennen wir Stützstellen.

$$x_0=a,\,x_1=x_0+h,...,x_k=x_0+k\cdot h,...,x_n=b$$

Die Funktionswerte der Stützstellen nennen wir Stützwerte.

$$y_k=f(x_k)=f(x_0+k\cdot h)=f(a+k\cdot h)\text{ mit } k=0,1,...,n$$

Mit der Formel für den Flächeninhalt eines geometrischen Trapez $A=\frac{a+b}{2}h$ können wir die Fläche des ersten Trapezes berechnen $A_1=\frac{y_0+y_1}{2}h$ und genau so dann die restlichen Flächen der Trapeze

$$A_n=\frac{y_{n-1}+y_n}{2}h$$

![trapezFlaecheFormel](/maths/trapezFlaecheFormel.png)
![trapezFormel](/maths/trapezFormel.png)

Die Summe aller Trapezflächen geben dann eine gute Näherung für den gesuchten Flächenwert. Welches zu einer ziemlich kurzen Formel zusammen gezogen werden kann indem man die Brüche gleichnamig macht und ein wenig ausklammert und faktorisiert.

$$
\begin{align*}
\int_{a}^{b}{f(x)\,dx}&\approx A_1+...+A_n \\
&= \frac{y_0+y_1}{2}h + \frac{y_1+y_2}{2}h +... + \frac{y_{n-1}+y_n}{2}h \\
&= (y_0+2y_1+2y_2+...+2y_{n-1}+y_n)\frac{h}{2} \\
&= (\frac{1}{2}\sum{}_1 + \sum{}_2)h
\end{align*}
$$

wobei $\sum{}_1=$ Die Summe der beiden äusseren Stützwerte $(y_1,y_n)$
und $\sum{}_2=$ Die Summe der inneren Stützwerte $(y_2,...y_{n-1})$.

Wenn $n \to \infty$ dann liefert die Trapezformel den exakten Integralwert.

Mehr dazu kann man auch [hier](https://www.youtube.com/watch?v=BZbzdsvpc3c) finden.

## Simpsonsche Formel

[link](https://www.youtube.com/watch?v=N0kFSTDvDcw)
[link](https://www.youtube.com/watch?v=VDfWERtVN9Y)
