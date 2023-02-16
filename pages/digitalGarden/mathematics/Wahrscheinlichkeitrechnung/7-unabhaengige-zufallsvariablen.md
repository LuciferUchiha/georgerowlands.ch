# Unabhängige Zufallsvariablen

## Unabhängigkeit von Zufallsvariablen

Eine unendliche Folge von Zufallsvariablen heisst stochastisch unabhängig, wenn jede endliche Teilfolge davon stochastisch unabhängig ist.

$$P(X_1 \in A_1,...X_n \in A_n)=P(X_1 \in A_1) \cdot ... \cdot P(X_n \in A_n)$$

:::note Beispiel Abhängigkeit von Zufallsvariablen

 Wir würfeln mit einem fairen Würfel dreimal.
 Die Zufallsvariable $X$ zählt die Anzahl an gewürfelten Einsen.
 Die Zufallsvaraible $Y$ zählt die Anzahl an Vieren in den ersten 2 Würfe.

Dann sind $X$ und $Y$ nicht stochastisch unabhängig, weil

 $$P(X=3,Y=2)=0 \neq P(X=3)\cdot P(Y=2)$$

:::

:::note Beispiel Unabhängigkeit von Zufallsvariablen

 Person A kommt zu einem zufälligen Zeitpunkt zwischen 12:00 und 12:45, Person B unabhängig davon zwischen 12:15 und 13:00 in ein Café.

- X: Ankunftszeit von Person A $X \sim U[0,45]$
- Y: Ankunftszeit von Person B $Y \sim U[15,60]$

 $$P(X\leq 30,Y\leq 30)=$$

:::

### Faltung

#### Diskrete Zufallsvariablen

Es seien $X, Y$ unabhängige diskrete Zufallsvariablen und Zähldichten $f_X, f_Y$ Dann hat die Summe $X+Y$ die Zähldichte

$$f_{X+Y}(z)=\sum_{x_i \in X}{f_X(x_i) \cdot f_y(z-x_i)}$$

Daraus können wir auch folgendes ableiten

$$X\sim Poi(\lambda_1), Y \sim Poi(\lambda_2) = X+Y \sim Poi(\lambda_1 + \lambda_2)$$

$$X\sim Bin(n_1,p), Y \sim Bin(n_2,p) = X+Y \sim Bin(n_1+n_2,p)$$

#### Stetige Zufallsvariablen

$$f_{X+Y}(z)=\int_{x_i=-\infty}^{\infty}{f_X(x_i) \cdot f_y(z-x_i) dx}$$

##### Additionstheorem Normalverteilung

Es seien $X_1, X_2,...,X_n$ unabhängige, normal verteilte Zufallsvariablen eines Zufalls-experimentes mit Erwartungswerten $\mu_i$ und Standardaweichungen $\sigma_i$, mit $a_i, a_2,...,a_n \in \mathbb{R}$ dann ist

$$Y=a_i X_1 + a_2 X_2 +...+a_n X_n$$

mit dem Erwartungswert $a_i \mu + a_2 \mu +...+a_n \mu$ und die Varianz  $a_i^2 \sigma^2 + a_2^2 \sigma^2 +...+a_n^2 \sigma^2$
