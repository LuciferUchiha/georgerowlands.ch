# Eigenschaften von Erwartungswerte und Varianz

Es seien $X,Y$ Zufallsvariablen und $a,c \in \mathbb{R}$ dann gelten die folgende Eigenschaften für Erwartungswerte und Varianz

## Erwartungswerte addieren

$$E(X+Y) = E(X)+E(Y)$$

:::note Beispiel Erwartungswerte addieren

Wir werfen eine Münze, solange bis zum ersten Mal Kopf, dann würfeln wir Solange, bis eine 6 kommt.

Wie hoch ist die totale erwartete Anzahl würfe?

$$X \sim Geo(1/2), Y \sim Geo(1/6), E(X+Y)=E(X)+E(Y)=8$$

:::

## Erwartungswert, Skalar multiplizieren

$$E(aX) = aE(X)$$

## Erwartungswert, Skalar addieren

$$E(X+c) = E(X)+c$$

## Erwartungswert, Funktion anwenden

$$E(g(X))=\sum_{x_i \in X}{g(x_i)\cdot P(X=x_i)}$$

Für all Funktionen $g: \mathbb{R} \mapsto \mathbb{R}$

:::note Beispiel Erwartungswert, Funktion anwenden

Ein Computerhändler hat 3 Computer für 500 CHF pro Stück gekauft, die er für 1000 CHF vor Neujahr verkaufen will.

Der händler weiss, dass er alle nicht verkauften Computer nach Neujahr garantiert für 200 CHF an ein Unternehmen verakufen kann.

Der Händler denkte wenn $X$ die Anzahl verkaufte Computer entspricht das er die folgende Verteilung hat.

:::

## Varianz, Skalar addieren

$$V(X+c) = V(X)$$

## Varianz, Skalar multiplizieren

$$V(aX)=a^2V(X)$$
