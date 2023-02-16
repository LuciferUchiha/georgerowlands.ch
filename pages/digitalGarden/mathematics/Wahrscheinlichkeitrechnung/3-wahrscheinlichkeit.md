# Wahrscheinlichkeit

## Laplace-Experimente

Wenn bei einem Zufallsexperiment alle Elementarereignisse die gleiche Wahrscheinlichkeit haben einzutreten, also das alle **gleichmöglich** sind reden wir von einem Laplace-Experiment. Man redet hier auch oft von einer Gleichverteilung.

Bei einer Ergebnismenge $|\Omega|=m$ also mit $m$ gleichmöglichen Elementarereignisse redet man von einem **Laplace-Raum**. Dabei haben alle Elementarereignisse $\omega_i$ die gleiche Wahrscheinlichkeit, sogenannte **Zähldichte**.

$$P(\{\omega_i\}) = p(\omega_i)= \frac{1}{m} \text{ mit }i=1,2,...,m$$

Das heisst die Wahrscheinlichkeit für ein Ereignis $A$ ist dann durch die Gleichung

$$P(A) = \sum_{\omega_i \in A}{p(\omega_i)} = |A| \cdot \frac{1}{m} = \frac{|A|}{m}$$

definiert. Man kann es auch ein wenig ausführlicher definieren als

$$P(A) = \frac{|A|}{|\Omega|} = \frac{\text{Anzahl Ergebnisse wo A eintritt}}{\text{Anzal aller möglichen Ergebnisse}}$$

Eine gute Video Erklärung dazu gibt es auch [hier](https://studyflix.de/statistik/laplace-experiment-1109)

:::note Beispiel Laplace-Experiment würfeln

 Beim Wurf eines Würfels haben alle 6 Augenzahlen die gleiche Wahrscheinlichkeit somit handelt es sich um ein Laplace-Experiment.

 Für jedes Elementarereignis gilt also $p(\omega_i) = \frac{1}{6}$

 Für das Ereignis "gerade Augenzahl" also $A = \{2,4,6\}$ ist die Wahrscheinlichkeit somit

 $$P(A)=\frac{3}{6} = \frac{1}{2} = 50\%$$

:::

## Wahrscheinlichkeitsaxiome von Kolmogoroff

Wir definieren nun den Begriff Wahrscheinlichkeit ein wenig genauer, dazu verwenden wir ein paar Axiome (Grundsätze). Die Wahrscheinlichkeit ist eine Funktion $P$ welches jedem Ereignis $E \subseteq \Omega$ eine Zahl $P(E) \in [0,1]$ zuorndet.

$$P: 2^{\Omega} \mapsto [0,1]$$

### Axiom 1

$P(E)$ ist eine nicht-negative Zahl, die höchstens gleich 1 ist

$$0 \leq P(E) \leq 1$$

### Axiom 2

Für das sichere Ereignis, $\Omega$ gilt

$$P(\Omega)=1$$

### Axiom 3

Für paarweise sich gegenseitig ausschliessende Ereignisse $A_1,A_2,A_3,...$ z.B. die Elementarereignisse gilt

$$P(A_1 \cup A_2 \cup A_3, ...) = P(A_1)+P(A_2)+P(A_3)+ ...$$

auch der sogenannte Additionsatz für sich gegenseitig ausschliessende Ereignisse.

## Folgerungen aus den Wahrscheinlichkeitsaxiome

Für das zu $A$ komplementäre Ereignis $\overline{A}$ gilt

$$P(\overline{A})= 1-P(A)$$

Für das unmögliche Ereignis, $\emptyset$ gilt

$$P(\emptyset)=0$$

weil es das komplementäre Ereignis zum sicheren Ereignis $\Omega$ ist.

Für sich 2 gegenseitig ausschliessende Ereignisse $A$ und $B$ gilt

$$P(A \cup B)= P(A) + P(B)$$

## Additionssatz

Für sich 2 gegenseitig ausschliessende Ereignisse $A$ und $B$ gilt anhand des 3. Axiom

$$P(A \cup B)= P(A) + P(B)$$

Somit haben wir die Wahrscheinlichkeit für wenn $A$ oder $B$ eintritt, da sie nicht gleichzeitig eintreten können weil sie sich gegenseitig ausschliessen.

Sind aber $A \cup B \neq \emptyset$, also schliessen sie sich nicht gegenseitig aus, dann gilt der folgende allgemeine Additionssatz

$$P(A \cup B)= P(A) + P(B) - P(A \cap B)$$

Dieser kann auch für sich gegenseitig ausschliessende Ereignisse verwendet werden da dann $P(A \cap B) = P(\emptyset)= 0$ und wir somit 0 subtrahieren.

## Bedingte Wahrscheinlichkeit

Oftmals intressiert uns die Wahrscheinlichkeit für das Eintreten des Ereignisses $B$ unter der **Voraussetzung** oder **Bedingung**, dass $A$ bereits eingetreten ist.

Wir nennen diese Wahrscheinlichkeit die **bedingte Wahrscheinlichkeit von B unter (Der Bedingung) A** und definieren sie als

$$P(B | A)= {P(A \cap B) \over P(A)}$$

wobei $P(A) \neq 0$

Eine gute Video Erklärung dazu gibt es auch [hier](https://studyflix.de/statistik/bedingte-wahrscheinlichkeit-1110)

:::note Beispiel bedingte Wahrscheinlichkeit würfeln

Wir würfeln mit 2 Würfeln und intressieren uns für die Würfe wo die Augensumme 8 ist. Die Frage die wir uns nun stellen ist was die Wahrscheinlichkeit, bei so einem Wurf ist, dassbeide Augenzahlen gerade sind.

- $A = \text{Die Augensumme ist 8}$
- $B = \text{Die Augenzahlen beider Würfel sind gerade}$

**1. Lösungsweg ohne Formel**

Das Ereignis $A$ wird durch die 5 Elementarereignisse $(2,6),(3,5),(4,4),(5,3),(6,2)$ gegeben.
Daraus sehen wir, dass bei 3 beide Zahlen gerade sind. Also haben wir

$$P(B | A) = {3 \over 5}$$

**2. Lösungsweg mit Formel**

 Wir wissen das es 36 gleichmögliche Elementarereignisse gibt. Daraus können wir zählen, dass $P(A)= {5 \over 36}$ und $P(A \cap B)={3 \over 36}$.

 $$P(B | A) ={{3 \over 36} \over {5 \over 36}} = {3 \over 5}$$

:::

## Multiplikationssatz

Wenn wir die Definitionsgleichung der bedingten Wahrscheinlichkeit nach $P(A \cap B)$ auflösen erhalten wir den folgenden Multiplikationssatz

$$P(A \cap B)=P(A) \cdot P(B|A)$$

weil $A \cap B = B \cap A$ gilt daher auch

$$P(A) \cdot P(B|A) = P(B) \cdot P(A|B)$$

:::note Beispiel Multiplikationssatz

 In einer Urne sind 6 Kugeln, 4 weiss, 2 schwarz. Wir entnehmen der Urne 2 Kugeln nach einander Ohne zurückzulegen. Mit welcher wahrscheinlichkeit sind beide weiss?

- $A=$ erste Kugel ist weiss
- $B=$ zweite Kugel ist weiss

 $P(A)= {4\over 6} = {2 \over 3} , P(B|A)= {3\over 5}$

 Uns intressiert nun die Wahrscheinlickeit, dass $A$ und $B$ gleichzeitig eintreten also $P(A \cap B)$ welches wir mit dem Multiplikationssatz berechnen können

 $$P(A \cap B) = {{2\over 3} \cdot {3 \over 5}} = {2 \over 5}$$

:::

## Stochastische unabhängigkeit

Es kann sein, dass die Wahrscheinlichkeit eines Ereignisses $B$ von einem anderen Ereignis $A$ abhängen kann. Dies führte uns zu der bedingten Wahrscheinlichkeit.

Wenn dies jedoch nicht der Fall ist, also wenn die Ereignisse nicht von einander abhängen, dann bezeichnen wir solche Ereignisse als **stochastisch unabhängig** und somit gilten dann

$$P(A | B) = P(A) \text{und} P(B | A) = P(B)$$

Aus dem Multiplikationssatz wird dann

$$P(A \cap B)=P(A) \cdot P(B|A)=P(A) \cdot P(B)$$

Wir können also definieren das zwei Ereignisse stochastisch unabhängig sind wenn

$$P(A \cap B)= P(A) \cdot P(B)$$

:::note Beispiel stochastische unabhängige Münzenwurfe

 Eine Münze wird 3x geworfen und wir betrachten die folgende Ereignisse:

- $A=$ Zahl beim 1. Wurf
- $B=$ Zahl beim 2. Wurf
- $C=$ Kopf beim 3. Wurf

 Sie sind alle stochastisch unabhängig, den sie sind völlig unabhängig von einander.

:::

## Mehrstufige Zufallsexperimente

Bei einem Mehrstufige Zufallsexperimente werden mehrere Zufallsexperimente nacheinander ausgeführt. Diese werden oftmals durch Baumdiagramme (Ereignisbäume) dargestellt. Dabei unterscheidet man zwischen Endergebnisse und Zwischenergebnisse.

Wir definiren noch folgende Regeln:

1. Die Wahrscheinlichkeiten entlang eines Pfades werden miteinander multipliziert.
2. Führen mehrere Pfade zum gleichen Endergebnis, so addiert man ihre Wahrscheinlichkeiten.

Eine gute Video Erklärung dazu gibt es auch [hier](https://studyflix.de/statistik/baumdiagramm-1107)

:::note Beispiel Mehrstufiges Zufallsexperiment

 In einer Urne befinden sich 6 Kugeln, 2 weiss und 4 schwarz. Wir entnehmen nacheinander ganz zufällig 2 Kugeln ohne zurücklegen, somit 2 Stufen und stellen uns die Frage mit welcher Wahrscheinlichkeit erhalten wir 2 gleichfarbige $A$ oder 2 verschiedenfarbige Kugeln $B$?

 **1. Stufe:**

- $P(W) = {2 \over 6} = {1 \over 3}$
- $P(S) = {4 \over 6} = {2 \over 3}$

 **2. Stufe:**
 Nach der 1. Ziehung sind nurnoch 5 Kugeln in der Urne, entweder wurde eine schwarze oder eine weisse entzogen.
 Falls es eine Weisse war haben wir:

- $P(W|W) = {1 \over 5}$
- $P(S|W) = {4 \over 5}$

 Falls es ein Schwarze war haben wir:

- $P(W|S) = {2 \over 5}$
- $P(S|S) = {3 \over 5}$

 Somit ergeben sich follgende Resultate:

 Die Pfade wo es gleichfarbige Kugeln sind $P(A)=P(WW) + P(SS) = {1 \over 3} \cdot {1 \over 5} + {2 \over 3} \cdot {2 \over 5} = {7 \over 15}$

 Die Pfade wo es verschiedenfarbige Kugeln sind $P(A)=P(WS) + P(SW) = {1 \over 3} \cdot {4 \over 5} + {2 \over 3} \cdot {2 \over 5} = {8 \over 15}$

 ![mehrstufigeZufallsexperimente](/maths/mehrstufigeZufallsexperimente.png)

:::

## Totale Wahrscheinlichkeit

Die **totale Wahrscheinlichkeit** für das eintreten des Ereignisses $B$ wobei $A_i$ die möglichen Zwischenereignisse auf dem weg zum Ereignis $B$ ist

Eine gute Video Erklärung dazu gibt es auch [hier](https://studyflix.de/statistik/satz-der-totalen-wahrscheinlichkeit-1111) und [hier](https://www.youtube.com/watch?v=0iJSW0VNddo)

$$P(B)= \sum_{i=1}^{n}{P(A_i)\cdot P(B|A_i)}$$

## Bayes' theorem

Unter der Voraussetzung, dass das Ereignis $B$ bereits eingetreten ist, gilt dann für die Wahrscheinlichkeit, dass dieses Ereignis über das Zwischenereignis $A_j$ die Bayessche Formel

$$P(A_j|B)= {P(A_j \cup B) \over P(B)} = {P(A_j) \cdot P(B | A_j) \over P(B)}$$

Eine gute Video Erklärung dazu gibt es auch [hier](https://studyflix.de/statistik/satz-von-bayes-1113)
und [hier](https://www.youtube.com/watch?v=wUDxQFbXqjA)

## Geburtstagsparadox

 Das Geburtstagsparadox ist ein Beispiel dafür, dass bestimmte Wahrscheinlichkeiten intuitiv häufig falsch geschätzt werden.

 Wir stellen uns die Frage "Was ist die Wahrscheinlichkeit, dass mindestens zwei Personen am selben Tag Geburtstag haben in einer Gruppe von $k$ personen?".

 Um dieses Problem anzugehen schauen wir zuerst an was die Wahrscheinlichkeit ist, dass die personen **nicht** am selben Tag geburtstag haben:

 Bei 2 Leuten: ${365 \over 365} \cdot {364 \over 365}$
 Bei 3 Leuten: ${365 \over 365} \cdot {364 \over 365}\cdot {363 \over 365}$
  etc.
  Diese Zahl wird näher zu 0 und nun können wir unsere Frage beantworten
  
$$P(gleich)=1-P(ungleich) \Leftrightarrow P(A)=1- \frac{365 \cdot (365-1)\cdot...\cdot (365-n+1)}{365^n}$$
  
## Bernoulli-Experiment

 Ein Bernoulli-Experiment ist ein Zufallsexperiment mit genau 2 möglichen Ergebnisse, Treffer oder nicht Treffer.

 Ein häufiges Beispiel dafür ist das Werfen eines Würfels. Wir intressieren uns nur ob wir eine 6 bekommen beim Würfeln. Das heisst wenn wir eine 6 würfeln betrachten wir es als ein Treffer alle andere Ergebnisse fassen wir zusammen als kein Treffer.

 Anders wie bei einem Laplace-Experiment müssen wie man oben sieht die Wahrscheinlichkeiten der Ergebnisse gleich sein. Im obigen Beispiel ist die Wahrscheinlichkeit für ein Treffer $\frac{1}{6}$.
