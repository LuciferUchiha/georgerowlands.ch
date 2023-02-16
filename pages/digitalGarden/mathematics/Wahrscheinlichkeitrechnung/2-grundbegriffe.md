# Grundbegriffe

## Zufallsexperiment

Ein **Zufallsexperiment** ist ein Experiment welches beliebig oft wiederholt werden kann und zu einem oder mehrere Ergebnisse führt, welche sich gegenseitig ausschliesen. Beim durchführen eines Zufallsexperiment lässt sich ein Ergebnis nicht voraussagen, sondern ist zufallsbedingt.

Ein paar einfache Beispiele für ein Zufallsexperiment wären eine Münze oder einen Würfel zu werfen. Genau so ist das ziehen einer Kugel aus einer Urne zufallsbedingt sofern man nicht in die Urne schauen kann und eine Kugel auswählt.

## Elementarereignis

Die möglichen sich aber gegenseitig ausschlissende Ergebnisse heissen **Elementarereignisse**. Mit **gegenseitig ausschlissend** heisst, dass sie nicht gleichzeitig passieren können z.B. kann eine Münze nicht gleichzeitig auf Kopf und Zahl landen. Elementarereignisse werden beschreiben mit

$$\omega_1,\omega_2,\omega_3,...$$

## Ergebnismenge

Die **Ergebnismenge** beschreibt alle möglichen Ergebnisse, und ist somit die Menge aller Elementarereignisse und wird geschrieben als $\Omega$. Dabei wird noch zwischen **endlichen** und **abzählbar-unendliche** Ergebnismengen unterschieden.

### Endliche Ergebnismenge

Enthält nur endlich viele Elementarereignisse

$$\Omega = \{\omega_1,\omega_2,\omega_3,...\omega_n\}$$

:::note Beispiel endliche Ergebnismenge

Beim Wurf eines Würfels sind 6 Augenzahlen möglich somit ist $\omega_i = i$ für $i=(1,2,...,6)$ und

$$\Omega = \{1,2,3,4,5,6\}$$

:::

### Abzählbar-unendliche Ergebnismenge

Enthält unendlich viele Elementarereignisse, die wir aber wie die Natürlichen Zahlen durchnummerieren können.

:::note Beispiel abzählbar-unendliche Ergebnismenge

 Wir werfen einen Würfel so lange bis wir zum ersten Mal eine 6 bekommen. Theoretisch kann dies unendliche lange dauern, aber wir können zählen bei welchem Wurf wir zum ersten Mal die 6 bekommen.

 Also haben wir $\omega_i = i$ für $i=(1,2,...)$ und

 $$\Omega = \{1,2,3,...\}$$

:::

## Ereignis

Ein **Ereignis** ist eine Zusammenfassung von Elementarereignisse. Anderst gesagt ist ein Ereignis eine Teilmenge der Ergebnismenge $A \subseteq \Omega$ .

:::note Beispiel Ereignisse beim würfeln

 Wir haben festgelegt das beim würfeln eines sechseitigen Würfels $\Omega = \{1,2,3,4,5,6\}$ ist. Wir können nun z.B. foglende Teilmenegen konstruieren.

- Würfeln einer geraden Zahl: $A=\{2,4,6\}$
- Würfeln einer durch 3 teilbare Zahl: $B=\{3,6\}$
- Würfeln einer Zahl grösser als 2: $C=\{3,4,5,6\}$

:::

### Unmögliches Ereignis

Entspricht ein Ereignis der leeren Menge $\emptyset$ so redet man vom sogenannten **unmöglichem Ereignis**, welches nie eintreten wird.

### Sicheres Ereignis

Enthählt ein Ereignis alle Elementarereignisse der Ergebnismenge also ist $A=\Omega$ so redet man vom sogenannten **sicherem Ereignis**, welches garantiert immer eintreten wird.

## Verknüpfung von Ereignissen

Wie wir gesehen haben sind Ereignisse eigentlich nur Mengen, dass heisst wir können auch Mengenoperationen auf sie durchführen.

### Vereinigung

Bei der Vereinigung von Ereignissen $A \cup B$ kann man aussagen, dass entweder tritt $A$ oder $B$ ein oder $A$ und $B$ treten gleichzeitig ein.

### Durchschnitt

Der Durchschnitt der Ereignisse $A \cap B$ besagt, dass $A$ und $B$ gleichzeitig eintreten.

### Kompliment

Das Kompliment zu $A$ also $A^c$ was aber auch oftmals als $\overline{A}$ geschrieben wird, besagt, dass A nicht eintritt.

:::note Beispiel verknüpfung von Ereignissen beim würfeln

 Wenn wir beim würfeln sagen, dass wir die Ereignisse "würfeln einer geraden Zahl" und "würfeln einer ungeraden Zahl" haben, also
 $A=\{2,4,6\}, B=\{1,3,5\}.

 Wir können nun also sagen

- $\overline{A}=B$
- $\overline{B}=A$
- $A \cup B = \Omega$
- $A \cap B = \emptyset$

Was auch alles Sin macht wenn man es sich überlegt.

:::

## De Morgan's Laws

De Morgan's Laws funktioniert auch mit Ereignisse

$$\begin{align*}
 \overline{A \cup B} &= \overline{A} \cap \overline{B} \\
 \overline{A \cap B} &= \overline{A} \cup \overline{B}
\end{align*}$$
