# Folgen

## Reele Folge

Ist eine Funktion $a:N \mapsto R$, $n \mapsto a_n$
**Index** = $n$
**Glieder** = Die reelen Zahlen $a_n$
Jedem Index $n$ wird eine reele Zahl (Folgenglied) $a_n$ zugeordnet.
**Aufzählende Darstellung**: Alle Folgenglieder der Reihe nach. $a_1, a_2,a_3, ...$
**Explite Darstellung**: Ein Bildungsgesetz welches ein Term mit $n$ ist.

:::note

$a_n={n-1 \over n}$
$a_1=0, a_2={1\over2}, a_3={2\over3}, ...$

:::

**Rekursive Darstellung** Eine Rechenvorschrift, wie aus den vorangegangenen Folgegliedern das n-te Folgeglied berechnet werden kann.
Bei rekursiver Folgen ist **k-te Ordnung** das tiefste $a_{n-k}$ in der Rechenvorschrift. Es müssen bis $a_k$ schon vorgegben werden.

:::note

**Heronverfahren**
$a_1$ = Näherungswert für $\sqrt b$

$$a_n={1\over2}(a_{n-1}+{b\over a_{n-1}})$$
ist durch eine rekursive Darstellung 1. Ordnung beschrieben.

:::

:::note

**Fibonacci-Folge**
$a_1 = 1, a_2=1$

$$a_n=a_{n-1}+a_{n-2}$$
ist durch eine rekursive Darstellung 2. Ordnung beschrieben.

:::

## Eigenschaften von Folgen

### Monotonie

Eine Folge $a_n$ heisst
**Monoton wachsend**, falls für alle Folgenglieder gilt $a_n \leq a_{n+1}$
**Strng Monoton wachsend**, falls für alle Folgenglieder gilt $a_n < a_{n+1}$
**Monoton fallend**, falls für alle Folgenglieder gilt $a_n \geq a_{n+1}$
**Strng Monoton fallend**, falls für alle Folgenglieder gilt $a_n > a_{n+1}$

:::note

$a_n=n^2$ ist streng monoton wachsend.
$b_n={1\over n}$ ist streng monoton fallend.
$c_n=(-1)^n$ ist weder monoton fallend not wachsend.
$d_n=1$ ist monoton wachsend und fallend aber nicht streng.

:::

#### Untersuchung

Für die Untersuchung der Monotonie werden oft Ausdrücke der Form $a_{n+1}-a_n$ oder ${a_{n+1} \over a_n} > 1$ betrachtet.

Z.b wenn $a_{n+1} - a_n < 0$ (eigentlich $a_{n+1}<a_n$) gilt, ist die Folge streng monoton fallend. $a_{n+1} - a_n \leq 0$ wäre nur monoton fallend.

:::note

Zeig das die Folge $a_n={2^{n+1}\over 3^n}$ streng monoton fallend ist.

1. Es muss gelten: $a_{n+1} - a_n < 0 \iff a_{n+1}<a_n$
   1. Gleichnamig machen: ${2^{n+1}\over 3^n} - {2^{n+2} \over 3^{n+1}} = {3*2^{n+1}\over 3*3^n}  - {2*2^{n+1} \over 3^{n+1}}$
   2. Vereinfachen: $(3-2)*2^{n+1} \over 3^{n+1}$
   3. $({2\over 3})^{n+1}>0$
2. Oder es muss gelten: ${a_{n+1} \over a_n} > 1$
   1. ${{2^{n+1}\over 3^n}\over {2^{n+2} \over 3^{n+1}}} = {{2^{n+1}\over 3^n} *{3^{n+1} \over 2^{n+2}}}$
   2. Kürzen: ${{2^{n+1}\over 3^n} *{3*3^{n} \over 2*2^{n+1}}}={3\over 2} > 1$

:::

### Berschränktheit

Eine Folge $a_n$ heisst
**Nach oben beschränkt**, wenn es eine Zahl $S$ gibt, so dass $a_n \leq S$ für alle $n \in N$ gilt. $S$ heisst eine obere Schranke der Folge.
**Nach unten beschränkt**, wenn es eine Zahl $s$ gibt, so dass $a_n \geq s$ für alle $n \in N$ gilt. $s$ heisst eine untere Schranke der Folge.

Hat eine Folge eine obere und untere Schranke ist sie eine **beschränkte** Folge.

:::note

$a_n=-2n^2+4 = 2,-4,-14,-28,...$ nach oben Beschränkt mit $S=2$
$a_n=n-5 = -5,-4,-3,...$ nach unten Beschränkt mit $s=5$
$a_n=(-1)^n = -1,1,-1,1,...$ Beschränkt mit $s=1$ und $S=1$
$a_n=(-1)^n *n^2= -1,4,-9,16,...$ weder nach oben noch nach unten

:::

### Konvergenz

:::note

Für $a_n={2n+3 \over n}$

| $a_1$ | $a_2$ | $a_3$ | $a_{10}$ | $a_{1000}$ | $a_{100000}$ |
| ----- | ----- | ----- | -------- | ---------- | ------------ |
| 5     | 3.5   | 3     | 2.3      | 2.003      | 2.00003      |

**Sprechweise**:
Die Folge $a_n$ strebt mit wachsendem $n$ gegen den Grenzwert 2.

**Schreibweise**:
$$\lim_{x \to \infty} {2n+3 \over n} = 2$$
Wird gelesen als "Limes von ${2n+3 \over n}$ für n gegen unendlich ist 2."

:::

#### $\varepsilon$-Umgebung

**$\varepsilon$-Umgebung oder $\varepsilon$-Streifen** ist ein Streifen mit einem Radius $\varepsilon$ um den vermuteten Grenzwert. Der Index des Folgengliedes, welches als erstes im Streifen liegt, nennt man **Eintauchzahl**, $N_{\varepsilon}$.

:::note

$a_n={2n+3 \over n}$ mit Streifen $\varepsilon = {1\over 2}$
![[Pasted image 20211016100108.png]]
Ist die Eintauchszahl $N_{1\over 2} = 7$, weil $a_6$ liegt auf dem Streifenrand, $a_7$ jedoch darin.

:::

#### Definition

Eine Folge $a_n$ konvergiert gegen einen **Grenzwert** $g \in R$ wenn es zu jeder noch so kleinen Zahl $\varepsilon >0$ eine Zahl $N_{\varepsilon}$ gibt, so dass $|a_n -g| < \varepsilon$ für alle $n \geq N_{\varepsilon}$.

Eine Folge die einen Grenzwert $g \in R$ besitzt, heisst **konvergent**. **Achtung** $\infty \notin R$!!!!!
Eine Folge die keinen Grenzwert besitzt, heisst **divergent**.

- Jede monoton wachsende (bzw. fallende) Folge, die beschränkt ist, ist immer konvergent.
- Das Produkt einer beschränkten Folge und einer Nullfolge ist immer eine Nullfolge.

#### Untersuchen

:::note

$a_n={n-1 \over n+2}$ vermuteter Grenzwert $g=1$
$|a_n-g|=|{n-1\over n+2}-1| < \varepsilon$
$\Rightarrow |{n-1\over n+2}-{n+2\over n+2}| < \varepsilon$
$\Rightarrow |{-3\over n+2}|={|-3|\over|n+2|} < \varepsilon$
$\Rightarrow {3\over n+2} < \varepsilon$
$\Rightarrow {3\over \varepsilon} -2< n$

Für $\varepsilon = 0.2$ erhält man $n>{3\over 0.2} -2 = 13$ daher ab $a_14$ beträgt Differenz von Folgenglied und Grenzwert weniger als $\varepsilon$
$N_{0.2}=14$

:::

## Spezielle Folgen

**Nullfolge** eine Folge die den Grenzwert 0 besitzt.
**Harmonische Folge** $a_n={1\over n}$ ist eine Nullfolge

## Geometrsiche Folge

Folgen der Form: $a_n= a_1 *q^{n-1}$ sind geometrische Folgen.
 Jedes Glied ist das geometrische Mittel seiner beiden Nachbarglieder $a_n=\sqrt {a_{n-1}+a_{n+1}}$

### Konvergenzkriterien für geometrische Folgen

Eine geometrische Folge $a_n= a_1* q^{n-1}$

- mit $|q|>1$ ist divergent
- mit $|q|<1$ ist konvergent mit Grenzwert 0
- mit $q=1$ ist eine konstante Folge $a_1$
- mit $q=-1$ ist divergent, da alternierend.

## Rechenregeln konvergente Folgen

Sind $a_n$ und $b_n$ konvergente Folgen mit den Grenzwerten $a$ bzw. $b$, so ist auch die Folge:

- $c*a_n$ konvergent mit $\lim_{n \to \infty} {c*a_n} = c*\lim_{n \to \infty} {a_n} = c*a$ für $c \in R$
- $a_n \pm b_n$ konvergent mit $\lim_{n \to \infty}{a_n \pm b_n}={{\lim_{n \to \infty}{a_n}} \pm {\lim_{n \to \infty}{b_n}}}={a \pm b}$
- $a_n *b_n$ konvergent mit $\lim_{n \to \infty}{a_n* b_n}={{\lim_{n \to \infty}{a_n}} * {\lim_{n \to \infty}{b_n}}}={a * b}$
- $a_n \over b_n$ konvergent mit $\lim_{n \to \infty}{a_n \over b_n}={{\lim_{n \to \infty}{a_n}} \over {\lim_{n \to \infty}{b_n}}}={a \over b}$ falls $b \neq 0$

## Rationale Folgen

Für eine rationale Folge, die im Zähler aus einem Polynom k-ten Grades
und im Nenner aus einem Polynom l-ten Grades besteht, gilt:
 $$\lim_{n \to \infty}{{a_kn^k+a_{k-1}n^{k-1}+...+a_0}\over{b_ln^l+b_{l-1}n^{l-1}+...+b_0}} = \begin{dcases}
        {a_k\over b_k} *\infty, falls\space k >l \\
        {a_k\over b_k} , falls\space k=l \\
        0 , falls\space k<l
\end{dcases}$$
