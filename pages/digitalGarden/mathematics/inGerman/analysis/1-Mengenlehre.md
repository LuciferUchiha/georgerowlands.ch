# Mengen

Wenn x ein Element der Menge M ist, schreiben wir wenn x in M bzw. nicht in M ist.
$x \in M$ oder $x \notin M$
Leere Menge enthält keine Elemente: $\emptyset$ oder $\{\}$

## Nätürliche Zahlen

$N = \{1,2,3,..\}$
$N_0 = \{0,1,2,3,..\}$
$N_0^{\leq k} = \{0,1,2,3,...k\}$
Gleichungen der Form
$a + x = b$ mit $a,b \in N$
lassen sich in der Menge der natürlichen Zahlen nicht lösen, falls $a > b$ ist.

## Ganze Zahlen

$Z=\{...,-2,-1,0,1,2,...\}$
$Z^+=\{1,2,3,...\}$
$Z^-=\{...,-2,-1\}$
Gleichungen der Form
$q*x = p$ mit  $p,q \in Z$
sind in der Menge der ganzen Zahlen dann nicht mehr lösbar, wenn $q$ kein Teiler von $p$ ist.

## Rationale Zahlen

$Q=\{{p \over q}:p\in Z, q\in N\}$
Durch Null darf nicht dividiert werden, d.h. ${p \over 0} \notin Q$
**Bruchdarstellung**: ${p \over q} \in Q$
**Dezimaldarstellung**: Ausdividieren Zähler durch Nenner ergibt:

- Abbrechende (endliche) Dezimalzahl: ${-1 \over 32} = -0.03125$
- Periodisch unendliche Dezimalzahl: ${23 \over 99} = 0.232323...=0.\overline{23}$

**Geometrsiche Darstellung**: Ein Punkt auf der Zahlengerade

## Irrationale Zahlen

Zahlen, die nicht rational sind, aber die sich als Dezimalzahl mit unendlich nicht periodisch vielen Nachkommastellen schreiben lassen, werden irrational genannt.

:::note
 $\sqrt2$ ist keine rationale Zahl, $\sqrt2 \notin Q$
 **Beweis**:
 Wir zeigen dies indirekt, durch die Annahme des Gegenteils.
 Annahme: $\sqrt 2 \in Q \Rightarrow  \sqrt 2 = {p \over q}$ wobei der Bruch gekürtzt ist, also p und q teilerfremd sind.
 Quadrieren: $2=({p \over q})^2 \Rightarrow 2q^2=p^2$ das heisst p ist gerade und können $p=2*z$ mit $z \in Z$ ersetzen.
 $2q^2 = 4z^2 \Rightarrow q^2=2z^2$ das heisst q ist gerade.
 Also sind p und q gerade haben also den gemeinsamen Teiler 2, also ein Wiederspruch.

:::

## Reele Zahlen

$R$ = Rationale und Irrationale Zahlen.

# Intervalle

Seien $a,b \in R$ mit $a<b$.
**Abgeschlossenes Intervall**: $[a,b] = \{x \in R \space|\space a\leq x \leq b\}$ also Inclusive.
**Offenes Intervall**: $(a,b) = \{x \in R \space|\space a < x < b\}$ also Exclusive.
**Halboffene Intervalle**: nur ein Randpunkt gehört zur Menge. z.B

- Rechtsoffenes Intervall: $[a,b)=\{x\in R \space|\space a\leq x<b\}$

Das Intervall enthält $a$, aber nicht $b$.

- Linkstsoffenes Intervall: $(a,b]=\{x\in R \space|\space a < x \leq b\}$

Das Intervall enthält $b$, aber nicht $a$.

**Nach oben oder unten abgeschlossene Intervalle**:
$[a,+\infty )= \{x \in R \space|\space a \leq x\}$
$(a,+\infty )= \{x \in R \space|\space a < x\}$
$(-\infty,b]= \{x \in R \space|\space x \leq b\}$
$(-\infty,b)= \{x \in R \space|\space x < b\}$
$(-\infty,+\infty )= R$

# Heron-Verfahren

Iterationsverfahren um reelle Zahl durch eine rationale Zahl mit beliebiger Genauigkeit annähern.  Die Idee ist, dass ein Quadrat mit Flächeninhalt $B$ eine Seitenlänge von $\sqrt B$ hat.

Um $\sqrt b$ anzunähern, wählt man Näherungswert $a_1$ z.B für $\sqrt 5$ kennen wir $\sqrt 4$ also $a_1=2$.
$$a_n={1\over2}(a_{n-1}+{b\over a_{n-1}})$$

# Betrag

Der Betrag(Absolutwert) für $a \in R$ beträgt:
$$|a|= max(a, -a)=
    \begin{dcases}
        a, a \geq 0 \\
        -a, a<0 \\
    \end{dcases}
$$
