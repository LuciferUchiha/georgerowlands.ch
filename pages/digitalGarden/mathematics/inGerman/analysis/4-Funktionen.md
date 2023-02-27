# Funktionen

Eine Funktion $f$ ordner jedem Element x einer Defintionsmenge(auch Definitionsbereich genannt) $D$ genau ein element $y$ einer Wertemenge(Wertebereich) zu.

Schreibweise: $f: D \mapsto W, x\mapsto y$

 Für das dem Element $x \in D$ zugeordnete Element der Wertemenge schreibt man im Allgemeinen $f(x)$.

## Reele Funktion

Unter einer reellen Funktion $f$ versteht man die Abbildung, die jedem $x \in D$ mit $D \subseteq R$ genau eine reelle Zahl $y$ aus einer Wertemenge $W$ zuordnet:
$$f:x\mapsto y=f(x), D \subseteq R \mapsto W \subseteq R$$

## Funktionseigenschaften

### Nullstelle

Eine Funktion $f$ besitzt eine Nullstelle in $x_0$, falls $f(x_0) = 0$ gilt.
Der Funktionsgraph schneidet die x-Achse in einer Nullstelle der Funktion.

### Gerade

Eine Funktion heisst gerade, falls $f(x) = f(−x)$ für alle $x\in D$ gilt.
Der Funktionsgraph einer geraden Funktion ist spiegelsymmetrisch zur y−Achse.

```ad-example
Die Funktion $f(x)=x^2$ ist gerade.
```

### Ungerade

Eine Funktion heißt ungerade, falls $f(−x) = −f(x)$ für alle $x\in D$ gilt.
Der Funktionsgraph einer ungerade Funktion ist punktsymmetrisch zum Koordinatenursprung.

```ad-example
Die Funktion $f(x)=x^3$ ist ungerade.
```

## Polynomfunktion

Eine Funktion $f: R \mapsto R$ der Form:
$$f(x)=a_nx^n+a_{n-1}x^{n-1}+...+a_1x+a_0$$ mit $a_n \neq 0$
heisst **Polynom vom Grad** $n$. Die reelen Zahlen $a_0,a_1,...,a_n$ heissen **Koeffizienten** der Polynoms.

```ad-example
$f_1(x)=x^3-x+2$ ist ein Polynom 3. Grades.
$f_2(x)=2x^7-4x^5+x^2-3x+2$ ist ein Polynom 7. Grades.
```

### Linearfaktoren

Ist $x_0$ eine Nullstelle des Polynoms $n$-ten Grades von $f$, dann wäre ein **Linearfaktor** von $f$:
$$f(x)=(x-x_0)(b_nx^{n-1}+..+b_2x+b_1)$$

Jedes Polynom $n$-ten Grades hat höchstes n verschiedene Nullstellen.
Besitzt ein Polynom $n$-ten Grades $n$ Nullstellen $x_1,x_2,..x_n$ dann lässt es sich als Produkt aus $n$ Linearfaktoren darstellen:
$$f(x)=a_nx^n+a_{n-1}x^{n-1}+a_1x+a_0=a_n(x-x_1)(x-x_2)...(x-x_{n-1})(x-x_n)$$

#### Zerlegung in Linearfaktoren

Die Abspaltung eines Linearfaktors erreicht man am besten mit Polynomdivision.

```ad-example
$f(x)=x^3-7x^2-10x+16$
Durch einsetzen, dass $x_1 = {\color{Red}1}$ eine Nullstelle des Polynoms d.h. $f(1) = 0$

$(x^3-7x^2-10x+16) : (x-{\color{Red}1})=x^2-6x-16$
$\underline{-(x^3-x^2)}$
$\quad -6x^2-10x$
$\quad \underline{-(-6x^2+6x)}$
$\quad \quad -16x+16$
$\quad\quad \underline{-(-16x+16)}$
$\quad\quad\quad0$

Wir erhalten dadurch: $f(x)=(x-1)(x^2-6x-16)$. 
$(x^2-6x-16)$ kann dann weiter mit der Polynomdivision zerteilen um die weiteren Linearfaktoren zu erhalten.

$f(x)=x^3-7x^2-10x+16=(x-1)(x+2)(x-8)$
```

## Rationale Funktion

Eine rationale Funktion ist eine Funktion, die sich als Bruch von zwei [[#Polynomfunktion]] $g(x)$ und $h(x)$ darstellen lässt.
$$f(x)={g(x) \over h(x)}={{a_mx^m+a_{m-1}x^{m-1}+...+a_1x+a_0}\over {b_nx^n+b_{n-1}x^{n-1}+...+b_1x+b_0}}$$

Ein [[#Polynomfunktion]] ist eine rationale Funktion wo $n=0$.

### Echt rationale Funktionen

Wenn $m<n$

### Unecht rationale Funktionen

Wenn $m \geq n$

### Eigenschaften

Sei $f(x) = g(x)\overh(x)$ eine rationale Funktion. Mit Zähler und Nenner soweit möglich in Linearfaktoren zerteilt und gemeinsame Linearfaktoren gekürtzt

#### Nullstellen

Die im Zähler($g(x)$) verbleibenden Linearfaktoren ergeben die **Nullstellen** der Funktion $f(x)$.

#### Polstellen

Die im Nenner ($h(x)$)verbleibenden Linearfaktoren ergeben die **Polstellen**
der Funktion $f(x)$.

##### Pollstelle $k$-ter Ordnung

Ist Linearfaktor im gekürzten Nenner in $k$-ter Ordnung $(x − x_0)^k, k \in N$  dann nennt man die Stelle $x_0$ eine Polstelle $k$−ter Ordnung.

##### Pollstelle mit Vorzeichenwechsel

Es sei $x_0$ eine Pollstelle $k$-ter Odnung.

- Ist $k$ gerade, so handelt es sich um eine **Pollstelle ohne Vorzeichenwechsel**.
- Ist $k$ ungerade, so handelt es sich um eine **Pollstelle mit Vorzeichenwechsel**.

```ad-example
$f(x)={1 \over (x+1)}$ hat bei $x=-1$ eine Pollstelle mit Vorzeichenwechsel.
$f(x)={1 \over (x-1)^2}$ hat bei $x=1$ eine Pollstelle ohne Vorzeichenwechsel.
```

#### Defintionslücken

Vor dem kürzen sind die Nullstellen im Nenner($h(x)$) für rationale Funktionen nicht definiert. Sie müssen explizit aus dem Definitionsbereich der Funktion herausgenommen werden, man spricht von **Definitionslücken**.

#### Hebbare Definitionslücken

Die vollständig weggekürzten Linearfaktoren im Nenner geben die
**hebbaren Definitionslücken** der Funktion $f(x)$ an.

### Verhalten rationale Funktionen im Unendlichen

Genau gleich wie [[2-Folgen#Rationale Folgen]].
Sei $f(x) = {g(x)\over h(x)}$ eine rationale Funktion, dann gilt für den Grenzwert:
 $$\lim_{n \to \infty}{f(x)} = \begin{dcases}
        0, grad\space g < grad \space h \\
        {a_n\over b_n} , grad \space g=grad \space h \\
        {a_n\over b_n} * \infty , grad \space g > grad\space h
\end{dcases}$$

## Umkehrfunktion

Eine Funktion $f: x \mapsto y, D \mapsto W$ heisst umkehrbar, wenn aus $x_1 \neq x_2$ stets folgt $f(x_1)\neq f(x_2)$
Ist die Funktion umkehrbar, dann gibt es zu jedem $y \in W$ genau ein $x \ in D$ $f^{-1}: y \mapsto f^{-1}(y)=f^{-1}(f(x))=x$ wird **Umkehrfunktion** genannt.

```ad-example
![[Pasted image 20211024113010.png]]
```

## Potenzfunktion

[[#Polynomfunktion]] der Form $p: x \mapsto ax^n, R \mapsto R$ für $a,n \in R$![[Pasted image 20211024113631.png|300]]
[[#Potenzfunktion]] haben [[#Wurzelfunktion]] als [[#Umkehrfunktion]] und umgekehrt.

```ad-example
$p(x) = x^2$ hat $p^{-1}(x)=\sqrt{x} = x^{1 \over 2}$
$p(x) = x^3$ hat $p^{-1}(x)=\sqrt[3]{x} = x^{1 \over 3}$
```

## Wurzelfunktion

Die Funktion$p^{-1}: x \mapsto\sqrt[n]{x}$ für n gerade $R^+ \mapsto R^+$, für n ungerade $R \mapsto R$ heisst $n$-te Wurzerlfunktion mit $n \in N$.

## Exponentialfunktion

$f: x \mapsto e^x$ mit $e=2.71828...=$ Eulersche Zahl heisst Exponentialfunktion.
![[Pasted image 20211024114845.png]]

### Rechenregeln der Exponentialfunktion

- $e^0=1$
- $e^{x+y}=e^x*e^y$
- $e^{-x}=(e^x)^-1={1 \over e^x}$
- $e^{nx}=(e^x)^n$
- $e^{1 \over n}=\sqrt[n]{e}$

## Logarithmusfunktion

Die [[#Umkehrfunktion]] zu [[#Exponentialfunktion]] wird **natürliche Logarithmusfunktion** genannt. $f: x \mapto ln(x), R^+ \mapsto R$

### Rechenregeln der Logarithmusfunktion

- $ln(1)=0$
- $ln(x*y)=ln(x)+ln(y)$
- $ln(x^n)= n*ln(x)$
- $ln(e^x)=x ln(e) = x$ weil $ln(e)=1$

## Trigonometrische Funktionen

Sinus- und Cosinusfunktion sind periodisch mit der Periode $2\pi$, d.h. es gilt
$f(x)=f(x+k*2\pi), k \in Z$

Die Funktionsgraphen von Sinus- und Cosinusfunktion sind kongruent. Durch Verschiebung um $2\pi$ nach links, geht die Cosinus-Kurve aus der Sinus-Kurve hervor.

### Trigonometrischer Pythagoras

$$sin^2(a)+cos^2(a)=1$$

### Sinus

$sin: x \mapsto sin(x), R \mapsto [-1,1]$

### Cosinus

$cos: x \mapsto cos(x), R \mapsto [-1,1]$

### Tangens

### Cotangens

## Grenzwert einer Funktion

$$f(x)={1-x^3-cos(2x)}\over x^2$$ ist für $x=0$ nicht definiert, hier besteht eine Definitionslücke. Wir können den Funktionswert an der Stelle $x = 0$zwar nicht berechnen, aber mit einer Folge $x_n$ beliebig nahe an die Definitionslücke herantasten.

### Rechtseitigen Grenzwert

Die Folge $x_n=1\over n$ für $n \to \infty$ konvergiert gegen 0. Wir können somit die Folgenglieder in die Funktion einsetzen:

Wir vermuten, dass die Funktionswerte gegen den Grenzwert 2 konvergieren.
Es gilt also für jede beliebige Folge $x_n \to 0$, dass $f(x_n) \to 2$ gilt. Man schreibt daher:
$$\lim_{n \to \infty}{f(x_n)}=\lim_{x \to \infty,(x>0)}{f(x)}=2$$ und bezeichnet diesen Wert als den **rechtseitigen Grenzwert** der Funktion $f(x)$ an der Stelle $x=0$

### Linkseitigen Grenzwert

Wir wollen nun eine Folge betrachten, die sich von links dem Wert 0 nähert. Z.B $x_n=-1\over n$. Hier erhalten wie folgende Wertetabelle beim einsetzen:
![[Pasted image 20211024102226.png]]
Aus der Wertetabelle entnehmen wir auch hier, dass die Folge der Funktionswerte $f(x_n)$ gegen den Wert 2 konvergiert.
Man schreibt daher:
$$\lim_{n \to \infty}{f(x_n)}=\lim_{x \to \infty,(x<0)}{f(x)}=2$$ und bezeichnet diesen Wert als den **linkseitigen Grenzwert** der Funktion $f(x)$ an der Stelle $x=0$

### Zusammenfassung Grenzwert einer Funktion

Betrachtet man bei der Grenzwertbetrachtung einer Funktion $f$ an der Stelle $x_0$ nur Zahlenfolgen $x_n$, die kleinere Werte als $x_0$ enthalten, dann bezeichnet man den Grenzwert als [[#Linkseitigen Grenzwert]]
$$\lim_{x \to x_0,(x<x_0)}{f(x)}=\lim_{h \to 0,(h>0)}{f(x_0-h)}=G_L$$
Zahlenfolgen mit grösseren Werten als $x_0$ erzeugen den [[#Rechtseitigen Grenzwert]]
$$\lim_{x \to x_0,(x>x_0)}{f(x)}=\lim_{h \to 0,(h>0)}{f(x_0+h)}=G_R$$

 Streben für jede gegen $x_0$ konvergente Zahlenfolge $x_n$ die Funktionswerte $f(x_n)$ gegen denselben Wert $G$, dann besitzt die Funktion $f$ an der Stelle $x_0$ den Grenzwert $G$
 $$\lim_{x \to x_0}{f(x)}=G=G_R=G_L$$

## Rechenregeln für Funktionsgrenzwerte

Seien $f$ und $g$ zwei Funktionen mit dem gleichen Grenzwerten $G,F$ bei $x_0$ dann gilt:

- $\lim_{x \to x_0}{(f(x)\pm g(x))}=F\pm G$
- $\lim_{x \to x_0}{(f(x)*g(x))}=F*G$
- $\lim_{x \to x_0}{(f(x)\over g(x))}=F\over G$ für $g(x_0) \neq 0$ und $G \neq 0$

## Stetigkeit

Eine Funktion heisst stetig, wenn ihr Graph kein Loch und keinen Sprung aufweist, d.h. wenn man beim Zeichnen ihres Graphen den Stift nicht absetzen muss.

Eine Funktion heisst an einer Stelle $x = x_0$ stetig, wenn der Grenzwert von $f(x)$ für $x \to x_0$ existiert und mit dem Funktionswert an der Stelle $x_0$ übereinstimmt:
$$\lim_{x \to x_0}{f(x)}=f(x_0)$$

 Eine Funktion, die an jeder Stelle ihres Definitionsbereiches $D$ stetig ist, nennt man eine **stetige Funktion** (auf $D$).  
Existiert der Grenzwert hingegen nicht oder ist er nicht gleich wie der Funktionswert, so ist die Funktion an dieser Stelle **unstetig**.

### Hebbare Unstetigkeitsstelle

Wenn bei einer Funktion f der linksseitige Grenzwert und der rechtsseitige Grenzwert existieren und gleich sind aber nicht mit dem Funktionswert $f(x_0)$ übereinstimmen oder die funktion an $x_0$ nicht definiert ist, dann kann man eine neue Funktion definieren, die an der Stelle $x_0$ stetig ist. Die Stelle $x_0$ heisst **hebbare Unstetigkeitsstelle**
$$\hat{f}=\begin{dcases}
        f(x), x\neq x_0 \\
        G, x=x_0
\end{dcases}$$

### Stetige Funktionen

- [[#Polynomfunktion]] sind für $R$ stetig.
- Exponentialfunktionen $f(x)=a^x, (a > 0, 1\neq 0)$ sind für $R$ stetig.  
- Logarithmusfunktionen $f(x)=log_a(x), (a > 0, 1\neq 0)$ sind für $x>0$ stetig.
- Trigonometrsiche Funktionen $cos(x), sin(x)$ sind für $R$ stetig.
- Hyperbelfunktionen $sinh(x), cosh(x), tanh(x))$sind für R stetig.

### Rechenregeln für Stetige Funktionen

Sind die Funktionen $f$ und $g$ auf ihrem ganzen Definitionsbereich stetig, insbesonders an der Stelle $x_0$ gilt:

- $f \pm g$ ist ebenfalls stetig in $x_0$
- $f * g$ ist ebenfalls stetig in $x_0$
- $f \over g$ ist ebenfalls stetig in $x_0$, falls $g(x_0) \neq 0$
- Die Komposition von $f \circ g$ ist ebenfalls an der Stelle $x_0$ stetig

## Steigung

Die allgemeine Geradengleichung lautet:
$$g(x)=m*(x-x_0)+y_0$$
$m$ ist dann die Steigung der Geraden.

### Sekante -  Differenzenquotient

Eine Gerade durch 2 Punkte $P_0(x_0/f(x_0))$ und $P_1(x_1/f(x_1))$ heisst **Sekante**.
Die Steigung der Sekante wird **Differenzenquotient** genannt.
$$m={\Delta f \over \Delta x }={{f(x1)-f(x_0)} \over {x_1-x_0}}={f(x_0 + \Delta x)-f(x_0)\over \Delta x}$$
Gleichung der Sekante lautet: $g(x)=m*(x-x_0)+f(x_0)$

### Tangente - Differenzialkoeffizient

Eine **Tangente** einer Funktion $f$ im Punkt $x_0$ ist eine Gerade durch einen Punkt $P(x_0/f(x_0))$. Der Grenzwert des [[#Sekante - Differenzenquotient]] wird als **Differnzialkoeffizient**, dafür gibt es verschiedende Schreibweisen:

$$f'(x_0)=\lim_{\Delta x \to x_0}{f(x_0 + \Delta x)-f(x_0)\over \Delta x}=\lim_{\Delta x \to x_0}{\Delta f \over \Delta x}={df \over dx}=\lim_{h \to x_0}{{f(x+h)-f(x)}\over h}$$

Gleichung der Tangente lautet: $g(x)=f(x_0)+f'(x_0)*(x-x_0)$

Existiert der Grenzwert des Differenzenquotienten dann nennt man die Funktion **differenzierbar** an der Stelle $x_0$. Der Grenzwert wird als **Ableitung** der Funktion $f$ an der Stelle $x_0$.

## Ableitungsfunktion

Die Funktion $f': x \mapsto f'(x)$ heisst die Ableitungsfunktion von $f(x)$ oder kurz **Ableitung von** $f()x$.
Die Ableitungsfunktion ordnet jedem Wert x die Steigung der Tangente an der Stelle x zu.

### Höhere Ableitung

Exisitiert zu einer Funktion $f$ Ableitung $f'$ und ist $f'(x)$ wieder differenzierbar, so bezeichnet man deren Ableitung als zweite Ableitung $f''(x)$. Es gilt also $f''=(f')'$
Die $n$-te Ableitung für $n>3$ schreibt man $f^{(n)}$, die Funktion ist dann $n$-mal differenzierbar.

## Differenzierbarkeit und Stetigkeit

- Jede differenzierbare Funktion ist auch stetig und hat an allen Stellen eine eindeutige Steigung.
- Ist eine Funktion $f$ an der Stelle $x_0$ nicht stetig, dann ist sie dort auch nicht differenzierbar.
