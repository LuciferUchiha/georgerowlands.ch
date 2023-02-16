# Partielle Integration

Die Partielle Integrationsmethode wird auch oft Produkt integration genannt. Wir können diese Integrationsmethode wenn schon aus einem Produkt von 2 Funktionen besteht oder es als Produkt von 2 Funktionen darstellbar ist z.B. $\int{x\cdot e^x \,dx}$

$$\int{f(x)\,dx}=\int{u(x)\cdot v(x)\,dx}$$

Wichtig dabei ist, dass auch eines der Faktoren einfach integrierbar ist, wir sehen also schnell das eines der Faktoren eine Ableitung ist.

Aus der Produktregel der Differentialrechnung können wir folgendes bilden

$$
\begin{align*}
&(u(x)\cdot v(x))'=u'(x)\cdot v(x)+u(x)\cdot v'(x) \\
&\Rightarrow u(x)\cdot v'(x)=(u(x)\cdot v(x))'-u'(x)\cdot v(x)
\end{align*}
$$

Unbestimmte Integration auf beiden Seiten führt dann zu

$$
\begin{align*}
&\int{u(x)\cdot v'(x)\,dx}= \int{(u(x)\cdot v(x))'\,dx}-\int{u'(x)\cdot v(x)\,dx} \\
&\Rightarrow \int{u(x)\cdot v'(x)\,dx}= u(x)\cdot v(x)-\int{u'(x)\cdot v(x)\,dx}
\end{align*}
$$

Mit dieser Formel kann man dann die Integration lösen wenn man $u(x)$ ableitet und $v(x)$ integriert.

Genau so kann man auch vorgehen wenn es ein bestimmtes Integral ist nur ist es dann

$$\int_a^b{u(x)\cdot v'(x)\,dx} = \Big|u(x)\cdot v(x)\Big|_a^b-\int_a^b{u'(x)\cdot v(x)\,dx} $$

Mehr dazu findest du auch [hier](https://studyflix.de/mathematik/partielle-integration-1862)

:::note Beispiel Partielle Integration

 Wir wollen das follgende Problem lösen

 $$\int{x\cdot e^x \, dx}=?$$

 Zuerst zerlegen wir den Integrand wie oben beschrieben.

- $u(x)=x$
- $u'(x)=1$
- $v'(x)=e^x$
- $v(x)=e^x$

 Aus der Formel können wir dann folgende berechnen

 $$\begin{align*}
 \int{x\cdot e^x \, dx}&=x \cdot e^x - \int{1 \cdot e^x \,dx} \\
 &\Rightarrow x \cdot e^x - e^x + C = (x-1) \cdot e^x + C
 \end{align*}$$

:::
