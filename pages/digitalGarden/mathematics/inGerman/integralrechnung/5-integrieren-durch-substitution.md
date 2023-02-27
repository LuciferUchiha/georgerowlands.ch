# Integration durch Substitutionen

Bei der Integration durch Substitutionen wollen wir mit Hilfe von geeigneten Variabel-Substitutionen das Integral vereinfachen oder wenn möglich sogar zu einem Grundintegral umwandeln.

Am besten können wir diese Methode verwenden wenn wir den folgenden Fall haben

$$\int{f(x)\,dx}=\int{f(g(x))\cdot g(x)'\,dx}$$

also wenn wir eine Verkettung von Funktionen haben und die innere Funktion abgeleitet im Integral vorkommt. Ein häufiges Beispiel ist

$$\int{x\cdot e^{x^2}}$$

weil es nicht normalerweise lösbar ist. Hier ist $g(x)=x^2$ was abgeleitet zu $g'(x)=2x$ wird wir haben aber nur $x$ nicht $2x$. Grund dafür ist die Faktorregel welche besagt das wir die 2 ja herausnehmen können, deshalb können wir Konstanten bei der obigen Voraussetzung ignorieren.

Der erste Schritt haben wir schon gemacht wir haben unsere variable zum Substituieren identifiziert $u=x^2$. Wir müssen aber alles was mit der alten Variable zu tun haben ersetzen, inklusive das $dx$. Um dies zu erreichen benutzen wir noch die folgende Formel $dx=\frac{du}{u'}=\frac{du}{2x}$.

Nun können wir in der Formel alles ersetzen

$$\int{x\cdot e^{x^2}}=\int{x\cdot e^u\, \frac{du}{2x}}$$

Dank der obigen Voraussetzung lässt sich das vordere $x$ wegkürzen.

$$\int{\frac{e^u\,du}{2}}=\int{\frac{1}{2}\cdot e^u\,du}=\frac{1}{2}\int{e^u\,du}$$

Nun haben wir ein Grundintegral und wir wissen das $e^u$ abgeleitet/integriert $e^u$ bleibt können wir das Integral lösen

$$\frac{1}{2}\int{e^u\,du}=\frac{1}{2}e^u +C$$

Oftmals will man noch die originale Variable beibehalten, dafür macht man dann eine Rücksubstitution.

$$\frac{1}{2}e^u +C=\frac{1}{2}e^{x^2} +C$$

## Integration durch Substitutionen eines bestimmten Integrals

Bei einem bestimmten Integral gehen wir genau gleich vor wie bei einem unbestimmten jedoch haben wir noch 2 weitere Schritte. Zwar müssen wir die Grenzen auch ersetzen und am Schluss dann das Integral ausrechnen. Dafür verwenden wir das folgende Beispiel

$$\int_{0}^{1}{x\cdot \sqrt{1+x^2}}$$

Wir setzen $u=1+x^2$ und somit auch $dx=\frac{du}{x'}=\frac{du}{2x}$

Nun müssen wir die Grenzen noch ersetzen. Für die untere Grenze ist $x=0$ und somit dann $u=1+0^2=1$. Für die obere Grenze $x=1$ und somit $u=1+1^2=2$. Nun können wir alles ersetzen.

$$\int_{0}^{1}{x\cdot \sqrt{1+x^2}}=\int_{u=1}^{u=2}{x\cdot \sqrt{u}\,\frac{du}{2x}}=\int_{1}^{2}{\frac{1}{2}\cdot \sqrt{u}\,du}=\frac{1}{2}\int_{1}^{2}{(u)^{\frac{1}{2}}\,du}$$

Weil $(\frac{2}{3}u^{\frac{3}{2}})'=(u)^{\frac{1}{2}}$ können wir schreiben

$$\frac{1}{2}\cdot\Big|\frac{2}{3}u^{\frac{3}{2}}\Big|_1^2=\frac{1}{3}\cdot\Big|\sqrt{u^3}\Big|_1^2=\frac{1}{3}(\sqrt{8}-1)$$
