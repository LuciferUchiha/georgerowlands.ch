# Ableitungsregeln

## Konstantenregel

Die Ableitung einer Konstanten ist 0.
$$f(x)=C \Rightarrow f'(x)=0, C \in R$$

## Faktorregel

Beim Ableiten einer Funktion, bleibt ein konstanter Faktor $k \in R$ vor einer Funktion unverändert erhalten.
$$g(x)=k*f(x) \Rightarrow g'(x)=k*f'(x)$$

## Ableitungen der trigonometrischen Funktionen

- $f(x)=sin(x) \Rightarrow f'(x)=cos(x)$
- $f(x)=cos(x) \Rightarrow f'(x)=-sin(x)$
- $f(x)=tan(x) \Rightarrow f'(x)={1\over cos^2(x)}=1+tan^2(x)$ für $x \neq (2k+1){\pi \over 2}$
- $f(x)=cot(x) \Rightarrow f'(x)=- {1\over sin^2(x)}=-1-cot^2(x)$ für $x \neq k \pi$

## Potenzregel

Die [[4-Funktionen#Potenzfunktion]] $f(x) = x^n$ ist für alle $x \in R$ differenzierbar.
$$f(x)=x^n \Rightarrow f'(x)=nx^{n-1}, n \in Z$$

```ad-example
![[Pasted image 20211024162633.png]]
```

## Summenregel

Die Ableitung einer Summe ist gleich der Summe der Ableitungen.
$$s(x)=f(x)\pm g(x) \Rightarrow s'(x)=f'(x) \pm g'(x)$$

```ad-example
![[Pasted image 20211024162805.png]]
```

## Produktregel

$$f(x)=u(x)* v(x) \Rightarrow f'(x)=u'(x)*v(x) + u(x)*v'(x)$$

```ad-example
![[Pasted image 20211024163316.png]]
```

### Allgemeine Produktregel

Allgemein gilt für die Ableitung eines Produktes aus $n$ Faktoren.
$$f(x)=u_1*u_2*...*u_n \Rightarrow f'(x)=u'_1*u_2*...u_n+u_1*u'_2*..*u_n+...+u_1*u_2*...*u'_n$$
So wäre: $(uvw)'=u'vw+uv'w+uvw'$ und $(uvwz)'=u'vwz+uv'wz+uvw'z+uvwz'$

## Quotientenregel

$$f(x)={u(x)\over v(x)} \Rightarrow f'(x)={{u'(x)*v(x)-u(x)*v'(x)}\over v(x)^2}$$

```ad-example
![[Pasted image 20211024164435.png]]
```

## Kettenregel

Unter der Verkettung der Funktionen $g$ und $h$ versteht man die Nacheinanderausführung der Funktionen. Man wendet die äussere Funktion $g$ auf das Ergebnis der inneren Funktion $h$. Also von innen nach aussen.
$f(x)=g(h(x)) \iff f(x)=(g \circ h)(x)$

Man setzt für die innere Funktion: $z=h(x)$
so dass sich für die äussere Funktion $f=g(z)=g(h(x))$

$$f(x)=g(h(x)) \Rightarrow f'(x)=g'(z)*h'(x)$$

```ad-example
![[Pasted image 20211024170605.png]]
```

## Ableitung der Exponentialfunktion

$$f(x)=e^{g(x)} \Rightarrow f'(x)= g'(x)* e^{g(x)}$$

```ad-example
![[Pasted image 20211024171328.png]]
```

## Ableitung der Logarithmusfunktion

$$f(x)=log_a(x) \Rightarrow f'(x)= \frac{1}{x * ln(a)}$$
wenn $f(x)=ln(x) \Rightarrow f'(x)= \frac{1}{x}$

```ad-example
![[Pasted image 20211024171755.png|100]]
```

## Ableitung der Umkehrfunktion

Die Funktion $f(x)$ sei differenzierbar mit der Ableitung $f'(x)$ und besitzt die Umkehrfunktion $x = g(y)$. Die Ableitung der Umkehrfunktion $g(y)$ ist
$$g'(y)={1\over f'(x)}$$

```ad-example
![[Pasted image 20211024172038.png]]
```
