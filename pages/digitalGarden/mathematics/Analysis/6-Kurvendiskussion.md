# Kurvendiskussion anhand von Ableitungen

## Monotonie

Die erste Ableitung an der Stelle $x_0$ beschreibt das Steigungsverhalten einer Funktion $f$ in der unmittelbaren Umgebung der Stelle $x_0$
$$f'(x_0)=\begin{dcases}
        <0 \Rightarrow \text{Funktion fällt, streng monoton fallend} \\
        >0 \Rightarrow \text{Funktion wächst, streng monoton wachsend}
\end{dcases}$$
![[Pasted image 20211024212936.png]]

```ad-example
![[Pasted image 20211024213308.png]]
```

## Krümmung

![[Pasted image 20211024213445.png]]
Die zweite Ableitung an der Stelle $x_0$ be-schreibt das Krümmungsverhalten einer Funktion $f$ in der unmittelbaren Umgebung der Stelle $x_0$:
$$f''(x_0)=\begin{dcases}
        <0 \Rightarrow \text{Rechtskrümmung, Steigung nimmt ab} \\
       <0 \Rightarrow \text{Linkskrümmung, Steigung nimmt zu}wachsend}
\end{dcases}$$

### Konkav

Ist $f''(x) < 0 \Rightarrow f'(x)$ ist streng monoton fallend $\Rightarrow f(x)$ ist konkav (Rechtsgekrümmt).

### Konvex

Ist $f''(x) > 0 \Rightarrow f'(x)$ ist streng monoton wachsend $\Rightarrow f(x)$ ist konvex (linksgekrümt).

## Extremwerte

### Lokales Maximum

Ist $f'(x_0)=0 und f''(x_0)<0$, dann ist x_0 ein lokales Maximum.

### Lokales Minimum

Ist $f'(x_0)=0 und f''(x_0)>0$, dann ist x_0 ein lokales Minimum.

```ad-example
![[Pasted image 20211024220630.png]]
```

## Wendepunkte und Sattelpunkte

![[Pasted image 20211024220732.png]]
Ist $f''(x_0)=0 und f'''(x_0)\neq 0$, dann ist $x_0$ ein Wendepunkt.
Wenn zusätzlich noch $f'(x_0)=0$, dann ist $x_0$ zusätzlich noch ein Sattelpunkt.

```ad-example
![[Pasted image 20211024220936.png]]
```
