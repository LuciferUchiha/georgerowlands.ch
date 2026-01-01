## Page 1

Fall2025
Probabilistic
Artiﬁcial Intelligence
Andreas Krause, Jonas Hübotter
Institute for Machine Learning
Department of Computer Science


## Page 2

Compiled on September 25,2025 .
This set of notes was written for the course P robabilistic Artificial Intelligence (263-5210 -00L) at ETH Zürich.
Distribution of these notes without the permission of the authors is prohibited.
©2025 ETH Zürich. All rights reserved.


## Page 3

Preface
Artiﬁcial intelligence commonly refers to the science and engineering
of artiﬁcial systems that can carry out tasks generally associated with
requiring aspects of human intelligence, such as playing games, trans-
lating languages, and driving cars. In recent years, there have been
exciting advances in learning-based, data-driven approaches towards
AI, and machine learning and deep learning have enabled computer
systems to perceive the world in unprecedented ways. Reinforcement
learning has enabled breakthroughs in complex games such as Go and
challenging robotics tasks such as quadrupedal locomotion.
A key aspect of intelligence is to not only make predictions, but reason
about the uncertainty in these predictions, and to consider this uncer-
tainty when making decisions. This is what this course on “Proba-
bilistic Artiﬁcial Intelligence” is about. The ﬁrst part covers proba-
bilistic approaches to machine learning. We discuss the differentiation
between “epistemic” uncertainty due to lack of data and “aleatoric”
uncertainty, which is irreducible and stems, e.g., from noisy observa-
tions and outcomes. We discuss concrete approaches towards proba-
bilistic inference, such as Bayesian linear regression, Gaussian process
models and Bayesian neural networks. Often, inference and making
predictions with such models is intractable, and we discuss modern
approaches to efﬁcient approximate inference.
The second part of the course is about taking uncertainty into account
in sequential decision tasks. We consider active learning and Bayesian
optimization — approaches that collect data by proposing experiments
that are informative for reducing the epistemic uncertainty. We then
consider reinforcement learning, a rich formalism for modeling agents
that learn to act in uncertain environments. After covering the basic
formalism of Markov Decision Processes, we consider modern deep
RL approaches that use neural network function approximation. We
close by discussing modern approaches in model-based RL, which har-
ness epistemic and aleatoric uncertainty to guide exploration, while
also reasoning about safety.


## Page 4

iv
Guide to the Reader
The material covered in this manuscript may support a one semester
graduate introduction to probabilistic machine learning and sequential
decision-making. We welcome readers from all backgrounds. How-
ever, we assume familiarity with basic concepts in probability, calculus,
linear algebra, and machine learning (e.g., neural networks) as covered
in a typical introductory course to machine learning. In Chapter 1,w e
give a gentle introduction to probabilistic inference, which serves as
the foundation for the rest of the manuscript. As part of this ﬁrst chap-
ter, we also review key concepts from probability theory. We provide
a chapter reviewing key concepts of further mathematical background
in the back of the manuscript.
Throughout the manuscript, we focus on key concepts and ideas rather
than their historical development. We encourage you to consult the
provided references for further reading and historical context to delve
deeper into the covered topics.
Accompanying this manuscript, we provide an extensive set of curated
examples as Jupyter notebooks which you can run and play with. You
can ﬁnd them at:
https://gitlab.inf.ethz.ch/OU-KRAUSE/pai-demos .
Finally, we have included a set of exercises at the end of each chapter.
When we highlight an exercise throughout the text, we use this ques-
tion mark: ? Problem 1.1 — so don’t be surprised when you stumble upon it. You
will ﬁnd solutions to all exercises in the back of the manuscript.
We hope you will ﬁnd this resource useful.
Contributing
We encourage you to raise issues and suggest ﬁxes for anything you
think can be improved. We are thankful for any such feedback!
Contact : pai-script@lists.inf.ethz.ch
Repository :https://gitlab.inf.ethz.ch/OU-KRAUSE/pai-script
Acknowledgements
We are grateful to Sebastian Curi for creating the original Jupyter note-
books that accompany the course at ETH Zürich and which were in-
strumental in the creation of many ﬁgures. We thank Hado van Hasselt
for kindly contributing Figure 12.1, and thank Tuomas Haarnoja (Haarnoja
et al., 2018 a) and Roberto Calandra (Chua et al., 2018 ) for kindly agree-
ing to have their ﬁgures included in this manuscript. Furthermore,


## Page 5

v
many of the exercises in these notes are adapted from iterations of the
course at ETH Zürich. Special thanks to all instructors that contributed
to the course material over the years. We also thank all students of
the course in the Fall of 2022 ,2023 , and 2024 who provided valuable
feedback on various iterations of this manuscript and corrected many
mistakes. Finally, we thank Zhiyuan Hu, Shyam Sundhar Ramesh,
Leander Diaz-Bone, Nicolas Menet, and Ido Hakimi for proofreading
parts of various drafts of this text.


## Page 6




## Page 7

Contents
1 Fundamentals of Inference 1
1.1Probability 2
1.2Probabilistic Inference 15
1.3Supervised Learning and Point Estimates 22
1.4Outlook: Decision Theory 29
I Probabilistic Machine Learning 35
2 Linear Regression 39
2.1Weight-space View 40
2.2Aleatoric and Epistemic Uncertainty 44
2.3Non-linear Regression 45
2.4Function-space View 45
3 Filtering 51
3.1Conditioning and Prediction 53
3.2Kalman Filters 54
4 Gaussian Processes 59
4.1Learning and Inference 60
4.2Sampling 61
4.3Kernel Functions 62
4.4Model Selection 67
4.5Approximations 70


## Page 8

viii
5 Variational Inference 83
5.1Laplace Approximation 83
5.2Predictions with a Variational Posterior 87
5.3Blueprint of Variational Inference 88
5.4Information Theoretic Aspects of Uncertainty 89
5.5Evidence Lower Bound 100
6 Markov Chain Monte Carlo Methods 113
6.1Markov Chains 114
6.2Elementary Sampling Methods 121
6.3Sampling using Gradients 124
7 Deep Learning 139
7.1Artiﬁcial Neural Networks 139
7.2Bayesian Neural Networks 142
7.3Approximate Probabilistic Inference 144
7.4Calibration 152
II Sequential Decision-Making 157
8 Active Learning 161
8.1Conditional Entropy 161
8.2Mutual Information 163
8.3Submodularity of Mutual Information 166
8.4Maximizing Mutual Information 168
8.5Learning Locally: Transductive Active Learning 172
9 Bayesian Optimization 177
9.1Exploration-Exploitation Dilemma 177
9.2Online Learning and Bandits 178
9.3Acquisition Functions 180
10 Markov Decision Processes 197
10.1Bellman Expectation Equation 199
10.2Policy Evaluation 201


## Page 9

ix
10.3Policy Optimization 203
10.4Partial Observability 209
11 Tabular Reinforcement Learning 217
11.1The Reinforcement Learning Problem 217
11.2Model-based Approaches 219
11.3Balancing Exploration and Exploitation 220
11.4Model-free Approaches 224
12 Model-free Reinforcement Learning 233
12.1Tabular Reinforcement Learning as Optimization 233
12.2Value Function Approximation 235
12.3Policy Approximation 238
12.4On-policy Actor-Critics 244
12.5Off-policy Actor-Critics 251
12.6Maximum Entropy Reinforcement Learning 256
12.7Learning from Preferences 260
13 Model-based Reinforcement Learning 273
13.1Planning 274
13.2Learning 281
13.3Exploration 287
A Mathematical Background 299
A.1Probability 299
A.2Quadratic Forms and Gaussians 301
A.3Parameter Estimation 302
A.4Optimization 313
A.5Useful Matrix Identities and Inequalities 319
B Solutions 321
Bibliography 385
Syllabus 393


## Page 10

x
Summary of Notation 395
Acronyms 401
Index 403


## Page 11

1
Fundamentals of Inference
Boolean logic is the algebra of statements which are either true or false.
Consider, for example, the statements
“If it is raining, the ground is wet.” and “It is raining.”
A quite remarkable property of Boolean logic is that we can combine
these premises to draw logical inferences which are new (true) state-
ments. In the above example, we can conclude that the ground must
be wet. This is an example of logical reasoning which is commonly
referred to as logical inference , and the study of artiﬁcial systems that
are able to perform logical inference is known as symbolic artiﬁcial in-
telligence .
But is it really raining? Perhaps it is hard to tell by looking out of the
window. Or we have seen it rain earlier, but some time has passed
since we have last looked out of the window. And is it really true that
if it rains, the ground is wet? Perhaps the rain is just light enough that
it is absorbed quickly, and therefore the ground still appears dry.
This goes to show that in our experience, the real world is rarely black
and white. We are frequently (if not usually) uncertain about the truth
of statements, and yet we are able to reason about the world and make
predictions. We will see that the principles of Boolean logic can be ex-
tended to reason in the face of uncertainty. The mathematical frame-
work that allows us to do this is probability theory, which — as we
will ﬁnd in this ﬁrst chapter — can be seen as a natural extension of
Boolean logic from the domain of certainty to the domain of uncer-
tainty. In fact, in the 20th century, Richard Cox and Edwin Thompson
Jaynes have done early work to formalize probability theory as the
“logic under uncertainty” (Cox, 2001 ; Jaynes, 2003 ).
In this ﬁrst chapter, we will brieﬂy recall the fundamentals of prob-
ability theory, and we will see how probabilistic inference can be used


## Page 12

2p r o b a b i l i s t i c a r t i f i c i a l i n t e l l i g e n c e
to reason about the world. In the remaining chapters, we will then
discuss how probabilistic inference can be performed efﬁciently given
limited computational resources and limited time, which is the key
challenge in probabilistic artiﬁcial intelligence .
1.1Probability
Probability is commonly interpreted in two different ways. In the fre-
quentist interpretation, one interprets the probability of an event (say
a coin coming up “heads” when ﬂipping it) as the limit of relative
frequencies in repeated independent experiments. That is,
Probability =lim
N→∞# events happening in Ntrials
N.
This interpretation is natural, but has a few issues. It is not very dif-
ﬁcult to conceive of settings where repeated experiments do not make
sense. Consider the outcome:
“Person X will live for at least 80years.”
There is no way in which we could conduct multiple independent ex-
periments in this case. Still, this statement is going to turn out either
true or false, as humans we are just not able to determine its truth
value beforehand. Nevertheless, humans commonly have beliefs about
statements of this kind. We also commonly reason about statements
such as
“The Beatles were more groundbreaking than The Monkees.”
This statement does not even have an objective truth value, and yet we
as humans tend to have opinions about it.
While it is natural to consider the relative frequency of the outcome
in repeated experiments as our belief, if we are not able to conduct
repeated experiments, our notion of probability is simply a subjective
measure of uncertainty about outcomes. In the early 20th century,
Bruno De Finetti has done foundational work to formalize this notion
which is commonly called Bayesian reasoning or the Bayesian interpre-
tation of probability (De Finetti, 2017 ).
We will see that modern approaches to probabilistic inference often
lend themselves to a Bayesian interpretation, even if such an interpre-
tation is not strictly necessary. For our purposes, probabilities will be
a means to an end: the end usually being solving some task. This task
may be to make a prediction or to take an action with an uncertain
outcome, and we can evaluate methods according to how well they
perform on this task. No matter the interpretation, the mathematical


## Page 13

fundamentals of inference 3
framework of probability theory which we will formally introduce in
the following is the same.
1.1.1Probability Spaces
A probability space is a mathematical model for a random experiment.
The set of all possible outcomes of the experiment Ωis called sample
space . An event A ↑Ωof interest may be any combination of possible
outcomes. The set of all events A↑P (Ω)that we are interested in
is often called the event space of the experiment.1This set of events is1We use P(Ω)to denote the power set
(set of all subsets) of Ω.required to be a σ-algebra over the sample space.
Deﬁnition 1.1(σ-algebra) .Given the set Ω, the set A↑P (Ω)is a
σ-algebra over Ωif the following properties are satisﬁed:
1.Ω↓A;
2.ifA↓A, then A↓A(closedness under complements ); and
3.if we have Ai↓A for all i, then/uniontext∞
i=1Ai↓A (closedness under
countable unions ).
Note that the three properties of σ-algebras correspond to character-
istics we universally expect when working with random experiments.
Namely, that we are able to reason about the event Ωthat any of the
possible outcomes occur, that we are able to reason about an event
not occurring, and that we are able to reason about events that are
composed of multiple (smaller) events.
Example 1.2: Event space of throwing a die
The event space Acan also be thought of as “how much infor-
mation is available about the experiment”. For example, if the
experiment is a throw of a die and Ωis the set of possible values
on the die: Ω={1, . . . , 6 }, then the following Aimplies that the
observer cannot distinguish between 1 and 3:
A.={∅,Ω,{1, 3, 5 },{2, 4, 6 }}.
Intuitively, the observer only understands the parity of the face of
the die.
Deﬁnition 1.3(Probability measure) .Given the set Ωand the σ-algebra
Aover Ω, the function
P:A→R
is aprobability measure onAif the Kolmogorov axioms are satisﬁed:
1.0↔P(A)↔1 for any A↓A;
2.P(Ω)=1; and






















































interesting take


## Page 14

4p r o b a b i l i s t i c a r t i f i c i a l i n t e l l i g e n c e
3.P(/uniontext∞
i=1Ai)=∑∞
i=1P(Ai)for any countable set of mutually disjoint
events {Ai↓A } i.2 2We say that a set of sets {Ai}iis disjoint
if for all i↗=jwe have Ai↘Aj=∅.
Remarkably, all further statements about probability follow from these
three natural axioms. For an event A↓A, we call P(A)theprobability
ofA. We are now ready to deﬁne a probability space.
Deﬁnition 1.4(Probability space) .Aprobability space is a triple (Ω,A,P)
where
•Ωis a sample space,
•Ais aσ-algebra over Ω, and
•Pis a probability measure on A.
Example 1.5: Borel σ-algebra over R
In our context, we often have that Ωis the set of real numbers R
or a compact subset of it. In this case, a natural event space is the
σ-algebra generated by the set of events
Ax.={x≃↓Ω:x≃↔x}.
The smallest σ-algebra Acontaining all sets Axis called the Borel
σ-algebra .Acontains all “reasonable” subsets of Ω(except for
some pathological examples). For example, Aincludes all single-
ton sets {x}, as well as all countable unions of intervals.
In the case of discrete Ω, in fact A=P(Ω), i.e., the Borel σ-
algebra contains allsubsets of Ω.
1.1.2Random Variables
The set Ωis often rather complex. For example, take Ωto be the set of
all possible graphs on nvertices. Then the outcome of our experiment
is a graph. Usually, we are not interested in a speciﬁc graph but rather
a property such as the number of edges, which is shared by many
graphs. A function that maps a graph to its number of edges is a
random variable.
Deﬁnition 1.6(Random variable) .Arandom variable X is a function
X:Ω→T
where Tis called target space of the random variable,3and where X3For a random variable that maps a
graph to its number of edges, T=N0.
For our purposes, you can generally as-
sume T↑R.respects the information available in the σ-algebra A. That is,4
4In our example of throwing a die, X
should assign the same value to the out-
comes 1, 3, 5.⇐S↑T:{ω↓Ω:X(ω)↓S}↓A .( 1.1)






















































notintoomuch detail
upuntil something
KXER
what wewent
sample graphs
allelenantry
Entitymins


## Page 15

fundamentals of inference 5
Concrete values xof a random variable Xare often referred to as states
orrealizations ofX. The probability that Xtakes on a value in S↑Tis
P(X↓S)=P({ω↓Ω:X(ω)↓S}).( 1.2)
1.1.3Distributions
Consider a random variable Xon a probability space (Ω,A,P), where
Ωis a compact subset of R, and Athe Borel σ-algebra.
In this case, we can refer to the probability that Xassumes a particular
state or set of states by writing
pX(x).=P(X=x)(in the discrete setting), ( 1.3)
PX(x).=P(X↔x).( 1.4)
Note that “ X=x” and “ X↔x” are merely events (that is, they char-
acterize subsets of the sample space Ωsatisfying this condition) which
are in the Borel σ-algebra, and hence their probability is well-deﬁned.
Hereby, pXand PXare referred to as the probability mass function
(PMF) and cumulative distribution function (CDF) of X, respectively.
Note that we can also implicitly deﬁne probability spaces through ran-
dom variables and their associated PMF/CDF, which is often very con-
venient.
We list some common examples for discrete distributions in appendix A. 1.1.
Further, note that for continuous variables, P(X=x)=0. Here, in-
stead we typically use the probability density function (PDF), to which
we (with slight abuse of notation) also refer with pX. We discuss den-
sities in greater detail in section 1.1.4.
We call the subset S↑Tof the domain of a PMF or PDF pXsuch that
all elements x↓Shave positive probability, pX(x)>0, the support of
the distribution pX. This quantity is denoted by X(Ω).
1.1.4Continuous Distributions
As mentioned, a continuous random variable can be characterized by
itsprobability density function (PDF). But what is a density? We can
derive some intuition from physics.
LetMbe a (non-homogeneous) physical object, e.g., a rock. We com-
monly use m(M)and vol (M)to refer to its mass and volume, respec-
tively. Now, consider for a point x↓Mand a ball Br(x)around xwith
radius rthe following quantities:
lim
r→0vol(Br(x)) = 0 lim
r→0m(Br(x)) = 0.






















































Ifthatare0
astheballgetssnaker sodoes massandvolume


## Page 16

6p r o b a b i l i s t i c a r t i f i c i a l i n t e l l i g e n c e
They appear utterly uninteresting at ﬁrst, yet, if we divide them, we
get what is called the density ofMatx.
lim
r→0m(Br(x))
vol(Br(x)).=ε(x).
We know that the relationship between density and mass is described
by the following formula:
m(M)=/integraldisplay
Mε(x)dx.
In other words, the density is to be integrated. For a small region I
around x, we can approximate m(I)⇒ε(x)·vol(I).
Crucially, observe that even though the mass of any particular point
xis zero, i.e., m({x})= 0, assigning a density ε(x)toxis useful for
integration and approximation. The same idea applies to continuous
random variables, only that volume corresponds to intervals on the
real line and mass to probability. Recall that probability density func-
tions are normalized such that their probability mass across the entire
real line integrates to one.
⇑202x0.00.10.20.30.4N(x; 0, 1)
Figure 1.1: PDF of the standard normal
distribution. Observe that the PDF is
symmetric around the mode.Example 1.7: Normal distribution / Gaussian
A famous example of a continuous distribution is the normal dis-
tribution , also called Gaussian . We say, a random variable Xis
normally distributed ,X⇓N (µ,σ2), if its PDF is
N(x;µ,σ2).=1⇔
2ϱσ2exp/parenleftbigg
⇑(x⇑µ)2
2σ2/parenrightbigg
.( 1.5)
We have E[X]=µand Var [X]=σ2. Ifµ=0 and σ2=1, this dis-
tribution is called the standard normal distribution . The Gaussian
CDF cannot be expressed in closed-form.
Note that the mean of a Gaussian distribution coincides with the
maximizer of its PDF, also called mode of a distribution.
We will focus in the remainder of this chapter on continuous distribu-
tions, but the concepts we discuss extend mostly to discrete distribu-
tions simply by “replacing integrals by sums”.
1.1.5Joint Probability
A joint probability (as opposed to a marginal probability) is the prob-
ability of two or more events occurring simultaneously:
P(A,B).=P(A↘B).( 1.6)






















































meaning


## Page 17

fundamentals of inference 7
In terms of random variables, this concept extends to joint distribu-
tions. Instead of characterizing a single random variable, a joint dis-
tribution is a function pX:Rn→R, characterizing a random vector
X.=[X1···Xn]↖. For example, if the Xiare discrete, the joint distri-
bution characterizes joint probabilities of the form
P(X=[x1,..., xn])=P(X1=x1,..., Xn=xn),
and hence describes the relationship among all variables Xi. For this
reason, a joint distribution is also called a generative model . We use Xi:j
to denote the random vector [Xi···Xj]↖.
We can “sum out” (respectively “integrate out”) variables from a joint
distribution in a process called “marginalization”:
Fact 1.8(Sum rule) .We have that
p(x1:i⇑1,xi+1:n)=/integraldisplay
Xi(Ω)p(x1:i⇑1,xi,xi+1:n)dxi.( 1.7)
1.1.6Conditional Probability
Conditional probability updates the probability of an event Agiven
some new information, for example, after observing the event B.
Deﬁnition 1.9(Conditional probability) .Given two events Aand B
such that P(B)>0, the probability of Aconditioned on Bis given as
P(A|B).=P(A,B)
P(B).( 1.8)
A
B
Ω
Figure 1.2: Conditioning an event Aon
another event Bcan be understood as re-
placing the universe of all possible out-
comes Ωby the observed outcomes B.
Then, the conditional probability is sim-
ply expressing the likelihood of Agiven
that Boccurred.Simply rearranging the terms yields,
P(A,B)=P(A|B)·P(B)=P(B|A)·P(A).( 1.9)
Thus, the probability that both Aand Boccur can be calculated by
multiplying the probability of event Aand the probability of Bcondi-
tional on Aoccurring.
We say Z⇓X|Y=y(or simply Z⇓X|y) ifZfollows the conditional
distribution
pX|Y(x|y).=pX,Y(x,y)
pY(y).( 1.10)
IfXand Yare discrete, we have that pX|Y(x|y)= P(X=x|Y=y)
as one would naturally expect.
Extending eq. ( 1.9) to arbitrary random vectors yields the product rule
(also called the chain rule of probability ).






















































whygeneraticemodel
whats thepoint ofthis


## Page 18

8p r o b a b i l i s t i c a r t i f i c i a l i n t e l l i g e n c e
Fact 1.10(Product rule) .Given random variables X 1:n,
p(x1:n)=p(x1)·n
∏
i=2p(xi|x1:i⇑1).( 1.11)
Combining sum rule and product rule, we can compute marginal
probabilities too:
p(x)=/integraldisplay
Y(Ω)p(x,y)dy=/integraldisplay
Y(Ω)p(x|y)·p(y)dy ﬁrst using the sum rule ( 1.7) then the
product rule ( 1.11)(1.12)
This is called the law of total probability (LOTP), which is colloquially
often referred to as conditioning onY. If it is difﬁcult to compute p(x)
directly, conditioning can be a useful technique when Yis chosen such
that the densities p(x|y)andp(y)are straightforward to understand.
1.1.7Independence
Two random vectors Xand Yareindependent (denoted X↙Y) if and
only if knowledge about the state of one random vector does not affect
the distribution of the other random vector, namely if their conditional
CDF (or in case they have a joint density, their conditional PDF) sim-
pliﬁes to
PX|Y(x|y)= PX(x),pX|Y(x|y)= pX(x).( 1.13)
For the conditional probabilities to be well-deﬁned, we need to assume
that pY(y)>0.
The more general characterization of independence is that XandYare
independent if and only if their joint CDF (or in case they have a joint
density, their joint PDF) can be decomposed as follows:
PX,Y(x,y)= PX(x)·PY(y),pX,Y(x,y)= pX(x)·pY(y).( 1.14)
The equivalence of the two characterizations (when pY(y)>0) is eas-
ily proven using the product rule: pX,Y(x,y)= pY(y)·pX|Y(x|y).
A “weaker” notion of independence is conditional independence.5 5We discuss in remark 1.11 how
“weaker” is to be interpreted in this con-
text.Two random vectors XandYareconditionally independent given a ran-
dom vector Z(denoted X↙Y|Z) iff, given Z, knowledge about the
value of one random vector Ydoes not affect the distribution of the
other random vector X, namely if
PX|Y,Z(x|y,z)= PX|Z(x|z),( 1.15a)
pX|Y,Z(x|y,z)= pX|Z(x|z).( 1.15b)






















































independence is
rarely the case
lanbetricky
unlessjustassumed
instead conditional indip
don't getthis
exactly


## Page 19

fundamentals of inference 9
Similarly to independence, we have that Xand Yare conditionally
independent given Zif and only if their joint CDF or joint PDF can be
decomposed as follows:
PX,Y|Z(x,y|z)= PX|Z(x|z)·PY|Z(y|z),( 1.16a)
pX,Y|Z(x,y|z)= pX|Z(x|z)·pY|Z(y|z).( 1.16b)
Remark 1.11: Common causes
How can conditional independence be understood as a “weaker”
notion of independence? Clearly, conditional independence does
not imply independence: a trivial example is X↙X|X↗=∝X↙
X.6 6X↙X|Xis true trivially. Neither does independence imply conditional independence:
for example, X↙Y↗=∝X↙Y|X+Y.7 7Knowing XandX+Yalready implies
the value of Y, and hence, X↗↙Y|X+
Y. When we say that conditional independence is a weaker notion we
mean to emphasize that Xand Ycan be “made” (conditionally)
independent by conditioning on the “right” Zeven if XandYare
dependent. This is known as Reichenbach’s common cause principle
which says that for any two random variables X↗↙Ythere exists a
random variable Z(which may be XorY) that causally inﬂuences
both XandY, and which is such that X↙Y|Z.
1.1.8Directed Graphical Models
Directed graphical models (also called Bayesian networks ) are often
used to visually denote the (conditional) independence relationships
of a large number of random variables. They are a schematic repre-
sentation of the factorization of the generative model into a product
of conditional distributions as a directed acyclic graph. Given the se-
quence of random variables {Xi}n
i=1, their generative model can be
expressed as
p(x1:n)=n
∏
i=1p(xi|parents (xi)) (1.17)
where parents (xi)is the set of parents of the vertex Xiin the directed
graphical model. In other words, the parenthood relationship encodes
a conditional independence of a random variable Xwith a random
variable Ygiven their parents:88More generally, vertices uand vare
conditionally independent given a set of
vertices ZifZ d-separates u andv, which
we will not cover in depth here.
X↙Y|parents (X), parents (Y).( 1.18)
Equation ( 1.17) simply uses the product rule and the conditional in-
dependence relationships to factorize the generative model. This can
greatly reduce the model’s complexity, i.e., the length of the product.
Yc
X1 ···
Xn
a1
an
Figure 1.3: Example of a directed
graphical model. The random vari-
ables X1,..., Xnare mutually indepen-
dent given the random variable Y. The
squared rectangular nodes are used to
represent dependencies on parameters
c,a1,..., an.






















































meaning
dontgetthis example


## Page 20

10 probabilistic artificial intelligence
An example of a directed graphical model is given in ﬁg. 1.3. Circu-
lar vertices represent random quantities (i.e., random variables). In
contrast, square vertices are commonly used to represent determinis-
tic quantities (i.e., parameters that the distributions depend on). In
the given example, we have that Xiis conditionally independent of all
other Xjgiven Y.Plate notation is a condensed notation used to repre-
sent repeated variables of a graphical model. An example is given in
ﬁg.1.4.
Y
c
Xi
ai
i↓1:n
Figure 1.4: The same directed graphical
model as in ﬁg. 1.3using plate notation.1.1.9Expectation
Theexpected value ormean E[X]of a random vector Xis the (asymp-
totic) arithmetic mean of an arbitrarily increasing number of indepen-
dent realizations of X. That is,9 9In inﬁnite probability spaces, absolute
convergence of E[X]is necessary for the
existence of E[X]. E[X].=/integraldisplay
X(Ω)x·p(x)dx (1.19)
A very special and often used property of expectations is their lin-
earity , namely that for any random vectors Xand YinRnand any
A↓Rm′n,b↓Rmit holds that
E[AX+b]=AE[X]+band E[X+Y]=E[X]+E[Y].( 1.20)
Note that XandYdo not necessarily have to be independent! Further,
ifXandYare independent then
E/bracketleftig
XY↖/bracketrightig
=E[X]·E[Y]↖.( 1.21)
The following intuitive lemma can be used to compute expectations of
transformed random variables.
Fact 1.12(Law of the unconscious statistician, LOTUS) .
E[g(X)]=/integraldisplay
X(Ω)g(x)·p(x)dx (1.22)
where g:X(Ω)→Rnis a “nice” function10 10gbeing a continuous function, which
is either bounded or absolutely inte-
grable (i.e.,/integraltext
|g(x)|p(x)dx<∞), is suf-
ﬁcient. This is satisﬁed in most cases.andXis a continuous ran-
dom vector. The analogous statement with a sum replacing the integral
holds for discrete random variables.
This is a nontrivial fact that can be proven using the change of variables
formula which we discuss in section 1.1.11.
Similarly to conditional probability, we can also deﬁne conditional ex-
pectations. The expectation of a continuous random vector Xgiven
thatY=yis deﬁned as
E[X|Y=y].=/integraldisplay
X(Ω)x·pX|Y(x|y)dx.( 1.23)






















































summary
orsumsto
discrete
interpretation
L
later used


## Page 21

fundamentals of inference 11
Observe that E[X|Y=·]deﬁnes a deterministic mapping from yto
E[X|Y=y]. Therefore, E[X|Y]is itself a random vector:
E[X|Y](ω)= E[X|Y=Y(ω)] (1.24)
where ω↓Ω. This random vector E[X|Y]is called the conditional
expectation ofXgiven Y.
Analogously to the law of total probability ( 1.12), one can condition an
expectation on another random vector. This is known as the tower rule
or the law of total expectation (LOTE).
Theorem 1.13(Tower rule) .Given random vectors XandY, we have
EY[EX[X|Y]]=E[X].( 1.25)
Proof sketch. We only prove the case where XandYhave a joint den-
sity. We have
E[E[X|Y]]=/integraldisplay/parenleftbigg/integraldisplay
x·p(x|y)dx/parenrightbigg
p(y)dy
=/integraldisplay/integraldisplay
x·p(x,y)dxdy by deﬁnition of conditional densities
(1.10)
=/integraldisplay
x/integraldisplay
p(x,y)dydx by Fubini’s theorem
=/integraldisplay
x·p(x)dx using the sum rule ( 1.7)
=E[X].
1.1.10 Covariance and Variance
Given two random vectors XinRnand YinRm, their covariance is
deﬁned as
Cov[X,Y].=E/bracketleftig
(X⇑E[X])(Y⇑E[Y])↖/bracketrightig
(1.26)
=E/bracketleftig
XY↖/bracketrightig
⇑E[X]·E[Y]↖(1.27)
=Cov[Y,X]↖↓Rn′m.( 1.28)
Covariance measures the linear dependence between two random vec-
tors since a direct consequence of its deﬁnition ( 1.26) is that given lin-
ear maps A↓Rn≃′n,B↓Rm≃′m, vectors c↓Rn≃,d↓Rm≃and random
vectors XinRnandYinRm, we have that
Cov[AX+c,BY+d]=ACov[X,Y]B↖.( 1.29)


## Page 22

12 probabilistic artificial intelligence
Two random vectors XandYare said to be uncorrelated if and only if
Cov[X,Y]=0. Note that if Xand Yare independent, then eq. ( 1.21)
implies that Xand Yare uncorrelated. The reverse does not hold in
general.
Remark 1.14: Correlation
The correlation of the random vectors Xand Yis a normalized
covariance,
Cor[X,Y](i,j).=Cov/bracketleftbig
Xi,Yj/bracketrightbig
/radicalig
Var[Xi]Var/bracketleftbig
Yj/bracketrightbig↓[⇑1, 1].( 1.30)
Two random vectors Xand Yare therefore uncorrelated if and
only if Cor [X,Y]=0.
There is also a nice geometric interpretation of covariance and
correlation. For zero mean random variables XandY, Cov [X,Y]
is an inner product.11 11That is,
•Cov[X,Y]is symmetric,
•Cov[X,Y]is linear (here we use
EX=EY=0), and
•Cov[X,X]∞0.The cosine of the angle θbetween XandY(that are not determin-
istic) coincides with their correlation,
cosθ=Cov[X,Y]
∈X∈∈Y∈=Cor[X,Y]. using the Euclidean inner product
formula, Cov [X,Y]=∈X∈∈Y∈cosθ(1.31)
cosθis also called a cosine similarity . Thus,
θ=arccos Cor [X,Y].( 1.32)
For example, if XandYare uncorrelated, then they are orthogonal
in the inner product space. If Cor [X,Y]=⇑1 then θ∋ϱ(that is,
XandY“point in opposite directions”), whereas if Cor [X,Y]=1
then θ∋0 (that is, XandY“point in the same direction”).
The covariance of a random vector XinRnwith itself is called its
variance :
Var[X].=Cov[X,X] (1.33)
=E/bracketleftig
(X⇑E[X])(X⇑E[X])↖/bracketrightig
(1.34)
=E/bracketleftig
XX↖/bracketrightig
⇑E[X]·E[X]↖(1.35)
=
Cov[X1,X1]··· Cov[X1,Xn]
.........
Cov[Xn,X1]···Cov[Xn,Xn]
.( 1.36)
The scalar variance Var [X]of a random variable Xis a measure of un-
certainty about the value of Xsince it measures the average squared


## Page 23

fundamentals of inference 13
deviation from E[X]. We will see that the eigenvalue spectrum of a
covariance matrix can serve as a measure of uncertainty in the multi-
variate setting.12 12Themultivariate setting (as opposed to
theunivariate setting) studies the joint
distribution of multiple random vari-
ables. Remark 1.15: Standard deviation
The length of a random variable Xin the inner product space
described in remark 1.14is called its standard deviation ,
∈X∈=/radicalig
Cov[X,X]=/radicalig
Var[X].=σ[X].( 1.37)
That is, the longer a random variable is in the inner product space,
the more “uncertain” we are about its value. If a random variable
has length 0, then it is deterministic.
The variance of a random vector Xis also called the covariance matrix
ofXand denoted by ΣX(orΣif the correspondence to Xis clear from
context). A covariance matrix is symmetric by deﬁnition due to the
symmetry of covariance, and is always positive semi-deﬁnite ? Problem 1.4 .
Two useful properties of variance are the following:
•It follows from eq. ( 1.29) that for any linear map A↓Rm′nand
vector b↓Rm,
Var[AX+b]=AVar[X]A↖.( 1.38)
In particular, Var [⇑X]=Var[X].
•It follows from the deﬁnition of variance ( 1.34) that for any two
random vectors XandY,
Var[X+Y]=Var[X]+Var[Y]+2Cov [X,Y].( 1.39)
In particular, if Xand Yare independent then the covariance term
vanishes and Var [X+Y]=Var[X]+Var[Y].
Analogously to conditional probability and conditional expectation,
we can also deﬁne conditional variance. The conditional variance of a
random vector Xgiven another random vector Yis the random vector
Var[X|Y].=E/bracketleftig
(X⇑E[X|Y])(X⇑E[X|Y])↖Y/bracketrightig
.( 1.40)
Intuitively, the conditional variance is the remaining variance when
we use E[X|Y]to predict Xrather than if we used E[X]. One can
also condition a variance on another random vector, analogously to
the laws of total probability ( 1.12) and expectation ( 1.25).


## Page 24

14 probabilistic artificial intelligence
Theorem 1.16(Law of total variance, LOTV) .
Var[X]=EY[Var X[X|Y]]+Var Y[EX[X|Y]].( 1.41)
Here, the ﬁrst term measures the average deviation from the mean of X
across realizations of Yand the second term measures the uncertainty
in the mean of Xacross realizations of Y. In section 2.2, we will see
that both terms have a meaningful characterization in the context of
probabilistic inference.
Proof sketch of LOTV. To simplify the notation, we present only a proof
for the univariate setting.
Var[X]=E/bracketleftig
X2/bracketrightig
⇑E[X]2
=E/bracketleftig
E/bracketleftig
X2|Y/bracketrightig/bracketrightig
⇑E[E[X|Y]]2by the tower rule ( 1.25)
=E/bracketleftig
Var[X|Y]+E[X|Y]2/bracketrightig
⇑E[E[X|Y]]2by the deﬁnition of variance ( 1.35)
=E[Var[X|Y]]+
E/bracketleftig
E[X|Y]2/bracketrightig
⇑E[E[X|Y]]2
=E[Var[X|Y]]+Var[E[X|Y]]. by the deﬁnition of variance ( 1.35)
1.1.11 Change of Variables
It is often useful to understand the distribution of a transformed ran-
dom variable Y=g(X)that is deﬁned in terms of a random variable
X, whose distribution is known. Let us ﬁrst consider the univariate
setting. We would like to express the distribution of Yin terms of the
distribution of X, that is, we would like to ﬁnd
PY(y)= P(Y↔y)=P(g(X)↔y)=P
X↔g⇑1(y)
.( 1.42)
When the random variables are continuous, this probability can be ex-
pressed as an integration over the domain of X. We can then use the
substitution rule of integration to “change the variables” to an inte-
gration over the domain of Y. Taking the derivative yields the density
pY.13There is an analogous change of variables formula for the multi-13The full proof of the change of vari-
ables formula in the univariate setting
can be found in section 6.7.2of “Math-
ematics for machine learning” (Deisen-
roth et al., 2020 ).variate setting.
Fact 1.17(Change of variables formula) .LetXbe a random vector
inRnwith density p Xand let g:Rn→Rnbe a differentiable and
invertible function. Then Y=g(X)is another random variable, whose
density can be computed based on p Xandgas follows:
pY(y)=pX(g⇑1(y))·det
Dg⇑1(y) (1.43)


## Page 25

fundamentals of inference 15
where Dg⇑1(y)is the Jacobian of g⇑1evaluated at y.
Here, the termdet
Dg⇑1(y)measures how much a unit volume
changes when applying g. Intuitively, the change of variables swaps
the coordinate system over which we integrate. The factordet
Dg⇑1(y)
corrects for the change in volume that is caused by this change in co-
ordinates.
Intuitively, you can think of the vector ﬁeld gas a perturbation to X,
“pushing” the probability mass around. The perturbation of a density
pXbygis commonly denoted by the pushforward
gωpX.=pYwhere Y=g(X).( 1.44)
This concludes our quick tour of probability theory, and we are well-
prepared to return to the topic of probabilistic inference.
1.2Probabilistic Inference
Recall the logical implication “If it is raining, the ground is wet.” from
the beginning of this chapter. Suppose that we look outside a window
and see that it is not raining: will the ground be dry? Logical reason-
ing does not permit drawing an inference of this kind, as there might
be reasons other than rain for which the ground could be wet (e.g.,
sprinklers). However, intuitively, by observing that it is not raining,
we have just excluded the possibility that the ground is wet because of
rain, and therefore we would deem it “more likely” that the ground is
dry than before. In other words, if we were to walk outside now and
the ground was wet, we would be more surprised than we would have
been if we had not looked outside the window before.
As humans, we are constantly making such “plausible” inferences of
our beliefs: be it about the weather, the outcomes of our daily deci-
sions, or the behavior of others. Probabilistic inference is the process of
updating such a prior belief P
W
to a posterior belief P
W|R
upon
observing Rwhere — to reduce clutter — we write Wfor “The ground
is wet” and Rfor “It is raining”.
The computational rule of probabilistic inference is Bayes’ rule.
Theorem 1.18(Bayes’ rule) .Given random vectors XinRnandYin
Rm, we have for any x↓Rn,y↓Rmthat
p(x|y)=p(y|x)·p(x)
p(y).( 1.45)


## Page 26

16 probabilistic artificial intelligence
Proof. Bayes’ rule is a direct consequence of the deﬁnition of condi-
tional densities ( 1.10) and the product rule ( 1.11).
It is useful to consider the meaning of each term separately:
•theposterior p (x|y)is the updated belief about xafter observing y,
•theprior p (x)is the initial belief about x,
•the(conditional) likelihood p (y|x)describes how likely the observa-
tions yare under a given value x,
•thejoint likelihood p (x,y)= p(y|x)p(x)combines prior and likeli-
hood,
•themarginal likelihood p (y)describes how likely the observations y
are across all values of x.
The marginal likelihood can be computed using the sum rule ( 1.7) or
the law of total probability ( 1.12),
p(y)=/integraldisplay
X(Ω)p(y|x)·p(x)dx.( 1.46)
Note, however, that the marginal likelihood is simply normalizing the
conditional distribution to integrate to one, and therefore a constant
with respect to x. For this reason, p(y)is commonly called the normal-
izing constant .
Example 1.19: Plausible inferences
Let us conﬁrm our intuition from the above example. The logical
implication “If it is raining, the ground is wet.” (denoted R→W)
can be succinctly expressed as P(W|R)=1. Since P(W)↔1, we
know that
P(R|W)=P(W|R)·P(R)
P(W)=P(R)
P(W)∞P(R).
That is, observing that the ground is wet makes it more likely to
be raining. From P(R|W)∞P(R)we know P
R|W↔P
R
,14 14since P
X=1⇑P(X)
which leads us to follow that
P
W|R=P
R|W·P(W)
P
R ↔P(W),
that is, having observed it not to be raining made the ground less
likely to be wet.
Example 1.19is called a plausible inference because the observation of R
does not completely determine the truth value of W, and hence, does
not permit logical inference. In the case, however, that logical inference
is permitted, it coincides with probabilistic inference.


## Page 27

fundamentals of inference 17
Example 1.20: Logical inferences
For example, if we were to observe that the ground is not wet,
then logical inference implies that it must not be raining: W→R.
This is called the contrapositive ofR→W.
Indeed, by probabilistic inference, we obtain analogously
P
R|W=P
W|R·P(R)
P
W =(1⇑P(W|R))·P(R)
P
W =0. asP(W|R)=1
Observe that a logical inference does not depend on the prior P(R):
Even if the prior was P(R)=1 in example 1.20, after observing that
the ground is not wet, we are forced to conclude that it is not raining to
maintain logical consistency. The examples highlight that while logical
inference does not require the notion of a prior, plausible ( probabilistic !)
inference does.
1.2.1Where do priors come from?
The computational rule of probabilistic inference as shown in eq. ( 1.45)
necessitates the speciﬁcation of a prior p(x). Different priors can lead
to the deduction of dramatically different posteriors, as one can easily
see by considering the extreme cases of a prior that is a point density
atx=x0and a prior that is “uniform” over Rn.15In the former case,15The latter is not a valid probability dis-
tribution, but we can still derive mean-
ing from the posterior as we discuss in
remark 1.22.the posterior will be a point density at x0regardless of the likelihood.
In other words, no evidence can alter the “prior belief” the learner
ascribed to x. In the latter case, the learner has “no prior belief”, and
therefore the posterior will be proportional to the likelihood. Both
steps of probabilistic inference are perfectly valid, though one might
debate which prior is more reasonable.
Someone who follows the Bayesian interpretation of probability might
argue that everything is conditional, meaning that the prior is simply
a posterior of all former observations. While this might seem natural
(“my world view from today is the combination of my world view
from yesterday and the observations I made today”), this lacks an
explanation for “the ﬁrst day”. Someone else who is more inclined
towards the frequentist interpretation might also object to the exis-
tence of a prior belief altogether, arguing that a prior is subjective and
therefore not a valid or desirable input to a learning algorithm. Put
differently, a frequentist “has the belief not to have any belief”. This is
perfectly compatible with probabilistic inference, as long as the prior
is chosen to be noninformative :
p(x)∝const. ( 1.47)


## Page 28

18 probabilistic artificial intelligence
Choosing a noninformative prior in the absence of any evidence is
known as the principle of indifference or the principle of insufﬁcient reason ,
which dates back to the famous mathematician Pierre-Simon Laplace.
Example 1.21: Why be indifferent?
Consider a criminal trial with three suspects, A, B, and C. The
collected evidence shows that suspect C can not have committed
the crime, however it does not yield any information about sus-
pects A and B. Clearly, any distribution respecting the data must
assign zero probability of having committed the crime to suspect
C. However, any distribution interpolating between (1, 0, 0 )and
(0, 1, 0 )respects the data. The principle of indifference suggests
that the desired distribution is (1
2,1
2,0), and indeed, any alterna-
tive distribution seems unreasonable.
Remark 1.22: Noninformative and improper priors
It is not necessarily required that the prior p(x)is a valid distri-
bution (i.e., integrates to 1). Consider for example, the noninfor-
mative prior p(x)∝1{x↓I}where I↑Rnis an inﬁnitely large
interval. Such a prior which is not a valid distribution is called an
improper prior . We can still derive meaning from the posterior of a
given likelihood and (improper) prior as long as the posterior is a
valid distribution.
Laplace’s principle of indifference can be generalized to cases where
some evidence is available. The maximum entropy principle , originally
proposed by Jaynes ( 1968 ), states that one should choose as prior from
all possible distributions that are consistent with prior knowledge, the
one that makes the least “additional assumptions”, i.e., is the least
“informative”. In philosophy, this principle is known as Occam’s razor
or the principle of parsimony . The “informativeness” of a distribution p
is quantiﬁed by its entropy which is deﬁned as
H[p].=Ex⇓p[⇑logp(x)].( 1.48)
The more concentrated pis, the less is its entropy; the more diffuse p
is, the greater is its entropy.16 16We give a thorough introduction to en-
tropy in section 5.4.
In the absence of any prior knowledge, the uniform distribution has
the highest entropy,17and hence, the maximum entropy principle sug-17This only holds true when the set
of possible outcomes of xﬁnite (or
a bounded continuous interval), as in
this case, the noninformative prior is a
proper distribution — the uniform dis-
tribution. In the “inﬁnite case”, there is
no uniform distribution and the nonin-
formative prior can be attained from the
maximum entropy principle as the lim-
iting solution as the number of possible
outcomes of xis increased.gests a noninformative prior (as does Laplace’s principle of indiffer-
ence). In contrast, if the evidence perfectly determines the value of x,
then the only consistent explanation is the point density at x. The
maximum entropy principle characterizes a reasonable choice of prior


## Page 29

fundamentals of inference 19
for these two extreme cases and all cases in between. The computa-
tional rule of probabilistic inference ( 1.45) can in fact be derived as a
consequence of the maximum entropy principle in the sense that the
posterior is the least “informative” distribution among all distributions
that are consistent with the prior and the observations ? Problem 5.7 .
1.2.2Conjugate Priors
If the prior p(x)and posterior p(x|y)are of the same family of distri-
butions, the prior is called a conjugate prior to the likelihood p(y|x).
This is a very desirable property, as it allows us to recursively ap-
ply the same learning algorithm implementing probabilistic inference.
We will see in chapter 2that under some conditions the Gaussian is
self-conjugate . That is, if we have a Gaussian prior and a Gaussian like-
lihood then our posterior will also be Gaussian. This will provide us
with the ﬁrst efﬁcient implementation of probabilistic inference.
Example 1.23: Conjugacy of beta and binomial distribution
As an example for conjugacy, we will show that the beta distribu-
tion is a conjugate prior to a binomial likelihood. Recall the PMF
of the binomial distribution
Bin(k;n,θ)=/parenleftbiggn
k/parenrightbigg
θk(1⇑θ)n⇑k(1.49)
and the PDF of the beta distribution,
Beta(θ;α,β)∝θα⇑1(1⇑θ)β⇑1,( 1.50)
We assume the prior θ⇓Beta(α,β)and likelihood k|θ⇓Bin(n,θ).
LetnH=kbe the number of heads and nT=n⇑kthe number of
tails in the binomial trial k. Then,
p(θ|k)∝p(k|θ)p(θ) using Bayes’ rule ( 1.45)
∝θnH(1⇑θ)nTθα⇑1(1⇑θ)β⇑1
=θα+nH⇑1(1⇑θ)β+nT⇑1.
Thus, θ|k⇓Beta(α+nH,β+nT).
This same conjugacy can be shown for the multivariate general-
ization of the beta distribution, the Dirichlet distribution , and the
multivariate generalization of the binomial distribution, the multi-
nomial distribution .


## Page 30

20 probabilistic artificial intelligence
1.2.3Tractable Inference with the Normal Distribution
Using arbitrary distributions for learning and inference is computa-
tionally very expensive when the number of dimensions is large —
even in the discrete setting. For example, computing marginal distri-
butions using the sum rule yields an exponentially long sum in the
size of the random vector. Similarly, the normalizing constant of the
conditional distribution is a sum of exponential length. Even to rep-
resent any discrete joint probability distribution requires space that is
exponential in the number of dimensions (cf. ﬁg. 1.5).X1··· Xn⇑1Xn P(X1:n)
0 ··· 0 0 0.01
0 ··· 0 1 0.001
0 ··· 1 0 0.213
............
1 ··· 1 1 0.0003
Figure 1.5: A table representing a joint
distribution of nbinary random vari-
ables. The table has 2nrows. The num-
ber of parameters is 2n⇑1 since the ﬁ-
nal probability is determined by all other
probabilities as they must sum to one.One strategy to get around this computational blowup is to restrict
the class of distributions. Gaussians are a popular choice for this pur-
pose since they have extremely useful properties: they have a compact
representation and — as we will see in chapter 2— they allow for
closed-form probabilistic inference.
In eq. ( 1.5), we have already seen the PDF of the univariate Gaus-
sian distribution. A random vector XinRnisnormally distributed ,
X⇓N (µ,Σ), if its PDF is
N(x;µ,Σ).=1
det(2ϱΣ)exp/parenleftbigg
⇑1
2(x⇑µ)↖Σ⇑1(x⇑µ)/parenrightbigg
(1.51)
where µ↓Rnis the mean vector and Σ↓Rn′nthe covariance matrix
? Problem 1.11 . We call Λ.=Σ⇑1theprecision matrix .Xis also called a Gaussian
random vector (GRV). N(0,I)is the multivariate standard normal distri-
bution . We call a Gaussian isotropic if its covariance matrix is of the
form Σ=σ2Ifor some σ2↓R. In this case, the sublevel sets of the
PDF are perfect spheres as can be seen in ﬁg. 1.6.
x⇑2
0
2y
⇑2020.00.10.2x⇑2
0
2y
⇑2020.00.20.4Figure 1.6: Shown are the PDFs of two-
dimensional Gaussians with mean 0and
covariance matrices
Σ1.=10
01
,Σ2.=1 0.9
0.9 1
respectively.
Note that a Gaussian can be represented using only O
n2
parameters.
In the case of a diagonal covariance matrix, which corresponds to n
independent univariate Gaussians ? Problem 1.8 , we just need O(n)parameters.






















































easyto
in
meaning
independence 0covariance


## Page 31

fundamentals of inference 21
In eq. ( 1.51), we assume that the covariance matrix Σis invertible,
i.e., does not have the eigenvalue 0. This is not a restriction since it
can be shown that a covariance matrix has a zero eigenvalue if and
only if there exists a deterministic linear relationship between some
variables in the joint distribution ? Problem 1.6 . As we have already seen that a
covariance matrix does not have negative eigenvalues ? Problem 1.4 , this ensures
thatΣandΛare positive deﬁnite.18 18The inverse of a positive deﬁnite ma-
trix is also positive deﬁnite.
An important property of the normal distribution is that it is closed
under marginalization and conditioning.
Theorem 1.24(Marginal and conditional distribution) .? Problem 1.9 Con-
sider the Gaussian random vector Xand ﬁx index sets A ↑[n]and
B↑[n]. Then, we have that for any such marginal distribution ,
XA⇓N(µA,ΣAA), ByµAwe denote [µi1,..., µik]where
A={i1,...ik}.ΣAAis deﬁned
analogously.(1.52)
and that for any such conditional distribution ,
XA|XB=xB⇓N(µA|B,ΣA|B)where (1.53a)
µA|B.=µA+ΣABΣ⇑1
BB(xB⇑µB), Here, µAcharacterizes the prior belief
andΣABΣ⇑1
BB(xB⇑µB)represents “how
different” xBis from what was
expected.(1.53b)
ΣA|B.=ΣAA⇑ΣABΣ⇑1
BBΣBA.( 1.53c)
Theorem 1.24provides a closed-form characterization of probabilis-
tic inference for the case that random variables are jointly Gaussian.
We will discuss in chapter 2, how this can be turned into an efﬁcient
inference algorithm.
Observe that upon inference, the variance can only shrink! Moreover,
how much the variance is reduced depends purely on where the obser-
vations are made (i.e., the choice of B) but not on what the observations
are. In contrast, the posterior mean µA|Bdepends afﬁnely on µB. These
are special properties of the Gaussian and do not generally hold true
for other distributions.
It can be shown that Gaussians are additive and closed under afﬁne
transformations ? Problem 1.10 . The closedness under afﬁne transformations ( 1.78)
implies that a Gaussian X⇓N(µ,Σ)is equivalently characterized as
X=Σ1/2Y+µ.( 1.54)
where Y⇓N(0,I)andΣ1/2is the square root of Σ.19Importantly, this19More details on the square root of a
symmetric and positive deﬁnite matrix
can be found in appendix A. 2.implies together with theorem 1.24and additivity ( 1.79) that:
Any afﬁne transformation of a Gaussian random vector
is a Gaussian random vector.






















































eatit noise
semi decomposition
meaning
affinely


## Page 32

22 probabilistic artificial intelligence
A consequence of this is that given any jointly Gaussian random vec-
torsXAandXB,XAcan be expressed as an afﬁne function of XBwith
added independent Gaussian noise. Formally, we deﬁne
XA.=AXB+b+εwhere ( 1.55a)
A.=ΣABΣ⇑1
BB,( 1.55b)
b.=µA⇑ΣABΣ⇑1
BBµB,( 1.55c)
ε⇓N(0,ΣA|B).( 1.55d)
It directly follows from the closedness of Gaussians under afﬁne trans-
formations ( 1.78) that the characterization of XAvia eq. ( 1.55) is equiv-
alent to XA⇓N(µA,ΣAA), and hence, anyGaussian XAcan be mod-
eled as a so-called conditional linear Gaussian , i.e., an afﬁne function of
another Gaussian XBwith additional independent Gaussian noise. We
will use this fact frequently to represent Gaussians in a compact form.
1.3Supervised Learning and Point Estimates
Throughout the ﬁrst part of this manuscript, we will focus mostly on
thesupervised learning problem where we want to learn a function
fε:X→Y
from labeled training data. That is, we are given a collection of labeled
examples, Dn.={(xi,yi)}n
i=1, where the xi↓X areinputs and the
yi↓Yareoutputs (called labels ), and we want to ﬁnd a function ˆfthat
best-approximates fε. It is common to choose ˆffrom a parameter-
ized function class F(Θ), where each function fϱis described by some
parameters ϱ↓Θ.
Ffε
ˆff
Figure 1.7: Illustration of estimation er-
rorandapproximation error .fεdenotes
the true function and ˆfis the best ap-
proximation from the function class F.
We do not specify here, how one could
quantify “error”. For more details, see
appendix A. 3.5.Remark 1.25: What this manuscript is about and not about
As illustrated in ﬁg. 1.7, the restriction to a function class leads to
two sources of error: the estimation error of having “incorrectly”
determined ˆfwithin the function class, and the approximation er-
rorof the function class itself. Choosing a “good” function class
/ architecture with small approximation error is therefore critical
for any practical application of machine learning. We will discuss
various function classes, from linear models to deep neural net-
works, however, determining the “right” function class will not
be the focus of this manuscript. To keep the exposition simple,
we will assume in the following that fε↓F(Θ)with parameters
ϱε↓Θ.






















































key meaning


## Page 33

fundamentals of inference 23
Instead, we will focus on the problem of estimation/inference
within a given function class. We will see that inference in smaller
function classes is often more computationally efﬁcient since the
search space is smaller or — in the case of Gaussians — has a
known tractable structure. On the other hand, larger function
classes are more expressive and therefore can typically better ap-
proximate the ground truth fω.
We differentiate between the task of regression where Y.=Rk,20and20The labels are usually scalar, so k=1.
the task of classiﬁcation where Y.=Cand Cis an m-element set of
classes. In other words, regression is the task of predicting a continu-
ous label, whereas classiﬁcation is the task of predicting a discrete class
label. These two tasks are intimately related: in fact, we can think of
classiﬁcation tasks as a regression problem where we learn a probabil-
ity distribution over class labels. In this regression problem, Y.=∆C
where ∆Cdenotes the set of all probability distributions over the set
of classes Cwhich is an (m→1)-dimensional convex polytope in the
m-dimensional space of probabilities [0, 1]m(cf. Appendix A. 1.2).
For now, let us stick to the regression setting. We will assume that
the observations are noisy, that is, yiiid↑p(·|xi,θω)for some known
conditional distribution p(·|xi,θ)butunknown parameter θω.21Our21The case where the labels are deter-
ministic is the special case of p(·|xi,θω)
being a point density at fω(xi).assumption can equivalently be formulated as
yi=fθ(xi)/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
signal+εi(xi)/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
noise(1.56)
where fθ(xi)is the mean of p(·|xi,θ)andεi(xi)= yi→fθ(xi)is some
independent zero-mean noise, for example (but not necessarily) Gaus-
sian.22When the noise distribution may depend on xi, the noise is said22It is crucial that the assumed noise dis-
tribution accurately reﬂects the noise of
the data. For example, using a (light-
tailed) Gaussian noise model in the pres-
ence of heavy-tailed noise will fail! We
discuss the distinction between light and
heavy tails in Appendix A. 3.2.to be heteroscedastic and otherwise the noise is called homoscedastic .
1.3.1Maximum Likelihood Estimation
A common approach to ﬁnding ˆfis to select the model f↓F(Θ)un-
der which the training data is most likely. This is called the maximum
likelihood estimate (or MLE):
ˆθMLE.=arg max
θ↓Θp(y1:n|x1:n,θ) (1.57)
=arg max
θ↓Θn
∏
i=1p(yi|xi,θ). using the independence of the training
data ( 1.56)






















































meaninginterpretation
hederelateto


## Page 34

24 probabilistic artificial intelligence
Such products of probabilities are often numerically unstable, which
is why one typically takes the logarithm:
=arg max
θ↓Θn
∑
i=1logp(yi|xi,θ)
/bracehtipupleft /bracehtipdownright/bracehtipdownleft /bracehtipupright
log-likelihood.( 1.58)
We will denote the negative log-likelihood byεnll(θ;Dn).
The MLE is often used in practice due to its desirable asymptotic prop-
erties as the sample size nincreases. We give a brief summary here
and provide additional background and deﬁnitions in Appendix A. 3.
To give any guarantees on the convergence of the MLE, we neces-
sarily need to assume that θωis identiﬁable.23If additionally, εnllis23That is, θω↔=θ=↗fω↔=fθfor any
θ↓Θ. In words, there is no other pa-
rameter θthat yields the same function
fθasθω.“well-behaved” then standard results say that the MLE is consistent
andasymptotically normal (Van der Vaart, 2000 ):
ˆθMLEP↘θωand ˆθMLED↘N(θω,Sn)asn↘∞.( 1.59)
Here, we denote by Snthe asymptotic variance of the MLE which
can be understood as measuring the “quality” of the estimate.24This24A “smaller” variance means that we
can be more conﬁdent that the MLE is
close to the true parameter.implies in some sense that the MLE is asymptotically unbiased. More-
over, the MLE can be shown to be asymptotically efﬁcient which is to
say that there exists no other consistent estimator with a “smaller”
asymptotic variance.25 25see Appendix A. 3.4.
The situation is quite different in the ﬁnite sample regime. Here, the
MLE need not be unbiased, and it is susceptible to overﬁtting to the
(ﬁnite) training data as we discuss in more detail in Appendix A. 3.5.
1.3.2Using Priors: Maximum a Posteriori Estimation
We can incorporate prior assumptions about the parameters θωinto the
estimation procedure. One approach of this kind is to ﬁnd the mode
of the posterior distribution, called the maximum a posteriori estimate (or
MAP estimate):
ˆθMAP.=arg max
θ↓Θp(θ|x1:n,y1:n) (1.60)
=arg max
θ↓Θp(y1:n|x1:n,θ)·p(θ) by Bayes’ rule ( 1.45) (1.61)
=arg max
θ↓Θlogp(θ)+n
∑
i=1logp(yi|xi,θ) taking the logarithm (1.62)
=arg min
θ↓Θ→logp(θ)/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
regularization+εnll(θ;Dn)/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
quality of ﬁt.( 1.63)
Here, the log-prior logp(θ)acts as a regularizer. Common regularizers
are given, for example, by






















































itit
mode
interpretation


## Page 35

fundamentals of inference 25
•p(θ)=N(θ;0,(2λ)→1I)which yields →logp(θ)= λ≃θ≃2
2+const,
•p(θ)= Laplace (θ;0,λ→1)which yields →logp(θ)= λ≃θ≃1+const,
•a uniform prior (cf. Section 1.2.1) for which the MAP is equivalent
to the MLE. In other words, the MLE is merely the mode of the
posterior distribution under a uniform prior.
The Gaussian and Laplace regularizers act as simplicity biases, pre-
ferring simpler models over more complex ones, which empirically
tends to reduce the risk of overﬁtting. However, one may also encode
more nuanced information about the (assumed) structure of θωinto
the prior.
An alternative way of encoding a prior is by restricting the function
class to some /tildewideΘ⇐Θ, for example to rotation- and translation-invariant
models as often done when the inputs are images. This effectively
sets p(θ)= 0 for all θ↓Θ\/tildewideΘbut is better suited for numerical
optimization than to impose this constraint directly on the prior.
Encoding prior assumptions into the function class or into the parame-
ter estimation can accelerate learning and improve generalization per-
formance dramatically, yet importantly, incorporating a prior can also
inhibit learning in case the prior is “wrong”. For example, when the
learning task is to differentiate images of cats from images of dogs,
consider the (stupid) prior that only permits models that exclusively
use the upper-left pixel for prediction. No such model will be able to
solve the task, and therefore starting from this prior makes the learn-
ing problem effectively unsolvable which illustrates that priors have to
be chosen with care.
1.3.3When does the prior matter?
We have seen that the MLE has desirable asymptotic properties, and
that MAP estimation can be seen as a regularized MLE where the type
of regularization is encoded by the prior. Is it possible to derive similar
asymptotic results for the MAP estimate?
To answer this question, we will look at the asymptotic effect of the
prior on the posterior more generally. Doob’s consistency theorem states
that assuming parameters are identiﬁable,26there exists /tildewideΘ⇒Θwith26This is akin to the assumption required
for consistency of the MLE, cf. Equa-
tion ( 1.59).p(/tildewideΘ)= 1 such that the posterior is consistent for any θω↓/tildewideΘ(Doob,
1949 ; Miller, 2016 ,2018 ):
θ|DnP↘θωasn↘∞.( 1.64)
In words, Doob’s consistency theorem tells us that for anyprior dis-
tribution, the posterior is guaranteed to converge to a point density
in the (small) neighborhood θω↓Bof the true parameter as long as
p(B)>0.27We call such a prior a well-speciﬁed prior .27Bcan for example be a ball of radius ε
around θω(with respect to some geome-
try of Θ).






















































11this


## Page 36

26 probabilistic artificial intelligence
Remark 1.27: Cromwell’s rule
In the case where |Θ|is ﬁnite, Doob’s consistency theorem strongly
suggests that the prior should not assign 0 probability (or proba-
bility 1 for that matter) to any individual parameter θ↓Θ, unless
we know with certainty that θω↔=θ. This is called Cromwell’s rule ,
and a prior obeying by this rule is always well-speciﬁed.
Under the same assumption that the prior is well-speciﬁed (and reg-
ularity conditions28), the Bernstein-von Mises theorem , which was ﬁrst28These regularity conditions are akin
to the assumptions required for asymp-
totic normality of the MLE, cf. Equa-
tion ( 1.59).discovered by Pierre-Simon Laplace in the early 19th century, estab-
lishes the asymptotic normality of the posterior distribution (Van der
Vaart, 2000 ; Miller, 2016 ):
θ|DnD↘N(θω,Sn) asn↘∞ (1.65)
and where Snis the same as the asymptotic variance of the MLE.29 29This has also been called the “Bayesian
central limit theorem”, which is a bit of a
misnomer since the theorem also applies
to likelihoods (when the prior is nonin-
formative) which are often used in fre-
quentist statistics.These results link probabilistic inference to maximum likelihood es-
timation in the asymptotic limit of inﬁnite data. Intuitively, in the
limit of inﬁnite data, the prior is “overwhelmed” by the observations
and the posterior becomes equivalent to the limiting distribution of
the MLE.30One can interpret the regime of inﬁnite data as the regime30More examples and discussion can be
found in section 17.8of Le Cam ( 1986 ),
chapter 8of Le Cam and Yang ( 2000 ),
chapter 10of Van der Vaart ( 2000 ), and
in Tanner ( 1991 ).where computational resources and time are unlimited and plausible
inferences evolve into logical inferences. This transition signiﬁes a shift
from the realm of uncertainty to that of certainty. The importance of
the prior surfaces precisely in the non-asymptotic regime where plau-
sible inferences are necessary due to limited computational resources and
limited time .
1.3.4Estimation vs Inference
You can interpret a single parameter vector θ↓Θas “one possible
explanation” of the data. Maximum likelihood and maximum a pos-
teriori estimation are examples of estimation algorithms which return
a one such parameter vector — called a point estimate . That is, given
the training set Dn, they return a single parameter vector ˆθn. We give
a more detailed account of estimation in Appendix A. 3.
Example 1.28: Point estimates and invalid logical inferences
To see why point estimates can be problematic, recall Example 1.20.
We have seen that the logical implication
“If it is raining, the ground is wet.”
can be expressed as P(W|R)=1. Observing that “The ground






















































thisdoesn't solve
alotofsense
intuition


## Page 37

fundamentals of inference 27
is wet.” does not permit logical inference, yet, the maximum like-
lihood estimate of RisˆRMLE =1. This is logically inconsistent
since there might be other explanations for the ground to be wet,
such as a sprinkler! With only a ﬁnite sample (say independently
observing ntimes that the ground is wet), we cannot rule out with
certainty that the ground is wet for other reasons than rain.
In practice, we never observe an inﬁnite amount of data. Example 1.28
demonstrates that on a ﬁnite sample, point estimates may perform
invalid logical inferences, and can therefore lure us into a false sense
of certainty.
Remark 1.29: MLE and MAP are approximations of inference
The MLE and MAP estimate can be seen as a naive approximation
of probabilistic inference, represented by a point density which
“collapses” all probability mass at the mode of the posterior dis-
tribution. This can be a relatively decent — even if overly simple
— approximation when the distribution is unimodal, symmetric,
and light-tailed as in Figure 1.8, but is usually a very poor approx-
imation for practical posteriors that are complex and multimodal.→2 0 2
ϱ0.00.10.20.30.4p(ϱ|D)
Figure 1.8: A the MLE/MAP are point
estimates at the mode ˆϱof the posterior
distribution p(ϱ|D).In this manuscript, we will focus mainly on algorithms for probabilistic
inference which compute or approximate the distribution p(θ|x1:n,y1:n)
over parameters. Returning a distribution over parameters is natural
since this acknowledges that given a ﬁnite sample with noisy observa-
tions, more than one parameter vector can explain the data.
1.3.5Probabilistic Inference and Prediction
The prior distribution p(θ)can be interpreted as the degree of our
belief that the model parameterized by θ“describes the (previously
seen) data best”. The likelihood captures how likely the training data
is under a particular model:
p(y1:n|x1:n,θ)=n
∏
i=1p(yi|xi,θ).( 1.66)
The posterior then represents our belief about the best model after
seeing the training data. Using Bayes’ rule ( 1.45), we can write it as31 31We generally assume that
p(θ|x1:n)= p(θ).
For our purposes, you can think of the
inputs x1:nas ﬁxed deterministic param-
eters, but one can also consider inputs
drawn from a distribution over X.p(θ|x1:n,y1:n)=1
Zp(θ)n
∏
i=1p(yi|xi,θ)where ( 1.67a)
Z.=/integraldisplay
Θp(θ)n
∏
i=1p(yi|xi,θ)dθ (1.67b)






















































hencebayesian
inference


## Page 38

28 probabilistic artificial intelligence
is the normalizing constant . We refer to this process of learning a model
from data as learning . We can then use our learned model for prediction
at a new input xωby conditioning on θ,
p(yω|xω,x1:n,y1:n)=/integraldisplay
Θp(yω,θ|xω,x1:n,y1:n)dθ by the sum rule ( 1.7)
=/integraldisplay
Θp(yω|xω,θ)·p(θ|x1:n,y1:n)dθ. by the product rule ( 1.11) and
yω⇑x1:n,y1:n|θ(1.68)
Here, the distribution over models p(θ|x1:n,y1:n)is called the posterior
and the distribution over predictions p(yω|xω,x1:n,y1:n)is called the
predictive posterior . The predictive posterior quantiﬁes our posterior
uncertainty about the “prediction” yω, however, since this is typically
a complex distribution, it is difﬁcult to communicate this uncertainty
to a human. One statistic that can be used for this purpose is the
smallest set Cδ(xω)⇒Rfor a ﬁxed δ↓(0, 1)such that
P(yω↓Cδ(xω)|xω,x1:n,y1:n)⇓1→δ.( 1.69)
That is, we believe with “conﬁdence” at least 1 →δthat the true value
ofyωlies in Cδ(xω). Such a set Cδ(xω)is called a credible set .0 5
yω0.000.050.100.150.20p(yω|xω,D)C(xω)Figure 1.9: Example of a 95% credible
set at xωwhere the predictive posterior is
Gaussian with mean µ(xω)and standard
deviation σ(xω). In this case, the gray
area integrates to ⇔0.95 for
C0.05(xω)=[ µ(xω)±1.96σ(xω)].We have seen here that the tasks of learning and prediction are inti-
mately related. Indeed, “prediction” can be seen in many ways as a
natural by-product of “reasoning” (i.e., probabilistic inference), where
we evaluate the likelihood of outcomes given our learned explana-
tions for the world. This intuition can be read off directly from Equa-
tion ( 1.68) where p(yω|xω,θ)corresponds to the likelihood of an out-
come given the explanation θand p(θ|x1:n,y1:n)corresponds to our
inferred belief about the world. We will see many more examples
of this link between probabilistic inference and prediction throughout
this manuscript.
The high-dimensional integrals of Equations ( 1.67b) and ( 1.68) are typ-
ically intractable, and represent the main computational challenge in
probabilistic inference. Throughout the ﬁrst part of this manuscript,
we will describe settings where exact inference is tractable, as well
as modern approximate inference algorithms that can be used when
exact inference is intractable.
1.3.6Recursive Probabilistic Inference and Memory
We have already alluded to the fact that probabilistic inference has a
recursive structure, which lends itself to continual learning and which
often leads to efﬁcient algorithms. Let us denote by
p(t)(θ).=p(θ|x1:t,y1:t) (1.70)


## Page 39

fundamentals of inference 29
the posterior after the ﬁrst tobservations with p(0)(θ)= p(θ). Now,
suppose that we have already computed p(t)(θ)and observe yt+1.W e
can recursively update the posterior as follows,
p(t+1)(θ)= p(θ|y1:t+1)
∝p(θ|y1:t)·p(yt+1|θ,y1:t) using Bayes’ rule ( 1.45)
=p(t)(θ)·p(yt+1|θ). using yt+1⇑y1:t|θ, see Figure 2.3 (1.71)
Intuitively, the posterior distribution at time t“absorbs” or “summa-
rizes” all seen data.
By unrolling the recursion of Equation ( 1.71), we see that regardless
of the philosophical interpretation of probability, probabilistic infer-
ence is a fundamental mechanism of learning. Even the MLE which
performs naive approximate inference without a prior (i.e., a uniform
prior), is based on p(n)(θ)∝p(y1:n|x1:n,θ)which is the result of n
individual plausible inferences, where the (t+1)-st inference uses the
posterior of the t-th inference as its prior.
So far we have been considering the supervised learning setting, where
all data is available a-priori. However, by sequentially obtaining the
new posterior and replacing our prior, we can also perform proba-
bilistic inference as data arrives online (i.e., in “real-time”). This is
analogous to recursive logical inference where derived consequences
are repeatedly added to the set of propositions to derive new conse-
quences. This also highlights the intimate connection between “rea-
soning” and “memory”. Indeed, the posterior distribution p(t)(θ)can
be seen as a form of memory that evolves with time t.
1.4Outlook: Decision Theory
How can we use our predictions to make concrete decisions under
uncertainty? We will study this question extensively in Part II of this
manuscript, but brieﬂy introduce some fundamental concepts here.
Making decisions using a probabilistic model p(y|x)of output y↓Y
given input x↓X, such as the ones we have discussed in the previous
section, is commonly formalized by
•a set of possible actions A, and
•a reward function r(y,a)↓Rthat computes the reward or utility of
taking action a↓A, assuming the true output is y↓Y.
Standard decision theory recommends picking the action with the
largest expected utility:
aω(x).=arg max
a↓AEy|x[r(y,a)].( 1.72)






















































in
KEETE
t.EE


## Page 40

30 probabilistic artificial intelligence
Here, aωis called the optimal decision rule because, under the given
probabilistic model, no other rule can yield a higher expected utility.
Let us consider some examples of reward functions and their corre-
sponding optimal decisions:
Example 1.30: Reward functions
Under the decision rule from Equation ( 1.72), different reward
functions rcan lead to different decisions. Let us examine two
reward functions for the case where Y=A=R? Problem 1.13 .
•Alternatively to considering ras a reward function, we can in-
terpret →ras the loss of taking action awhen the true output
isy. If our goal is for our actions ato “mimic” the output y,a
natural choice is the squared loss, →r(y,a)=( y→a)2. It turns
out that under the squared loss, the optimal decision is simply
the mean: aω(x)= E[y|x].
•To contrast this, we consider the asymmetric loss,
→r(y,a)= c1max{y→a,0}/bracehtipupleft/bracehtipdownright/bracehtipdownleft /bracehtipupright
underestimation error+ c2max{a→y,0}/bracehtipupleft/bracehtipdownright/bracehtipdownleft /bracehtipupright
overestimation error,
which penalizes underestimation and overestimation differently.
When y|x↑N(µx,σ2
x)then the optimal decision is
aω(x)= µx+σx·Φ→1/parenleftbiggc1
c1+c2/parenrightbigg
/bracehtipupleft /bracehtipdownright/bracehtipdownleft /bracehtipupright
pessimism / optimism
where Φis the CDF of the standard normal distribution.32 32Recall that the CDF Φof the standard
normal distribution is a sigmoid with its
inverse satisfying
Φ→1(u)

<0 if u<0.5,
=0 if u=0.5,
>0 if u>0.5.Note
that if c1=c2, then the second term vanishes and the optimal
decision is the same as under the squared loss. If c1>c2, the
second term is positive (i.e., optimistic ) to avoid underestima-
tion, and if c1<c2, the second term is negative (i.e., pessimistic )
to avoid overestimation. We will ﬁnd these notions of optimism
and pessimism to be useful in many decision-making scenarios.
While Equation ( 1.72) describes how to make optimal decisions given
a (posterior) probabilistic model, it does not tell us how to learn or
improve this model in the ﬁrst place. That is, these decisions are only
optimal under the assumption that we cannot use their outcomes and
our resulting observations to update our model and inform future de-
cisions. When we start to consider the effect of our decisions on future
data and future posteriors, answering “how do I make optimal deci-
sions?” becomes more complex, and we will study this in Part II on
sequential decision-making.






















































fthf.in
V
derivations


## Page 41

fundamentals of inference 31
Discussion
In this chapter, we have learned about the fundamental concepts of
probabilistic inference. We have seen that probabilistic inference is the
natural extension of logical reasoning to domains with uncertainty. We
have also derived the central principle of probabilistic inference, Bayes’
rule, which is simple to state but often computationally challenging. In
the next part of this manuscript, we will explore settings where exact
inference is tractable, as well as modern approaches to approximate
probabilistic inference.
Overview of Mathematical Background
We have included brief summaries of the fundamentals of parame-
ter estimation (mean estimation in particular) and optimization in
Appendices A. 3and A. 4, respectively, which we will refer back to
throughout the manuscript. Appendix A. 2discusses the correspon-
dence of Gaussians and quadratic forms . Appendix A. 5comprises a
list of useful matrix identities and inequalities.
Problems
1.1.Properties of probability.
Let(Ω,A,P)be a probability space. Derive the following properties
of probability from the Kolmogorov axioms:
1.For any A,B↓A, ifA⇒Bthen P(A)↖P(B).
2.For any A↓A,P/parenleftbig
A/parenrightbig=1→P(A).
3.For any countable set of events {Ai↓A } i,
P/parenleftigg∞
i=1Ai
↖∞
∑
i=1P(Ai).( 1.73)
which is called a union bound .
1.2.Random walks on graphs.
LetGbe a simple connected ﬁnite graph. We start at a vertex uofG.
At every step, we move to one of the neighbors of the current vertex
uniformly at random, e.g., if the vertex has 3 neighbors, we move to
one of them, each with probability 1/3. What is the probability that
the walk visits a given vertex veventually?


## Page 42

32 probabilistic artificial intelligence
1.3.Law of total expectation.
Show that if {Ai}k
i=1are a partition of ΩandXis a random vector,
E[X]=k
∑
i=1E[X|Ai]·P(Ai).( 1.74)
1.4.Covariance matrices are positive semi-deﬁnite.
Prove that a covariance matrix Σis always positive semi-deﬁnite. That
is, all of its eigenvalues are greater or equal to zero, or equivalently,
x↙Σx⇓0 for any x↓Rn.
1.5.Probabilistic inference.
As a result of a medical screening, one of the tests revealed a serious
disease in a person. The test has a high accuracy of 99% (the prob-
ability of a positive response in the presence of a disease is 99% and
the probability of a negative response in the absence of a disease is
also 99%). However, the disease is quite rare and occurs only in one
person per 10 000. Calculate the probability of the examined person
having the identiﬁed disease.
1.6.Zero eigenvalues of covariance matrices.
We say that a random vector XinRnis not linearly independent if for
some ε↓Rn\{0},ε↙X=0.
1.Show that if Xis not linearly independent, then Var [X]has a zero
eigenvalue.
2.Show that if Var [X]has a zero eigenvalue, then Xis not linearly
independent.
Hint: Consider the variance of ϱ↙Xwhere ϱis the eigenvector corre-
sponding to the zero eigenvalue.
Thus, we have shown that Var [X]has a zero eigenvalue if and only if X
is not linearly independent.
1.7.Product of Gaussian PDFs.
Letµ1,µ2↓Rnbe mean vectors and Σ1,Σ2↓Rn∝nbe covariance
matrices. Prove that
N(x;µ,Σ)∝N(x;µ1,Σ1)·N(x;µ2,Σ2) (1.75)
for some mean vector µ↓Rnand covariance matrix Σ↓Rn∝n. That
is, show that the product of two Gaussian PDFs is proportional to the
PDF of a Gaussian.


## Page 43

fundamentals of inference 33
1.8.Independence of Gaussians.
Show that two jointly Gaussian random vectors, XandY, are indepen-
dent if and only if XandYare uncorrelated.
1.9.Marginal / conditional distribution of a Gaussian.
Prove Theorem 1.25. That is, show that
1.every marginal of a Gaussian is Gaussian; and
2.conditioning on a subset of variables of a joint Gaussian is Gaus-
sian
by ﬁnding their corresponding PDFs.
Hint: You may use that for matrices ΣandΛsuch that Σ→1=Λ,
•ifΣandΛare symmetric,

xA
xB↙
ΛAA ΛAB
ΛBA ΛBB
xA
xB
=x↙
AΛAAxA+x↙
AΛABxB+x↙
BΛBAxA+x↙
BΛBBxB
=x↙
A(ΛAA→ΛABΛ→1
BBΛBA)xA+
(xB+Λ→1
BBΛBAxA)↙ΛBB(xB+Λ→1
BBΛBAxA),
•Λ→1
BB=ΣBB→ΣBAΣ→1
AAΣAB,
•Λ→1
BBΛBA=→ΣBAΣ→1
AA.
The ﬁnal two equations follow from the general characterization of the inverse
of a block matrix (Petersen et al., 2008 , section 9.1.3).
1.10.Closedness properties of Gaussians.
Recall the notion of a moment-generating function (MGF) of a random
vector XinRnwhich is deﬁned as
ϕX(t).=Eexp
t↙X
, for all t↓Rn.( 1.76)
An MGF uniquely characterizes a distribution. The MGF of the multi-
variate Gaussian X↑N(µ,Σ)is
ϕX(t)= exp/parenleftbigg
t↙µ+1
2t↙Σt/parenrightbigg
.( 1.77)
This generalizes the MGF of the univariate Gaussian from Equation (A. 40).
Prove the following facts.
1.Closedness under afﬁne transformations: Given an n-dimensional Gaus-
sian X↑N(µ,Σ), and A↓Rm∝nandb↓Rm,
AX+b↑N(Aµ+b,AΣA↙).( 1.78)


## Page 44

34 probabilistic artificial intelligence
2.Additivity: Given two independent Gaussian random vectors X↑N(µ,Σ)
andX′↑N(µ′,Σ′)inRn,
X+X′↑N(µ+µ′,Σ+Σ′).( 1.79)
These properties are unique to Gaussians and a reason for why they
are widely used for learning and inference.
1.11.Expectation and variance of Gaussians.
Derive that E[X]=µand Var [X]=Σwhen X↑N(µ,Σ).
Hint: First derive the expectation and variance of a univariate standard nor-
mal random variable.
1.12.Non-afﬁne transformations of Gaussians.
Answer the following questions with yesorno.
1.Does there exist any non-afﬁne transformation of a Gaussian ran-
dom vector which is Gaussian? If yes, give an example.
2.LetX,Y,Zbe independent standard normal random variables. Is
X+YZ∞
1+Z2Gaussian?
1.13.Decision theory.
Derive the optimal decisions under the squared loss and the asymmet-
ric loss from Example 1.30.


## Page 45

part I
Probabilistic Machine Learning


## Page 46




## Page 47

Preface to Part I
As humans, we constantly learn about the world around us. We learn
to interact with our physical surroundings. We deepen our under-
standing of the world by establishing relationships between actors, ob-
jects, and events. And we learn about ourselves by observing how
we interact with the world and with ourselves. We then continuously
use this knowledge to make inferences and predictions, be it about the
weather, the movement of a ball, or the behavior of a friend.
With limited computational resources, limited genetic information, and
limited life experience, we are not able to learn everything about the
world to complete certainty. We saw in Chapter 1that probability the-
ory is the mathematical framework for reasoning with uncertainty in
the same way that logic is the mathematical framework for reasoning
with certainty. We will discuss two kinds of uncertainty: “aleatoric”
uncertainty which cannot be reduced under computational constraints,
and “epistemic” uncertainty which can be reduced by observing more
data.
An important aspect of learning is that we do not just learn once, but
continually. Bayes’ rule allows us to update our beliefs and reduce
our uncertainty as we observe new data — a process that is called
probabilistic inference . By taking the former posterior as the new prior,
probabilistic inference can be performed continuously and repeated
indeﬁnitely as we observe more and more data.
world
perception
prior p(θ)model p(θ|D)
D
Figure 1.10: A schematic illustration of
probabilistic inference in the context of
the (supervised) learning of a model θ
from perceived data D. The prior model
p(θ)can equip the model with anything
from substantial, to little, to no prior
knowledge.Our sensory information is often noisy and imperfect, which is another
source of uncertainty. The same is true for machines, even if they can
sometimes sense aspects of the world more accurately than humans.
We discuss how one can infer latent structure of the world from sensed
data, such as the state of a dynamical system like a car, in a process
that is called ﬁltering .
In this ﬁrst part of the course, we examine how we can build machines
that are capable of (continual) learning and inference. First, we in-
troduce probabilistic inference in the context of linear models which


## Page 48

38 probabilistic artificial intelligence
make predictions based on ﬁxed (often hand-designed) features. We
then discuss how probabilistic inference can be scaled to kernel meth-
ods and Gaussian processes which use a large (potentially inﬁnite)
number of features, and to deep neural networks which learn features
dynamically from data. In these models, exact inference is typically in-
tractable, and we discuss modern methods for approximate inference
such as variational inference and Markov chain Monte Carlo. We high-
light a tradeoff between curiosity (i.e., extrapolating beyond the given
data) and conformity (i.e., ﬁtting the given data), which surfaces as a
fundamental principle of probabilistic inference in the regime where
the data and our computational resources are limited.


## Page 49

2
Linear Regression
As a ﬁrst example of probabilistic inference, we will study linear mod-
els for regression1which assume that the output y↓Ris a linear1As we have discussed in Section 1.3,
regression models can also be used for
classiﬁcation. The canonical example of
a linear model for classiﬁcation is logis-
tic regression, which we will discuss in
Section 5.1.1.function of the input x↓Rd:
y⇔w↙x+w0
where w↓Rdare the weights and w0↓Ris the intercept. Observe
that if we deﬁne the extended inputs x′.=(x,1)and w′.=(w,w0),
then w′↙x′=w↙x+w0, implying that without loss of generality it
sufﬁces to study linear functions without the intercept term w0.W e
will therefore consider the following function class of linear models
f(x;w).=w↙x.→202x→202y
Figure 2.1: Example of linear regression
with the least squares estimator (shown
in blue).We will consider the supervised learning task of learning weights w
from labeled training data {(xi,yi)}n
i=1. We deﬁne the design matrix ,
X.=
x↙
1
...
x↙
n
↓Rn∝d,( 2.1)
as the collection of inputs and the vector y.=[y1···yn]↙↓Rnas the
collection of labels. For each noisy observation (xi,yi), we deﬁne the
value of the approximation of our model, fi.=w↙xi. Our model at
the inputs Xis described by the vector f.=[f1···fn]↙which can be
expressed succinctly as f=Xw.
The most common way of estimating wfrom data is the least squares
estimator ,
ˆwls.=arg min
w↓Rdn
∑
i=1(yi→w↙xi)2=arg min
w↓Rd≃y→Xw≃2
2,( 2.2)


## Page 50

40 probabilistic artificial intelligence
minimizing the squared difference between the labels and predictions
of the model. A slightly different estimator is used for ridge regression ,
ˆwridge.=arg min
w↓Rd≃y→Xw≃2
2+λ≃w≃2
2(2.3)
where λ>0. The squared L2regularization term λ≃w≃2
2penalizes
large wand thus reduces the “complexity” of the resulting model.22Ridge regression is more robust to mul-
ticollinearity than standard linear re-
gression. Multicollinearity occurs when
multiple independent inputs are highly
correlated. In this case, their individual
effects on the predicted variable cannot
be estimated well. Classical linear re-
gression is highly volatile to small input
changes. The regularization of ridge re-
gression reduces this volatility by intro-
ducing a bias on the weights towards 0.It can be shown that the unique solutions to least squares and ridge
regression are given by
ˆwls=(X↙X)→1X↙y and ( 2.4)
ˆwridge=(X↙X+λI)→1X↙y,( 2.5)
respectively if the Hessian of the loss is positive deﬁnite (i.e., the loss
is strictly convex) ? Problem 2.1(1) which is the case as long as the columns of X
are not linearly dependent. Least squares regression can be seen as
ﬁnding the orthogonal projection of yonto the column space of X, as
is illustrated in Figure 2.2? Problem 2.1(2) .yXˆwls
span{X}Figure 2.2: Least squares regression
ﬁnds the orthogonal projection of yonto
span{X}(here illustrated as the plane).2.0.1Maximum Likelihood Estimation
Since our function class comprises linear functions of the form w↙x,
the observation model from Equation ( 1.56) simpliﬁes to
yi=wω↙xi+εi (2.6)
for some weight vector w, where for the purpose of this chapter we
will additionally assume that εi↑N(0,σ2
n)is homoscedastic Gaussian
noise.3This observation model is equivalently characterized by the
3εiis called additive white Gaussian noise . Gaussian likelihood,
yi|xi,w↑N(w↙xi,σ2
n). using Equation ( 1.55) (2.7)
Based on this likelihood we can compute the MLE ( 1.57) of the weights:
ˆwMLE=arg max
w↓Rdn
∑
i=1logp(yi|xi,w)= arg min
w↓Rdn
∑
i=1(yi→w↙xi)2. plugging in the Gaussian likelihood and
simplifying
Note that therefore ˆ wMLE=ˆwls.
In practice, the noise variance σ2
nis typically unknown and also has to
be determined, for example, through maximum likelihood estimation.
It is a straightforward exercise to check that the MLE of σ2
ngiven ﬁxed
weights wisˆσ2
n=1
n∑n
i=1(yi→w↙xi)2? Problem 2.2 .
2.1Weight-space View
The most immediate and natural probabilistic interpretation of lin-
ear regression is to quantify uncertainty about the weights w. Recall






















































canbederived viaaprior
alimeEli
i
meaning


## Page 51

linear regression 41
that probabilistic inference requires speciﬁcation of a generative model
comprised of prior and likelihood. Throughout this chapter, we will
use the Gaussian prior,
w↑N(0,σ2
pI),( 2.8)
and the Gaussian likelihood from Equation ( 2.7). We will discuss pos-
sible (probabilistic) strategies for choosing hyperparameters such as
the prior variance σ2
pand the noise variance σ2
nin Section 4.4.
w
σ2
n
yi
σ2
p
xi
i↓1:n
Figure 2.3: Directed graphical model of
Bayesian linear regression in plate nota-
tion.Remark 2.1: Why a Gaussian prior?
The choice of using a Gaussian prior may seem somewhat arbi-
trary at ﬁrst sight, except perhaps for the nice analytical proper-
ties of Gaussians that we have seen in Section 1.2.3and which will
prove useful. The maximum entropy principle (cf. Section 1.2.1)
provides a more fundamental justiﬁcation for Gaussian priors since
turns out that Nhas the maximum entropy among all distribu-
tions on Rdwith known mean and variance ? Problem 5.6 .
Next, let us derive the posterior distribution over the weights.
logp(w|x1:n,y1:n)
=logp(w)+logp(y1:n|x1:n,w)+const by Bayes’ rule ( 1.45)
=logp(w)+n
∑
i=1logp(yi|xi,w)+const using independence of the samples
=→1
2
σ→2
p≃w≃2
2+σ→2
nn
∑
i=1(yi→w↙xi)2
+const using the Gaussian prior and likelihood
=→1
2
σ→2
p≃w≃2
2+σ→2
n≃y→Xw≃2
2
+const using ∑n
i=1(yi→w↙xi)2=≃y→Xw≃2
2
=→1
2
σ→2
pw↙w+σ→2
n
w↙X↙Xw→2y↙Xw+y↙y
+const
=→1
2
w↙(σ→2
nX↙X+σ→2
pI)w→2σ→2
ny↙Xw
+const. ( 2.9)
Observe that the log-posterior is a quadratic form in w, so the posterior
distribution must be Gaussian:
w|x1:n,y1:n↑N(µ,Σ) see Equation (A. 12) (2.10a)
where we can read off the mean and variance to be
µ.=σ→2
nΣX↙y,( 2.10b)
Σ.=
σ→2
nX↙X+σ→2
pI→1
.( 2.10c)
This also shows that Gaussians with known variance and linear like-
lihood are self-conjugate, a property that we had hinted at in Sec-
tion1.2.2. It can be shown more generally that Gaussians with known






















































meaning
howisthislinkedto
Fifi.net


## Page 52

42 probabilistic artificial intelligence
variance are self-conjugate to any Gaussian likelihood (Murphy, 2007 ).
For other generative models, the posterior can typically not be ex-
pressed in closed-form — this is a very special property of Gaussians!
2.1.1Maximum a Posteriori Estimation
Computing the MAP estimate for the weights,
ˆwMAP=arg max
wlogp(y1:n|x1:n,w)+logp(w)
=arg min
w≃y→Xw≃2
2+σ2
n
σ2p≃w≃2
2, using that the likelihood and prior are
Gaussian(2.11)
we observe that this is identical to ridge regression with weight decay
λ.=σ2
n/σ2
p:ˆwMAP =ˆwridge. Equation ( 2.11) is simply the MLE loss
with an additional L2-regularization (originating from the prior) that
encourages keeping weights small. Recall that the MAP estimate cor-
responds to the mode of the posterior distribution, which in the case
of a Gaussian is simply its mean µ. As to be expected, µcoincides with
the analytical solution to ridge regression from Equation ( 2.5).→101ϱ1→1.0→0.50.00.51.0ϱ2
Figure 2.4: Level sets of L2-(blue)
andL1-regularization ( red), correspond-
ing to Gaussian and Laplace priors, re-
spectively. It can be seen that L1-
regularization is more effective in en-
couraging sparse solutions (that is, so-
lutions where many components are set
to exactly 0).Example 2.2: Lasso as the MAP estimate with a Laplace prior
One problem with ridge regression is that the contribution of
nearly-zero weights to the L2-regularization term is negligible.
Thus, L2-regularization is typically not sufﬁcient to perform vari-
able selection (that is, set some weights to zero entirely), which is
often desirable for interpretability of the model.
A commonly used alternative to ridge regression is the least ab-
solute shrinkage and selection operator (orlasso), which regularizes
with the L1-norm:
ˆwlasso.=arg min
w↓Rd≃y→Xw≃2
2+λ≃w≃1.( 2.12)
It turns out that lasso can also be viewed as probabilistic infer-
ence, using a Laplace prior w↑Laplace (0,h)with length scale h
instead of a Gaussian prior.
Computing the MAP estimate for the weights yields,
ˆwMAP=arg max
wlogp(y1:n|x1:n,w)+logp(w)
=arg min
wn
∑
i=1(yi→w↙xi)2+σ2
n
h≃w≃1using that the likelihood is Gaussian
and the prior is Laplacian(2.13)
which coincides with the lasso with weight decay λ.=σ2
n/h.






















































derivation
dittyHeel
featureselention
Ifdostand
forsomething


## Page 53

linear regression 43
To make predictions at a test point xω, we deﬁne the (model-)predicted
point fω.=ˆw↙
MAPxωand obtain the label prediction
yω|xω,x1:n,y1:n↑N(fω,σ2
n).( 2.14)
Here we observe that using point estimates such as the MAP estimate
does not quantify uncertainty in the weights. The MAP estimate sim-
ply collapses all mass of the posterior around its mode. This can be
harmful when we are highly unsure about the best model, e.g., because
we have observed insufﬁcient data.
2.1.2Probabilistic Inference
Rather than selecting a single weight vector ˆ wto make predictions, we
can use the full posterior distribution. This is known as Bayesian linear
regression (BLR) and illustrated with an example in Figure 2.5.
→0.5 0.0 0.5 1.0 1.5
x→10123y
→0.50.0 0.5 1.0 1.5
x→4→20246yFigure 2.5: Comparison of linear regres-
sion (MLE) ,ridge regression (MAP es-
timate) , and Bayesian linear regression
when the data is generated according to
y|w,x↑N(w↙x,σ2
n).
The true mean is shown in black, the
MLE in blue, and the MAP estimate in
red. The dark gray area denotes the epis-
temic uncertainty of Bayesian linear re-
gression and the light gray area the addi-
tional homoscedastic noise. On the left,
σn=0.15. On the right, σn=0.7.
To make predictions at a test point xω, we let fω.=w↙xωwhich has
the distribution
fω|xω,x1:n,y1:n↑N(µ↙xω,xω↙Σxω). using the closedness of Gaussians
under linear transformations ( 1.78)(2.15)
Note that this does not take into account the noise in the labels σ2
n. For
the label prediction yω, we obtain
yω|xω,x1:n,y1:n↑N(µ↙xω,xω↙Σxω+σ2
n). using additivity of Gaussians ( 1.79) (2.16)
2.1.3Recursive Probabilistic Inference
We have already discussed the recursive properties of probabilistic in-
ference in Section 1.3.6. For Bayesian linear regression with a Gaussian
prior and likelihood, this principle can be used to derive an efﬁcient
online algorithm since also the posterior is a Gaussian,
p(t)(w)=N(w;µ(t),Σ(t)),( 2.17)
which can be stored efﬁciently using only O/parenleftbig
d2/parenrightbig
parameters. This
leads to an efﬁcient online algorithm for Bayesian linear regression


## Page 54

44 probabilistic artificial intelligence
with time-independent(!) memory complexity O(d)and round com-
plexity O/parenleftbig
d2/parenrightbig
? Problem 2.5 . The interpretation of Bayesian linear regression as an
online algorithm also highlights similarities to other sequential models
such as Kalman ﬁlters, which we discuss in Chapter 3. In Example 3.5,
we will learn that online Bayesian linear regression is, in fact, an ex-
ample of a Kalman ﬁlter.
2.2Aleatoric and Epistemic Uncertainty
The predictive posterior distribution from Equation ( 2.16) highlights a
decomposition of uncertainty wherein xω↙Σxωcorresponds to the un-
certainty about our model due to the lack of data (commonly referred
to as the epistemic uncertainty ) and σ2
ncorresponds to the uncertainty
about the labels that cannot be explained by the inputs and any model
from the model class (commonly referred to as the aleatoric uncertainty ,
“irreducible noise”, or simply “(label) noise”) ? Problem 2.6 .
A natural probabilistic approach is to represent epistemic uncertainty
with a probability distribution over models. Intuitively, the variance
of this distribution measures our uncertainty about the model and its
mode corresponds to our current best (point) estimate. The distri-
bution over weights of a linear model is one example, and we will
continue to explore this approach for other models in the following
chapters.
It is a practical modeling choice how much inaccuracy to attribute to
epistemic or aleatoric uncertainty. Generally, when a poor model is
used to explain a process, more inaccuracy has to be attributed to irre-
ducible noise. For example, when a linear model is used to “explain”
a nonlinear process, most uncertainty is aleatoric as the model cannot
explain the data well. As we use more expressive models, a larger
portion of the uncertainty can be explained by the data.
Epistemic and aleatoric uncertainty can be formally deﬁned in terms
of the law of total variance ( 1.41),
Var[yω|xω]=Eθ
Varyω[yω|xω,θ]
/bracehtipupleft /bracehtipdownright/bracehtipdownleft /bracehtipupright
aleatoric uncertainty+Varθ
Eyω[yω|xω,θ]
/bracehtipupleft /bracehtipdownright/bracehtipdownleft /bracehtipupright
epistemic uncertainty.( 2.18)
Here, the mean variability of predictions yωaveraged across all mod-
elsθis the estimated aleatoric uncertainty . In contrast, the variability of
the mean prediction yωunder each model θis the estimated epistemic
uncertainty . This decomposition of uncertainty will appear frequently
throughout this manuscript.






















































Aexerciasleet
uhposition
t.itiiitan
SEE ismodel
familypostchoice


## Page 55

linear regression 45
2.3Non-linear Regression
We can use linear regression not only to learn linear functions. The
trick is to apply a nonlinear transformation φ:Rd↘Reto the fea-
tures xi, where dis the dimension of the input space and eis the
dimension of the designed feature space . We denote the design ma-
trix comprised of transformed features by Φ↓Rn∝e. Note that if the
feature transformation φis the identity function then Φ=X.→2 0 2
x→50510y
Figure 2.6: Applying linear regression
with a feature space of polynomials of
degree 10. The least squares estimate is
shown in blue, ridge regression in red,
andlasso in green.Example 2.3: Polynomial regression
Letφ(x).=[x2,x,1]andw.=[a,b,c]. Then the function that our
model learns is given as
f=ax2+bx+c.
Thus, our model can exactly represent all polynomials up to de-
gree 2.
However, to learn polynomials of degree mindinput dimensions,
we need to apply the nonlinear transformation
φ(x)=[ 1,x1,..., xd,x2
1,..., x2
d,x1·x2,..., xd→1·xd,
...,
xd→m+1····· xd].
Note that the feature dimension eis∑m
i=0(d+i→1
i)=Θ(dm).4 4Observe that the vector contains (d+i→1
i)
monomials of degree ias this is the
number of ways to choose itimes from
ditems with replacement and without
consideration of order. To see this, con-
sider the following encoding: We take
a sequence of d+i→1 spots. Select-
ing any subset of ispots, we interpret
the remaining d→1 spots as “barriers”
separating each of the ditems. The se-
lected spots correspond to the number
of times each item has been selected. For
example, if 2 items are to be selected out
of a total of 4 items with replacement,
one possible conﬁguration is “ ∈| |∈| ”
where ∈denotes a selected spot and |
denotes a barrier. This conﬁguration en-
codes that the ﬁrst and third item have
each been chosen once. The number of
possible conﬁgurations — each encoding
a unique outcome — is therefore (d+i→1
i).Thus,
the dimension of the feature space grows exponentially in the de-
gree of polynomials and input dimensions. Even for relatively
small mandd, this becomes completely unmanageable.
The example of polynomials highlights that it may be inefﬁcient to
keep track of the weights w↓Rewhen eis large, and that it may
be useful to instead consider a reparameterization which is of dimen-
sion nrather than of the feature dimension.
2.4Function-space View
Let us now look at Bayesian linear regression through a different lens.
Previously, we have been interpreting it as a distribution over the
weights wof a linear function f=Φw. The key idea is that for a
ﬁnite set of inputs (ensuring that the design matrix is well-deﬁned),
we can equivalently consider a distribution directly over the estimated
function values f. We call this the function-space view of Bayesian linear
regression.






















































monoids
whatdoelthis nec_


## Page 56

46 probabilistic artificial intelligence
Instead of considering a prior over the weights w↑N(0,σ2
pI)as we
have done previously, we now impose a prior directly on the values of
our model at the observations. Using that Gaussians are closed under
linear maps ( 1.78), we obtain the equivalent prior
f|X↑N(ΦE[w],ΦVar[w]Φ↙)=N(0,σ2
pΦΦ↙
/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
K) (2.19)
where K↓Rn∝nis the so-called kernel matrix . Observe that the entries
of the kernel matrix can be expressed as K(i,j)= σ2
p·φ(xi)↙φ(xj).x1xnx→4→202yf1fn
Figure 2.7: An illustration of the
function-space view. The model is de-
scribed by the points (xi,fi).You may say that nothing has changed, and you would be right —
that is precisely the point. Note, however, that the shape of the kernel
matrix is n∝nrather than the e∝ecovariance matrix over weights,
which becomes unmanageable when eis large. The kernel matrix K
has entries only for the ﬁnite set of observed inputs. However, in
principle, we could have observed any input, and this motivates the
deﬁnition of the kernel function
k(x,x′).=σ2
p·φ(x)↙φ(x′) (2.20)
for arbitrary inputs xandx′. A kernel matrix is simply a ﬁnite “view”
of the kernel function,
K=
k(x1,x1)···k(x1,xn)
.........
k(xn,x1)···k(xn,xn)
(2.21)
Observe that by deﬁnition of the kernel matrix in Equation ( 2.19), the
kernel matrix is a covariance matrix and the kernel function measures
the covariance of the function values f(x)and f(x′)given inputs x
andx′:
k(x,x′)= Cov
f(x),f(x′)
.( 2.22)
Moreover, note that we have reformulated5the learning algorithm5we often say “kernelized”
such that the feature space is now implicit in the choice of kernel, and
the kernel is deﬁned by inner products of (nonlinearly transformed)
inputs. In other words, the choice of kernel implicitly determines the
class of functions that fis sampled from (without expressing the func-
tions explicitly in closed-form), which encodes our prior beliefs. This
is known as the kernel trick .
2.4.1Learning and Predictions
We have already kernelized the Bayesian linear regression prior. The
posterior distribution f|X,yis again Gaussian due to the closedness






















































Kernel trisk
anyfunitions


## Page 57

linear regression 47
properties of Gaussians, analogously to our derivation of the prior
kernel matrix in Equation ( 2.19).
It remains to show that we can also rely on the kernel trick for predic-
tions. Given the test point xω, we deﬁne
˜Φ.=
Φ
φ(xω)↙
,˜y.=
y
yω
,˜f.=
f
fω
.
We immediately obtain ˜f=˜Φw. Analogously to our analysis of pre-
dictions from the weight-space view, we add the label noise to obtain
the estimate ˜ y=˜f+˜εwhere ˜ ε.=[ε1···εnεω]↙↑N(0,σ2
nI)is the
independent label noise. Applying the same reasoning as we did for
the prior, we obtain
˜f|X,xω↑N(0,˜K) (2.23)
where ˜K.=σ2
p˜Φ˜Φ↙. Adding the label noise yields
˜y|X,xω↑N(0,˜K+σ2
nI).( 2.24)
Finally, we can conclude from the closedness of Gaussian random vec-
tors under conditional distributions ( 1.53) that the predictive posterior
yω|xω,X,yfollows again a normal distribution. We will do a full
derivation of the posterior and predictive posterior in Section 4.1.
2.4.2Efﬁcient Polynomial Regression
But how does the kernel trick address our concerns about efﬁciency
raised in Section 2.3? After all, computing the kernel for a feature
space of dimension estill requires computing sums of length ewhich
is prohibitive when eis large. The kernel trick opens up a couple of
new doors for us:
1.For certain feature transformations φ, we may be able to ﬁnd an
easier to compute expression equivalent to φ(x)↙φ(x′).
2.If this is not possible, we could approximate the inner product by
an easier to compute expression.
3.Or, alternatively, we may decide not to care very much about the
exact feature transformation and simply experiment with kernels
that induce some feature space (which may even be inﬁnitely di-
mensional).
We will explore the third approach when we revisit kernels in Sec-
tion 4.3. A polynomial feature transformation can be computed efﬁ-
ciently in closed-form.
Fact 2.4.For the polynomial feature transformation φup to degree m from
Example 2.3, it can be shown that up to constant factors,
φ(x)↙φ(x′)=( 1+x↙x′)m.( 2.25)






















































iii
Howdoesthishelp


## Page 58

48 probabilistic artificial intelligence
For example, for input dimension 2, the kernel (1+x↙x′)2corre-
sponds to the feature vector φ(x)=[ 1∞
2x1∞
2x2∞
2x1x2x2
1x2
2]↙.
Discussion
We have explored a probabilistic perspective on linear models, and
seen that classical approaches such as least squares and ridge regres-
sion can be interpreted as approximate probabilistic inference. We
then saw that we can even perform exact probabilistic inference efﬁ-
ciently if we adopt a Gaussian prior and Gaussian noise assumption.
These are already powerful tools, which are often applied also to non-
linear models if we treat the latent feature space — which was either
human-designed or learned via deep learning — as ﬁxed. In the next
chapter, we will digress brieﬂy from the storyline on “learning” to see
how we can adopt a similar probabilistic perspective to track latent
states over time. Then, in Chapter 4, we will see how we can use the
function-space view and kernel trick to learn ﬂexible nonlinear models
with exact probabilistic inference, without ever explicitly representing
the feature space.
Problems
2.1.Closed-form linear regression.
1.Derive the unique solutions to least squares and ridge regression
from Equations ( 2.4) and ( 2.5).
2.For an n∝mmatrix Aand vector x↓Rm, we call ΠAxthe orthogo-
nal projection of xonto span {A}={Ax′|x′↓Rm}. In particular,
an orthogonal projection satisﬁes x→ΠAx⇑Ax′for all x′↓Rm.
Show that ˆ wlsfrom Equation ( 2.4) is such that Xˆwlsis the unique
closest point to yon span {X}, i.e., it satisﬁes Xˆwls=ΠXy.
2.2.MLE of noise variance.
Show that the MLE of σ2
ngiven ﬁxed weights wis
ˆσ2
n=1
nn
∑
i=1(yi→w↙xi)2.( 2.26)
2.3.Variance of least squares around training data.
Show that the variance of a prediction at the point [1xω]↙is small-
est when xωis the mean of the training data. More formally, show
that if inputs are of the form xi=[1xi]↙where xi↓Rand ˆ wlsis
the least squares estimate, then Var
yω|[1xω]↙,ˆwls
is minimized for
xω=1
n∑n
i=1xi.


## Page 59

linear regression 49
2.4.Bayesian linear regression.
Suppose you are given the following observations
X=
11
12
21
22
,y=
2.4
4.3
3.1
4.9

and assume the data follows a linear model with homoscedastic noise
N(0,σ2
n)where σ2
n=0.1.
1.Find the maximum likelihood estimate ˆ wMLEgiven the data.
2.Now assume that we have a prior p(w)=N(w;0,σ2
pI)with σ2
p=0.05.
Find the MAP estimate ˆ wMAP given the data and the prior.
3.Use the posterior p(w|X,y)to get a posterior prediction for the
label yωatxω=[33]↙. Report the mean and the variance of this
prediction.
4.How would you have to change the prior p(w)such that
ˆwMAP↘ˆwMLE?
2.5.Online Bayesian linear regression.
1.Can you design an algorithm that updates the posterior (as op-
posed to recalculating it from scratch using Equation ( 2.10)) in a
smarter way? The requirement is that the memory should not grow
asO(t).
2.Ifdis large, computing the inverse every round is very expen-
sive. Can you use the recursive structure you found in the previ-
ous question to bring down the computational complexity of every
round to O/parenleftbig
d2/parenrightbig
?
The resulting efﬁcient online algorithm is known as online Bayesian
linear regression .
2.6.Aleatoric and epistemic uncertainty of BLR.
Prove for Bayesian linear regression that xω↙Σxωis the epistemic un-
certainty and σ2
nthe aleatoric uncertainty in yωunder the decomposi-
tion of Equation ( 2.18).
2.7.Hyperpriors.
We consider a dataset {(xi,yi)}n
i=1of size n, where xi↓Rddenotes
the feature vector and yi↓Rdenotes the label of the i-th data point.
Letεibe i.i.d. samples from the Gaussian distribution N(0,λ→1)for a
given λ>0. We collect the labels in a vector y↓Rn, the features in
a matrix X↓Rn∝d, and the noise in a vector ε↓Rn. The labels are
generated according to y=Xw+ε.


## Page 60

50 probabilistic artificial intelligence
To perform Bayesian Linear Regression, we consider the prior distribu-
tion over the parameter vector wto be N(µ,λ→1Id), where Iddenotes
thed-dimensional identity matrix and µ↓Rdis a hyperparameter.
1.Given this Bayesian data model, what is the conditional covariance
matrix Σy.=Var[y|X,µ,λ]?
2.Calculate the maximum likelihood estimate of the hyperparame-
terµ.
3.Since we are unsure about the hyperparameter µ, we decide to
model our uncertainty about µby placing the “ hyperprior ”µ↑N(0,Id).
Is the posterior distribution p(µ|X,y,λ)a Gaussian distribution?
If yes, what are its mean vector and covariance matrix?
4.What is the posterior distribution p(λ|X,y,µ)?
Hint: For any a ↓R,A↓Rn∝nit holds that det(aA)=andet(A).


## Page 61

3
Filtering
Before we continue in Chapter 4with the function-space view of re-
gression, we want to look at a seemingly different but very related
problem. We will study Bayesian learning and inference in the state
space model , where we want to keep track of the state of an agent over
time based on noisy observations. In this model, we have a sequence
of (hidden) states (Xt)t↓N0where Xtis in Rdand a sequence of obser-
vations (Yt)t↓N0where Ytis in Rm.
The process of keeping track of the state using noisy observations is
also known as Bayesian ﬁltering orrecursive Bayesian estimation . Fig-
ure3.1illustrates this process, where an agent perceives the current
state of the world and then updates its beliefs about the state based on
this observation.
perception
model p(θ|D1:t)
Dt
perception
model p(θ|D1:t+1)
Dt+1
world t world t+1
Figure 3.1: Schematic view of Bayesian
ﬁltering: An agent perceives the current
state of the world and updates its belief
accordingly.
We will discuss Bayesian ﬁltering more broadly in the next section. A
Kalman ﬁlter is an important special case of a Bayes’ ﬁlter, which uses
a Gaussian prior over the states and conditional linear Gaussians to
describe the evolution of states and observations. Analogously to the






















































Wasthisslapped


## Page 62

52 probabilistic artificial intelligence
previous chapter, we will see that inference in this model is tractable
due to the closedness properties of Gaussians.
Deﬁnition 3.1(Kalman ﬁlter) .AKalman ﬁlter is speciﬁed by a Gaus-
sian prior over the states,
X0↑N(µ,Σ),( 3.1)
and a conditional linear Gaussian motion model andsensor model ,
Xt+1.=FXt+εt F↓Rd∝d,εt↑N(0,Σx),( 3.2)
Yt.=HXt+ηt H↓Rm∝d,ηt↑N(0,Σy),( 3.3)
respectively. The motion model is sometimes also called transition
model ordynamics model . Crucially, Kalman ﬁlters assume that Fand
Hare known. In general, Fand Hmay depend on t. Also, εand η
may have a non-zero mean, commonly called a “drift”.
X1
X2
X3 ···
Y1
Y2
Y3
Figure 3.2: Directed graphical model of a
Kalman ﬁlter with hidden states Xtand
observables Yt.Because Kalman ﬁlters use conditional linear Gaussians, which we
have already seen in Equation ( 1.55), their joint distribution (over all
variables) is also Gaussian. This means that predicting the future states
of a Kalman ﬁlter is simply inference with multivariate Gaussians. In
Bayesian ﬁltering, however, we do not only want to make predictions
occasionally. In Bayesian ﬁltering, we want to keep track of states, that
is, predict the current state of an agent online.1To do this efﬁciently,1Here, online is common terminology to
say that we want to perform inference
at time twithout exposure to times t+
1,t+2, . . . , so in “real-time”.we need to update our belief about the state of the agent recursively,
similarly to our recursive Bayesian updates in Bayesian linear regres-
sion (see Section 2.1.3).
From the directed graphical model of a Kalman ﬁlter shown in Fig-
ure3.2, we can immediately gather the following conditional indepen-
dence relations,2 2Alternatively, they follow from the def-
inition of the motion and sensor models
as linear updates.Xt+1⇑X1:t→1,Y1:t→1|Xt,( 3.4)
Yt⇑X1:t→1|Xt (3.5)
Yt⇑Y1:t→1|Xt→1.( 3.6)
The ﬁrst conditional independence property is also known as the Markov
property , which we will return to later in our discussion of Markov
chains and Markov decision processes. This characterization of the
Kalman ﬁlter, yields the following factorization of the joint distribu-
tion:
p(x1:t,y1:t)=t
∏
i=1p(xi|x1:i→1)p(yi|x1:t,y1:i→1) using the product rule ( 1.11)
=p(x1)p(y1|x1)t
∏
i=2p(xi|xi→1)p(yi|xi). using the conditional independence
properties from ( 3.4), (3.5), and ( 3.6)(3.7)


## Page 63

filtering 53
3.1Conditioning and Prediction
We can describe Bayesian ﬁltering by the following recursive scheme
with the two phases, conditioning (also called “update”) and prediction :
Algorithm 3.2:Bayesian ﬁltering
1start with a prior over initial states p(x0)
2fort=1to∞do
3 assume we have p(xt|y1:t→1)
4 conditioning : compute p(xt|y1:t)using the new observation yt
5 prediction : compute p(xt+1|y1:t)
Let us consider the conditioning step ﬁrst:
p(xt|y1:t)=1
Zp(xt|y1:t→1)p(yt|xt,y1:t→1) using Bayes’ rule ( 1.45)
=1
Zp(xt|y1:t→1)p(yt|xt). using the conditional independence
structure ( 3.6)(3.8)
For the prediction step, we obtain,
p(xt+1|y1:t)=/integraldisplay
p(xt+1,xt|y1:t)dxt using the sum rule ( 1.7)
=/integraldisplay
p(xt+1|xt,y1:t)p(xt|y1:t)dxt using the product rule ( 1.11)
=/integraldisplay
p(xt+1|xt)p(xt|y1:t)dxt. using the conditional independence
structure ( 3.4)(3.9)
In general, these distributions can be very complicated, but for Gaus-
sians (i.e., Kalman ﬁlters) they can be expressed in closed-form.
Remark 3.3: Bayesian smoothing
Bayesian smoothing is a closely related task to Bayesian ﬁltering.
While Bayesian ﬁltering methods estimate the current state based
only on observations obtained before and at the current time step,
Bayesian smoothing computes the distribution of Xk|y1:twhere
t>k. That is Bayesian smoothing estimates Xkbased on data until
and beyond time k. Note that if k=t, then Bayesian smoothing
coincides with Bayesian ﬁltering.
Analogously to Equation ( 3.8),
p(xk|y1:t)∝p(xk|y1:k)p(yk+1:t|xk).( 3.10)
If we assume a Gaussian prior and conditional Gaussian transi-
tion and dynamics models (this is called Kalman smoothing ), then


## Page 64

54 probabilistic artificial intelligence
by the closedness properties of Gaussians, Xk|y1:tis a Gaus-
sian. Indeed, all terms of Equation ( 3.10) are Gaussian PDFs and
as seen in Equation ( 1.75), the product of two Gaussian PDFs is
again proportional to a Gaussian PDF.
The ﬁrst term, Xk|y1:k, is the marginal posterior of the hidden
states of the Kalman ﬁlter which can be obtained with Bayesian
ﬁltering.
By conditioning on Xk+1, we have for the second term,
p(yk+1:t|xk)=/integraldisplay
p(yk+1:t|xk,xk+1)p(xk+1|xk)dxk+1 using the sum rule ( 1.7) and product
rule ( 1.11)
=/integraldisplay
p(yk+1:t|xk+1)p(xk+1|xk)dxk+1 using the conditional independence
structure ( 3.5)
=/integraldisplay
p(yk+1|xk+1)p(yk+2:t|xk+1)p(xk+1|xk)dxk+1 using the conditional independence
structure ( 3.6)
(3.11)
Let us have a look at the terms in the product:
•p(yk+1|xk+1)is obtained from the sensor model,
•p(xk+1|xk)is obtained from the transition model, and
•p(yk+2:t|xk+1)can be computed recursively backwards in time.
This recursion results in linear equations resembling a Kalman
ﬁlter running backwards in time.
Thus, in the setting of Kalman smoothing, both factors of Equa-
tion ( 3.10) can be computed efﬁciently: one using a (forward)
Kalman ﬁlter; the other using a “backward” Kalman ﬁlter. More
concretely, in time O(t), we can compute the two factors for all
k↓[t]. This approach is known as two-ﬁlter smoothing or the
forward-backward algorithm .
3.2Kalman Filters
Let us return to the setting of Kalman ﬁlters where priors and likeli-
hoods are Gaussian. Here, we will see that the update and prediction
steps can be computed in closed form.
3.2.1Conditioning
The conditioning operation in Kalman ﬁlters is also called the Kalman
update. Before introducing the general Kalman update, let us consider
a simpler example:






















































relation toformalbaliprop


## Page 65

filtering 55
Example 3.4: Random walk in 1d
We use the simple motion and sensor models,3 3This corresponds to F=H=Iand a
drift of 0.
Xt+1|xt↑N (xt,σ2
x),( 3.12a)
Yt|xt↑N (xt,σ2
y).( 3.12b)
LetXt|y1:t↑N (µt,σ2
t)be our belief at time t. It can be shown
that Bayesian ﬁltering yields the belief Xt+1|y1:t+1↑N (µt+1,σ2
t+1)
at time t+1 where ? Problem 3.1
µt+1.=σ2
yµt+(σ2
t+σ2
x)yt+1
σ2
t+σ2x+σ2y,σ2
t+1.=(σ2
t+σ2
x)σ2
y
σ2
t+σ2x+σ2y.( 3.13)
Although looking intimidating at ﬁrst, this update has a very nat-
ural interpretation. Let us deﬁne the following quantity,
λ.=σ2
t+σ2
x
σ2
t+σ2x+σ2y=1→σ2
y
σ2
t+σ2x+σ2y↓[0, 1].( 3.14)
Using λ, we can write the updated mean as a convex combination
of the previous mean and the observation,
µt+1=(1→λ)µt+λyt+1 (3.15)
=µt+λ(yt+1→µt).( 3.16)
Intuitively, λis a form of “gain” that inﬂuences how much of the
new information should be incorporated into the updated mean.
For this reason, λis also called Kalman gain .
The updated variance can similarly be rewritten,
σ2
t+1=λσ2
y=(1→λ)(σ2
t+σ2
x).( 3.17)
In particular, observe that if µt=yt+1(i.e., we observe our predic-
tion), we have µt+1=µtas there is no new information. Similarly,
forσ2
y↘∞(i.e., we do not trust our observations), we have
λ↘0,µt+1=µt,σ2
t+1=σ2
t+σ2
x.
In contrast, for σ2
y↘0, we have
λ↘1,µt+1=yt+1,σ2
t+1=0.1 2 3 4 5 6
txFigure 3.3: Hidden states during a ran-
dom walk in one dimension.
The general formulas for the Kalman update follow the same logic as
in the above example of a one-dimensional random walk. Given the


## Page 66

56 probabilistic artificial intelligence
prior belief Xt|y1:t↑N(µt,Σt), we have
Xt+1|y1:t+1↑N(µt+1,Σt+1)where ( 3.18a)
µt+1.=Fµt+Kt+1(yt+1→HFµt),( 3.18b)
Σt+1.=(I→Kt+1H)(FΣtF↙+Σx).( 3.18c)
Hereby, Kt+1is the Kalman gain ,
Kt+1.=(FΣtF↙+Σx)H↙(H(FΣtF↙+Σx)H↙+Σy)→1↓Rd∝m.
(3.18d)
Note that ΣtandKtcan be computed ofﬂine as they are independent
of the observation yt+1.Fµtrepresents the expected state at time t+1,
and hence, HFµtcorresponds to the expected observation. Therefore,
the term yt+1→HFµtmeasures the error in the predicted observation
and the Kalman gain Kt+1appears as a measure of relevance of the
new observation compared to the prediction.
Example 3.5: Bayesian linear regression as a Kalman ﬁlter
Even though they arise from a rather different setting, it turns
out that Kalman ﬁlters are a generalization of Bayesian linear re-
gression! To see this, recall the online Bayesian linear regression
algorithm from Section 2.1.3. Observe that by keeping attempting
to estimate the (hidden) weights wωfrom sequential noisy obser-
vations yt, this algorithm performs Bayesian ﬁltering! Moreover,
we have used a Gaussian prior and likelihood. This is precisely
the setting of a Kalman ﬁlter!
Concretely, we are estimating the constant (i.e., F=I,ε=0)
hidden state xt=w(t)with prior w(0)↑N(0,σ2
pI).
Our sensor model is time-dependent, since in each iteration we
observe a different input xt. Furthermore, we only observe a
scalar-valued label yt.4 4That is, m=1 in our general Kalman
ﬁlter formulation from above.Formally, our sensor model is character-
ized by ht=x↙
tand noise ηt=εtwith εt↑N(0,σ2
n).
You will show in ? Problem 3.2 that the Kalman update ( 3.18) is the online
equivalent to computing the posterior of the weights in Bayesian
linear regression.
3.2.2Predicting
Using now that the marginal posterior of Xtis a Gaussian due to the
closedness properties of Gaussians, we have
Xt+1|y1:t↑N(ˆµt+1,ˆΣt+1),( 3.19)


## Page 67

filtering 57
and it sufﬁces to compute the prediction mean ˆ µt+1and covariance
matrix ˆΣt+1.
For the mean,
ˆµt+1=E[xt+1|y1:t]
=E[Fxt+εt|y1:t] using the motion model ( 3.2)
=FE[xt|y1:t] using linearity of expectation ( 1.20) and
E[εt]=0
=Fµt. using the mean of the Kalman update (3.20)
For the covariance matrix,
ˆΣt+1=E
(xt+1→ˆµt+1)(xt+1→ˆµt+1)↙y1:t
using the deﬁnition of the covariance
matrix ( 1.36)
=FE
(xt→µt)(xt→µt)↙y1:t
F↙+E
εtε↙
t
using ( 3.20), the motion model ( 3.2) and
thatεtis independent of the
observations =FΣtF↙+Σx.( 3.21)
Optional Readings
Kalman ﬁlters and related models are often called temporal models .
For a broader look at such models, read chapter 15of “Artiﬁcial
intelligence: a modern approach” (Russell and Norvig, 2002).
Discussion
In this chapter, we have introduced Kalman ﬁlters as a special case of
probabilistic ﬁltering where probabilistic inference can be performed
in closed form. Similarly to Bayesian linear regression, probabilistic in-
ference is tractable due to assuming Gaussian priors and likelihoods.
Indeed, learning linear models and Kalman ﬁlters are very closely re-
lated as seen in Example 3.5, and we will further explore this relation-
ship in Problem 4.3. We will refer back to ﬁltering in the second part
of this manuscript when we discuss sequential decision-making with
partial observability of the state space. Next, we return to the storyline
on “learning” using exact probabilistic inference.
Problems
3.1.Kalman update.
Derive the predictive distribution Xt+1|y1:t+1(3.13) of the Kalman ﬁl-
ter described in the above example using your knowledge about mul-
tivariate Gaussians from Section 1.2.3.
Hint: First compute the predictive distribution X t+1|y1:t.


## Page 68

58 probabilistic artificial intelligence
3.2.Bayesian linear regression as a Kalman ﬁlter.
Recall the speciﬁc Kalman ﬁlter from Example 3.5. With this model
the Kalman update ( 3.18) simpliﬁes to
kt=Σt→1xt
x↙
tΣt→1xt+σ2n,( 3.22a)
µt=µt→1+kt(yt→x↙
tµt→1),( 3.22b)
Σt=Σt→1→ktx↙
tΣt→1,( 3.22c)
with µ0=0andΣ0=σ2
pI. Note that the Kalman gain ktis a vector
inRd. We assume σ2
n=σ2
p=1 for simplicity.
Prove by induction that the (µt,Σt)produced by the Kalman update
are equivalent to (µ,Σ)from the posterior of Bayesian linear regres-
sion ( 2.10) given x1:t,y1:t. You may use that Σ→1
tkt=xt.
Hint: In the inductive step, ﬁrst prove the equivalence of Σtand then expand
Σ→1
tµtto prove the equivalence of µt.
3.3.Parameter estimation using Kalman ﬁlters.
Suppose that we want to estimate the value of an unknown constant π
using uncorrelated measurements
yt=π+ηt,ηt↑N(0,σ2
y).
1.How can this problem be formulated as a Kalman ﬁlter? Compute
closed form expressions for the Kalman gain and the variance of
the estimation error σ2
tin terms of t,σ2
y, and σ2
0.
2.What is the Kalman ﬁlter when t↘∞?
3.Suppose that one has no prior assumptions on π, meaning that
µ0=0 and σ2
0↘∞. Which well-known estimator does the Kalman
ﬁlter reduce to in this case?


## Page 69

4
Gaussian Processes
Let us remember our ﬁrst attempt from Chapter 2at scaling up Bayesian
linear regression to nonlinear functions. We saw that we can model
nonlinear functions by transforming the input space to a suitable higher-
dimensional feature space, but found that this approach scales poorly
if we require a large number of features. We then found something
remarkable: by simply changing our perspective from a weight-space
view to a function-space view, we could implement Bayesian linear
regression without ever needing to compute the features explicitly.
Under the function-space view, the key object describing the class of
functions we can model is not the features φ(x), but instead the kernel
function which only implicitly deﬁnes a feature space. Our key ob-
servation in this chapter is that we can therefore stop reasoning about
feature spaces, and instead directly work with kernel functions that
describe “reasonable” classes of functions.
We are still concerned with the problem of estimating the value of
a function f:X↘ Rat arbitrary points xω↓X given training
data {xi,yi}n
i=1, where the labels are assumed to be corrupted by ho-
moscedastic Gaussian noise with variance σ2
n,
yi=f(xi)+εi,εi↑N(0,σ2
n).
As in Chapter 2on Bayesian linear regression, we denote by Xthe
design matrix (collection of training inputs) and by ythe vector of
training labels. We will represent the unknown function value at a
point x↓Xby the random variable fx.=f(x). The collection of these
random variables is then called a Gaussian process if any ﬁnite subset
of them is jointly Gaussian:
Deﬁnition 4.1(Gaussian process, GP) .AGaussian process is an inﬁnite
set of random variables such that any ﬁnite number of them are jointly
Gaussian and such that they are consistent under marginalization.1 1That is, if you take a joint distribution
fornvariables and marginalize out one
of them, you should recover the joint dis-
tribution for the remaining n→1 vari-
ables.






















































Iii
date


## Page 70

60 probabilistic artificial intelligence
The fact that with a Gaussian process, any ﬁnite subset of the random
variables is jointly Gaussian is the key property allowing us to perform
exact probabilistic inference. Intuitively, a Gaussian process can be
interpreted as a normal distribution over functions — and is therefore
often called an “inﬁnite-dimensional Gaussian”.xp(f(x))yf(x)
Figure 4.1: A Gaussian process can be
interpreted as an inﬁnite-dimensional
Gaussian over functions. At any loca-
tion xin the domain, this yields a dis-
tribution over values f(x)shown in red.
The blue line corresponds to the MAP
estimate (i.e., mean function of the Gaus-
sian process), the dark gray region corre-
sponds to the epistemic uncertainty and
the light gray region denotes the addi-
tional aleatoric uncertainty.A Gaussian process is characterized by a mean function µ:X↘Rand
acovariance function (orkernel function )k:X∝X↘ Rsuch that for
any set of points A.={x1,..., xm}⇒X , we have
fA.=[fx1···fxm]↙↑N(µA,KAA) (4.1)
where
µA.=
µ(x1)
...
µ(xm)
,KAA.=
k(x1,x1)···k(x1,xm)
.........
k(xm,x1)···k(xm,xm)
.( 4.2)
We write f↑G P (µ,k). In particular, given a mean function, covari-
ance function, and using the homoscedastic noise assumption,
yω|xω↑N(µ(xω),k(xω,xω)+σ2
n),( 4.3)
with σ2(xω)= k(xω,xω)as the epistemic uncertainty and σ2
nas the
aleatoric uncertainty. Commonly, for notational simplicity, the mean
function is taken to be zero. Note that for a ﬁxed mean this is not a
restriction, as we can simply apply the zero-mean Gaussian process to
the difference between the mean and the observations.2 2For alternative ways of representing a
mean function, refer to section 2.7of
“Gaussian processes for machine learn-
ing” (Williams and Rasmussen, 2006 ). 4.1Learning and Inference
First, let us look at learning and inference in the context of Gaussian
processes. With slight abuse of our previous notation, let us denote the
set of observed points by A.={x1,..., xn}. Given a prior f↑G P(µ,k)
and the noisy observations yi=f(xi)+εiwith εi↑N(0,σ2
n), we can
then write the joint distribution of the observations y1:nand the noise-
free prediction fωat a test point xωas

y
fω
|xω,X↑N(˜µ,˜K), where ( 4.4)
˜µ.=
µA
µ(xω)
,˜K.=
KAA+σ2
nIk xω,A
k↙
xω,Ak(xω,xω)
,kx,A.=
k(x,x1)
...
k(x,xn)
.
(4.5)
Deriving the conditional distribution using ( 1.53), we obtain that the
Gaussian process posterior is given by
f|x1:n,y1:n↑G P(µ′,k′), where ( 4.6)






















































random variable
Keneldefineseverything
rest isjustsomeoffset
whywritten
thismay


## Page 71

gaussian processes 61
µ′(x).=µ(x)+k↙
x,A(KAA+σ2
nI)→1(yA→µA),( 4.7)
k′(x,x′).=k(x,x′)→k↙
x,A(KAA+σ2
nI)→1kx′,A.( 4.8)
Observe that analogously to Bayesian linear regression, the posterior
covariance can only decrease when conditioning on additional data,
and is independent of the observations yi.
We already studied inference in the function-space view of Bayesian
linear regression, but did not make the predictive posterior explicit.
Using Equation ( 4.6), the predictive posterior at xωis simply
fω|xω,x1:n,y1:n↑N(µ′(xω),k′(xω,xω)).( 4.9)
4.2Sampling
Often, we are not interested in the full predictive posterior distribution,
but merely want to obtain samples of our Gaussian process model. We
will brieﬂy examine two approaches.
1.For the ﬁrst approach, consider a discretized subset of points
f.=[f1,..., fn]
that we want to sample.3Note that f↑N(µ,K). We have already3For example, if we want to render the
function, the length of this vector could
be guided by the screen resolution.seen in Equation ( 1.54) that
f=K1/2ε+µ (4.10)
where K1/2is the square root of Kand ε↑N(0,I)is standard
Gaussian noise.4However, computing the square root of Ktakes4We discuss square roots of matrices in
Appendix A. 2.O/parenleftbig
n3/parenrightbig
time.
2.For the second approach, recall the product rule ( 1.11),
p(f1,..., fn)=n
∏
i=1p(fi|f1:i→1).
That is the joint distribution factorizes neatly into a product where
each factor only depends on the “outcomes” of preceding factors.
We can therefore obtain samples one-by-one, each time condition-
ing on one more observation:
f1↑p(f1)
f2↑p(f2|f1)
f3↑p(f3|f1,f2)
...(4.11)
This general approach is known as forward sampling . Due to the ma-
trix inverse in the formula of the GP posterior ( 4.6), this approach
also takes O/parenleftbig
n3/parenrightbig
time.
We will discuss more efﬁcient approximate sampling methods in Sec-
tion4.5.






















































pderivationandmeaning
showhow


## Page 72

62 probabilistic artificial intelligence
4.3Kernel Functions
We have seen that kernel functions are the key object describing the
class of functions a Gaussian process can model. Depending on the
kernel function, the “shape” of functions that are realized from a Gaus-
sian process varies greatly. Let us recap brieﬂy from Section 2.4what
a kernel function is:
Deﬁnition 4.2(Kernel function) .Akernel function k :X∝X ↘ R
satisﬁes
•k(x,x′)= k(x′,x)for any x,x′↓X(symmetry), and
•KAAis positive semi-deﬁnite for any A⇒X.
The two deﬁning conditions ensure that for any A⇒X,KAAis a valid
covariance matrix. We say that a kernel function is positive deﬁnite if
KAAis positive deﬁnite for any A⇒X.
Intuitively, the kernel function evaluated at locations xandx′describes
how f(x)and f(x′)are related, which we can express formally as
k(x,x′)= Cov
f(x),f(x′)
.( 4.12)
Ifxand x′are “close”, then f(x)and f(x′)are usually taken to be
positively correlated, encoding a “smooth” function.
In the following, we will discuss some of the most common kernel
functions, how they can be combined to create “new” kernels, and
how we can characterize the class of functions they can model.
4.3.1Common Kernelsxf(x)
Figure 4.2: Functions sampled according
to a Gaussian process with a linear ker-
nel and φ=id.f(x)xFigure4.3: Functions sampled accord-ing to a Gaussian process with a linearkernel andφ(x)=[1,x,x2](left) andφ(x)=sin(x)(right).First, we look into some of the most commonly used kernels. Often anadditional factorσ2(output scale) is added, which we assume here tobe 1 for simplicity.1.Thelinear kernelis deﬁned ask(x,x′;φ).=φ(x)↙φ(x′)(4.13)whereφis a nonlinear transformation as introduced in Section2.3or the identity.Remark4.3: GPs with linear kernel and BLRA Gaussian process with a linear kernel is equivalent to Bayesianlinear regression. This follows directly from the function-spaceview of Bayesian linear regression (see Section2.4) and com-paring the derived kernel function (2.20) with the deﬁnition ofthe linear kernel (4.13).






















































becauseitisanovaranh


## Page 73

gaussian processes 63
2.TheGaussian kernel (also known as squared exponential kernel orra-
dial basis function (RBF) kernel ) is deﬁned as
k(x,x′;h).=exp/parenleftigg
→≃x→x′≃2
2
2h2
(4.14)
where his its length scale . The larger the length scale h, the smoother
the resulting functions.5Furthermore, it turns out that the feature5As the length scale is increased, the ex-
ponent of the exponential increases, re-
sulting in a higher dependency between
locations. space (think back to Section 2.4!) corresponding to the Gaussian
kernel is “inﬁnitely dimensional”, as you will show in ? Problem 4.1 . So the
Gaussian kernel already encodes a function class that we were not
able to model under the weight-space view of Bayesian linear re-
gression.f(x)
xFigure 4.4: Functions sampled according
to a Gaussian process with a Gaussian
kernel and length scales h=5 (left) and
h=1 (right).
→2 0 2
x→x′0.000.250.500.751.00k(x→x′)
Figure 4.5: Gaussian kernel with length
scales h=1,h=0.5, and h=0.2.3.TheLaplace kernel (also known as exponential kernel ) is deﬁned as
k(x,x′;h).=exp/parenleftbigg
→≃x→x′≃2
h/parenrightbigg
.( 4.15)
As can be seen in Figure 4.7, samples from a GP with Laplace
kernel are non-smooth as opposed to the samples from a GP with
Gaussian kernel.
→2 0 2
x→x′0.000.250.500.751.00k(x→x′)
Figure 4.6: Laplace kernel with length
scales h=1,h=0.5, and h=0.2.f(x)
xFigure 4.7: Functions sampled accord-
ing to a Gaussian process with a Laplace
kernel and length scales h=10 000 (left)
andh=10 (right).
4.TheMatérn kernel trades the smoothness of the Gaussian and the
Laplace kernels. As such, it is frequently used in practice to model






















































hit
interpretation


## Page 74

64 probabilistic artificial intelligence
“real world” functions that are relatively smooth. It is deﬁned as
k(x,x′;ν,h).=21→ν
Γ(ν)/parenleftigg∞
2ν≃x→x′≃2
hν
Kν/parenleftigg∞
2ν≃x→x′≃2
h
(4.16)
where Γis the Gamma function, Kνthe modiﬁed Bessel function
of the second kind, and ha length scale parameter. For ν=1/2, the
Matérn kernel is equivalent to the Laplace kernel. For ν↘∞, the
Matérn kernel is equivalent to the Gaussian kernel. The resulting
functions are ∋ν△→1 times mean square differentiable.6In partic-6Refer to Remark A. 12for the deﬁni-
tions of mean square continuity and dif-
ferentiability.ular, GPs with a Gaussian kernel are inﬁnitely many times mean
square differentiable whereas GPs with a Laplace kernel are mean
square continuous but not mean square differentiable.
4.3.2Composing Kernels
Given two kernels k1:X∝X ↘ Rand k2:X∝X ↘ R, they can
be composed to obtain a new kernel k:X∝X↘ Rin the following
ways:
•k(x,x′).=k1(x,x′)+k2(x,x′),
•k(x,x′).=k1(x,x′)·k2(x,x′),
•k(x,x′).=c·k1(x,x′)for any c>0,
•k(x,x′).=f(k1(x,x′))for any polynomial fwith positive coefﬁ-
cients or f=exp.
For example, the additive structure of a function f(x).=f1(x)+f2(x)
can be easily encoded in GP models. Suppose that f1↑G P (µ1,k1)
and f2↑G P (µ2,k2), then the distribution of the sum of those two
functions f=f1+f2↑G P(µ1+µ2,k1+k2)is another GP.7 7We use f.=f1+f2to denote the func-
tion f(·)= f1(·)+f2(·).
Whereas the addition of two kernels k1and k2can be thought of as
anORoperation (i.e., the kernel has high value if either k1ork2have
high value), the multiplication of k1and k2can be thought of as an
AND operation (i.e., the kernel has high value if both k1andk2have
high value). For example, the product of two linear kernels results in
functions which are quadratic.
As mentioned previously, the constant cof a scaled kernel function
k′(x,x′).=c·k(x,x′)is generally called the output scale of a kernel,
and it scales the variance Var [f(x)]=c·k(x,x)of the predictions f(x)
from GP(µ,k′).
Optional Readings
For a broader introduction to how kernels can be used and com-
bined to model certain classes of functions, read






















































pnhi.FIthis
thethis
Eti


## Page 75

gaussian processes 65
•chapter 2of “Automatic model construction with Gaussian pro-
cesses” (Duvenaud, 2014 ) also known as the “kernel cookbook”,
•chapter 4of “Gaussian processes for machine learning” (Williams
and Rasmussen, 2006 ).
4.3.3Stationarity and Isotropy
Kernel functions are commonly classiﬁed according to two properties:
Deﬁnition 4.4(Stationarity and isotropy) .A kernel k:Rd∝Rd↘R
is called
•stationary (orshift-invariant ) if there exists a function ˜ksuch that
˜k(x→x′)= k(x,x′), and
•isotropic if there exists a function ˜ksuch that ˜k(≃x→x′≃)= k(x,x′)
with ≃·≃any norm.
Note that stationarity is a necessary condition for isotropy. In other
words, isotropy implies stationarity.
Example 4.5: Stationarity and isotropy of kernels
stationary isotropic
linear kernel no no
Gaussian kernel yes yes
k(x,x′).=exp(→≃x→x′≃2
M)
where Mis positive semi-deﬁniteyes no ≃·≃Mdenotes the Mahalanobis norm
induced by matrix M
Forx′=x, stationarity implies that the kernel must only depend
on0. In other words, a stationary kernel must depend on relative
locations only. This is clearly not the case for the linear kernel,
which depends on the absolute locations of xand x′. Therefore,
the linear kernel cannot be isotropic either.
For the Gaussian kernel, isotropy follows immediately from its
deﬁnition.
The last kernel is clearly stationary by deﬁnition, but not isotropic
for general matrices M. Note that for M=Iit is indeed isotropic.
Stationarity encodes the idea that relative location matters more than
absolute location: the process “looks the same” no matter where we
shift it in the input space. This is often appropriate when we believe
the same statistical behavior holds across the entire domain (e.g., no
region is special). Isotropy goes one step further by requiring that


## Page 76

66 probabilistic artificial intelligence
the kernel depends only on the distance between points, so that all
directions in the space are treated equally. In other words, there is no
preferred orientation or axis. This is especially useful in settings where
we expect uniform behavior in every direction (as with the Gaussian
kernel). Such kernels are simpler to specify and interpret since we
only need a single “scale” (like a length scale) rather than multiple
parameters or directions.
4.3.4Reproducing Kernel Hilbert Spaces
We can characterize the precise class of functions that can be modeled
by a Gaussian process with a given kernel function. This correspond-
ing function space is called a reproducing kernel Hilbert space (RKHS),
and we will discuss it brieﬂy in this section.
Recall that Gaussian processes keep track of a posterior distribution
f|x1:n,y1:nover functions. We will in fact show later that the corre-
sponding MAP estimate ˆfcorresponds to the solution to a regularized
optimization problem in the RKHS space of functions. This duality
is similar to the duality between the MAP estimate of Bayesian linear
regression and ridge regression we observed in Chapter 2. So what is
the reproducing kernel Hilbert space of a kernel function k?
Deﬁnition 4.6(Reproducing kernel Hilbert space, RKHS) .Given a ker-
nelk:X∝X↘ R, its corresponding reproducing kernel Hilbert space is
the space of functions fdeﬁned as
Hk(X).=
f(·)=n
∑
i=1αik(xi,·):n↓N,xi↓X,αi↓R
.( 4.17)
The inner product of the RKHS is deﬁned as
▽f,g̸k.=n
∑
i=1n′
∑
j=1αiα′
jk(xi,x′
j),( 4.18)
where g(·)= ∑n′
j=1α′
jk(x′
j,·), and induces the norm ≃f≃k=
▽f,f̸k.
You can think of the norm as measuring the “smoothness” or “com-
plexity” of f.? Problem 4.4(2)
It is straightforward to check that for all x↓X ,k(x,·)↓H k(X).
Moreover, the RKHS inner product ▽·,·̸ksatisﬁes for all x↓X and
f↓H k(X)that f(x)= ▽f(·),k(x,·)̸kwhich is also known as the
reproducing property ? Problem 4.4(1) . That is, evaluations of RKHS functions fare
inner products in Hk(X)parameterized by the “feature map” k(x,·).
Therepresenter theorem (Schölkopf et al., 2001) characterizes the solu-
tion to regularized optimization problems in RKHSs:






















































Hifi.im


## Page 77

gaussian processes 67
Theorem 4.7(Representer theorem) .? Problem 4.5 Let k be a kernel and let λ>0.
For f ↓Hk(X)and training data {(xi,f(xi))}n
i=1, letL(f(x1),..., f(xn))↓
R∪{∞}denote any loss function which depends on f only through its eval-
uation at the training points. Then, any minimizer
ˆf↓arg min
f↓Hk(X)L(f(x1),..., f(xn)) + λ≃f≃2
k(4.19)
admits a representation of the form
ˆf(x)= ˆε↙kx,{xi}n
i=1=n
∑
i=1ˆαik(x,xi)for some ˆε↓Rn.( 4.20)
This statement is remarkable: the solutions to general regularized op-
timization problems over the generally inﬁnite-dimensional space of
functions Hk(X)can be represented as a linear combination of the
kernel functions evaluated at the training points. The representer the-
orem can be used to show that the MAP estimate of a Gaussian process
corresponds to the solution of a regularized linear regression problem
in the RKHS of the kernel function, namely, ? Problem 4.6
ˆf.=arg min
f↓Hk(X)→logp(y1:n|x1:n,f)+1
2≃f≃2
k.( 4.21)
Here, the ﬁrst term corresponds to the likelihood, measuring the “qual-
ity of ﬁt”. The regularization term limits the “complexity” of ˆf. Reg-
ularization is necessary to prevent overﬁtting since in an expressive
RKHSs, there may be many functions that interpolate the training data
perfectly. This shows the close link between Gaussian process regres-
sion and Bayesian linear regression, with the kernel function kgener-
alizing the inner product of feature maps to feature spaces of possi-
bly “inﬁnite dimensionality”. Because solutions can be represented as
linear combinations of kernel evaluations at the training points, Gaus-
sian processes remain computationally tractable even though they can
model functions over “inﬁnite-dimensional” feature spaces.
4.4Model Selection
We have not yet discussed how to pick the hyperparameters θ(e.g.,
parameters of kernels). A common technique in supervised learning
is to select hyperparameters θ, such that the resulting function esti-
mate ˆfθleads to the most accurate predictions on hold-out validation
data. After reviewing this approach, we contrast it with a probabilistic
approach to model selection, which avoids using point estimates of ˆfθ
and rather utilizes the full posterior.


## Page 78

68 probabilistic artificial intelligence
4.4.1Optimizing Validation Set Performance
A common approach to model selection is to split our data Dinto
separate training set Dtrain .={(xtrain
i,ytrain
i)}n
i=1and validation sets
Dval .={(xval
i,yval
i)}m
i=1. We then optimize the model for a parameter
candidate θjusing the training set. This is usually done by picking a
point estimate (like the MAP estimate),
ˆfj.=arg max
fp(f|xtrain
1:n,ytrain
1:n).( 4.22)
Then, we score θjaccording to the performance of ˆfjon the validation
set,
ˆθ.=arg max
θjp(yval
1:m|xval
1:m,ˆfj).( 4.23)
This ensures that ˆfjdoes not depend on Dval.
Remark 4.8: Approximating population risk
Why is it useful to separate the data into a training and a vali-
dation set? Recall from Appendix A. 3.5that minimizing the em-
pirical risk without separating training and validation data may
lead to overﬁtting as both the loss and ˆfjdepend on the same
data D. In contrast, using independent training and validation
sets, ˆfjdoes not depend on Dval, and we have that
1
mm
∑
i=1ε(yval
i|xval
i,ˆfj)⇔E(x,y)↑P
ε(y|x,ˆfj)
,( 4.24)
using Monte Carlo sampling.8 8We generally assume Diid↑P, in par-
ticular, we assume that the individual
samples of the data are i.i.d.. Recall
that in this setting, Hoeffding’s inequal-
ity (A. 41) can be used to gauge how
large mshould be.In words, for reasonably large m,
minimizing the empirical risk as we do in Equation ( 4.23) approx-
imates minimizing the population risk.
While this approach often is quite effective at preventing overﬁtting
as compared to using the same data for training and picking ˆθ, it still
collapses the uncertainty in finto a point estimate. Can we do better?
4.4.2Maximizing the Marginal Likelihood
We have already seen for Bayesian linear regression, that picking a
point estimate loses a lot of information. Instead of optimizing the
effects of θfor a speciﬁc point estimate ˆfof the model f,maximizing
the marginal likelihood optimizes the effects of θacross all realizations
off. In this approach, we obtain our hyperparameter estimate via
ˆθMLE.=arg max
θp(y1:n|x1:n,θ) using the deﬁnition of marginal
likelihood in Bayes’ rule ( 1.45)(4.25)


## Page 79

gaussian processes 69
=arg max
θ/integraldisplay
p(y1:n,f|x1:n,θ)df by conditioning on fusing the sum rule
(1.7)
=arg max
θ/integraldisplay
p(y1:n|x1:n,f,θ)p(f|θ)df. using the product rule ( 1.11) (4.26)
Remarkably, this approach typically avoids overﬁtting even though we
do not use a separate training and validation set. The following ta-
ble provides an intuitive argument for why maximizing the marginal
likelihood is a good strategy.
likelihood prior
“underﬁt” model
(too simple θ)small for “almost all” flarge
“overﬁt” model
(too complex θ)large for “few” f
small for “most” fsmall
“just right” moderate for “many” fmoderateTable 4.1: The table gives an intuitive ex-
planation of effects of parameter choices
θon the marginal likelihood. Note that
words in quotation marks refer to in-
tuitive quantities, as we have inﬁnitely
many realizations of f.
all possible data setsmarginal likelihoodsimpleintermediate
complexFigure 4.8: A schematic illustration of
the marginal likelihood of a simple, in-
termediate, and complex model across
all possible data sets.For an “underﬁt” model, the likelihood is mostly small as the data
cannot be well described, while the prior is large as there are “fewer”
functions to choose from. For an “overﬁt” model, the likelihood is
large for “some” functions (which would be picked if we were only
minimizing the training error and not doing cross validation) but small
for “most” functions. The prior is small, as the probability mass has
to be distributed among “more” functions. Thus, in both cases, one
term in the product will be small. Hence, maximizing the marginal
likelihood naturally encourages trading between a large likelihood and
a large prior.
In the context of Gaussian process regression, recall from Equation ( 4.3)
that
y1:n|x1:n,θ↑N(0,Kf,θ+σ2
nI) (4.27)
where Kf,θdenotes the kernel matrix at the inputs x1:ndepending on
the kernel function parameterized by θ. We write Ky,θ.=Kf,θ+σ2
nI.
Continuing from Equation ( 4.25), we obtain
ˆθMLE=arg max
θN(y;0,Ky,θ)
=arg min
θ1
2y↙K→1
y,θy+1
2log det/parenleftbig
Ky,θ/parenrightbig+n
2log 2 π taking the negative logarithm (4.28)
=arg min
θ1
2y↙K→1
y,θy+1
2log det/parenleftbig
Ky,θ/parenrightbig
the last term is independent of θ (4.29)
The ﬁrst term of the optimization objective describes the “goodness of
ﬁt” (i.e., the “alignment” of ywith Ky,θ). The second term character-
izes the “volume” of the model class. Thus, this optimization naturally
trades the aforementioned objectives.


## Page 80

70 probabilistic artificial intelligence
Marginal likelihood maximization is an empirical Bayes method. Often
it is simply referred to as empirical Bayes . It also has the nice property
that the gradient of its objective (the MLL loss) can be expressed in
closed-from ? Problem 4.7 ,
∂
∂ϱjlogp(y1:n|x1:n,θ)=1
2tr/parenleftigg
(εε↙→K→1
y,θ)∂Ky,θ
∂ϱj
(4.30)
where ε.=K→1
y,θyand tr (M)is the trace of a matrix M. This optimiza-
tion problem is, in general, non-convex. Figure 4.10gives an example
of two local optima according to empirical Bayes.0 100 200
# of iterations0.00.51.0MLL lossFigure 4.9: An example of model selec-
tion by maximizing the log likelihood
(without hyperpriors) using a linear ,
quadratic ,Laplace ,Matérn (ν= 3/2),
andGaussian kernel, respectively. They
are used to learn the function
x∀↘sin(x)
x+ε,ε↑N(0, 0.01 )
using SGD with learning rate 0.1.Taking a step back, observe that taking a probabilistic perspective on
model selection naturally led us to consider all realizations of our
model finstead of using point estimates. However, we are still us-
ing point estimates for our model parameters θ. Continuing on our
probabilistic adventure, we could place a prior p(θ)on them too.9We
9Such a prior is called hyperprior .could use it to obtain the MAP estimate (still a point estimate!) which
adds an additional regularization term
ˆθMAP.=arg max
θp(θ|x1:n,y1:n) (4.31)
=arg min
θ→logp(θ)→logp(y1:n|x1:n,θ). using Bayes’ rule ( 1.45) and then taking
the negative logarithm(4.32)
An alternative approach is to consider the full posterior distribution
over parameters θ. The resulting predictive distribution is, however,
intractable,
p(yω|xω,x1:n,y1:n)=/integraldisplay/integraldisplay
p(yω|xω,f)·p(f|x1:n,y1:n,θ)·p(θ)df dθ.
(4.33)
Recall that as the mode of Gaussians coincides with their mean, the
MAP estimate corresponds to the mean of the predictive posterior.
As a ﬁnal note, observe that in principle, there is nothing stopping us
from descending deeper in the probabilistic hierarchy. The prior on the
model parameters θis likely to have parameters too. Ultimately, we
need to break out of this hierarchy of dependencies and choose a prior.
4.5Approximations
To learn a Gaussian process, we need to invert n∝nmatrices, hence
the computational cost is O/parenleftbig
n3/parenrightbig
. Compare this to Bayesian linear re-
gression which allows us to learn a regression model in O/parenleftbig
nd2/parenrightbig
time
(even online) where dis the feature dimension. It is therefore natural
to look for ways of approximating a Gaussian process.






















































YE


## Page 81

gaussian processes 71
→5.0 →2.5 0.0 2.5 5.0
x→2→1012f(x)→5.0 →2.5 0.0 2.5 5.0
x→2→1012100101lengthscaleh10→1100101noise standard deviation σnFigure 4.10: The top plot shows contour
lines of an empirical Bayes with two lo-
cal optima. The bottom two plots show
the Gaussian processes corresponding
to the two optimal models. The left
model with smaller lengthscale is chosen
within a more ﬂexible class of models,
while the right model explains more ob-
servations through noise. Adapted from
ﬁgure 5.5of “Gaussian processes for
machine learning” (Williams and Ras-
mussen, 2006 ).


## Page 82

72 probabilistic artificial intelligence
4.5.1Local Methods
Recall that during forward sampling, we had to condition on a larger
and larger number of previous samples. When sampling at a loca-
tionx, a very simple approximation is to only condition on those sam-
ples x′that are “close” (where |k(x,x′)|⇓τfor some τ>0). Essen-
tially, this method “cuts off the tails” of the kernel function k. However,
τhas to be chosen carefully as if τis chosen too large, samples become
essentially independent.
This is one example of a sparse approximation of a Gaussian process. We
will discuss more advanced sparse approximations known as “induc-
ing point methods” in Section 4.5.3.
4.5.2Kernel Function Approximation
Another method is to approximate the kernel function directly. The
idea is to construct a “low-dimensional” feature map φ:Rd↘Rm
that approximates the kernel,
k(x,x′)⇔φ(x)↙φ(x′).( 4.34)
Then, we can apply Bayesian linear regression, resulting in a time com-
plexity of O/parenleftbig
nm2+m3/parenrightbig
.
One example of this approach are random Fourier features , which we
will discuss in the following.
0 1
Re0iImϕcosϕsinϕeiϕ
Figure4.11: Illustration of Euler’s for-mula. It can be seen thateiϕcorrespondsto a (counter-clockwise) rotation on theunit circle asϕvaries from 0 to 2π.Remark4.9: Fourier transformFirst, let us remind ourselves of Fourier transformations. TheFourier transform is a method of decomposing frequencies intotheir individual components.RecallEuler’s formulawhich states that for anyx↓R,eix=cosx+isinx(4.35)whereiis the imaginary unit of complex numbers. The formula isillustrated in Figure4.11. Note thate→i2πxcorresponds to rotatingclockwise around the unit circle inR2— completing a rotationwheneverx↓Rreaches the next natural number.We can scalexby a frequencyξ:e→i2πξx. Ifx↓Rd, we can alsoscale each componentjofxby a different frequencyξ(j). Multi-plying a functionf:Rd↘Rwith the rotation around the unitcircle with given frequenciesξ, yields a quantity that describes the






















































Idon'tand


## Page 83

gaussian processes 73
amplitude of the frequencies ξ,
ˆf(ξ).=/integraldisplay
Rdf(x)e→i2πξ↙xdx.( 4.36)
ˆfis called the Fourier transform off.fis called the inverse Fourier
transform ofˆf, and can be computed using
f(x)=/integraldisplay
Rdˆf(ξ)ei2πξ↙xdξ.( 4.37)
It is common to write ω.=2πξ. See Figure 4.12for an example.
Refer to “But what is the Fourier Transform? A visual introduc-
tion” (Sanderson, 2018 ) for a visual introduction.→11
x01f(x)
→ππ
ω02ˆf(ω)
Figure 4.12: The Fourier transform of a
rectangular pulse,
f(x).=
1x↓[→1, 1]
0 otherwise,
is given by
ˆf(ω)=/integraldisplay1
→1e→iωxdx=1
iω
eiω→e→iω
=2 sin(ω)
ω.Because a stationary kernel k:Rd∝Rd↘Rcan be interpreted as a
function in one variable, it has an associated Fourier transform which
we denote by p(ω). That is,
k(x→x′)=/integraldisplay
Rdp(ω)eiω↙(x→x′)dω.( 4.38)
Fact 4.10(Bochner’s theorem) .A continuous stationary kernel on Rdis
positive deﬁnite if and only if its Fourier transform p (ω)is non-negative.
Bochner’s theorem implies that when a continuous and stationary
kernel is positive deﬁnite and scaled appropriately, its Fourier trans-
form p(ω)is a proper probability distribution. In this case, p(ω)is
called the spectral density of the kernel k.
Remark 4.11: Eigenvalue spectrum of stationary kernels
When a kernel kis stationary (i.e., a univariate function of x→x′),
its eigenfunctions (with respect to the usual Lebesgue measure)
turn out to be the complex exponentials exp (iω↙(x→x′)). In
simpler terms, you can think of these exponentials as “building
blocks” at different frequencies ω. The spectral density p(ω)as-
sociated with the kernel tells you how strongly each frequency
contributes, i.e., how large the corresponding eigenvalue is.
A key insight of this analysis is that the rate at which these magni-
tudes p(ω)decay with increasing frequency ωreveals the smooth-
ness of the processes governed by the kernel. If a kernel allocates
more “power” to high frequencies (meaning the spectral density
decays slowly), the resulting processes will appear “rougher”.
Conversely, if high-frequency components are suppressed, the pro-
cess will appear “smoother”.


## Page 84

74 probabilistic artificial intelligence
For an in-depth introduction to the eigenfunction analysis of ker-
nels, refer to section 4.3of “Gaussian processes for machine learn-
ing” (Williams and Rasmussen, 2006 ).
Example 4.12: Spectral density of the Gaussian kernel
The Gaussian kernel with length scale hhas the spectral density
p(ω)=/integraldisplay
Rdk(x→x′;h)e→iω↙(x→x′)d(x→x′) using the deﬁnition of the Fourier
transform ( 4.36)
=/integraldisplay
Rdexp/parenleftigg
→≃x≃2
2
2h2→iω↙x
dx using the deﬁnition of the Gaussian
kernel ( 4.14)
=(2h2π)d/2exp/parenleftigg
→h2≃ω≃2
2
2
.( 4.39)
The key idea is now to interpret the kernel as an expectation,
k(x→x′)=/integraldisplay
Rdp(ω)eiω↙(x→x′)dω from Equation ( 4.38)
=Eω↑p
eiω↙(x→x′)
by the deﬁnition of expectation ( 1.19)
=Eω↑p
cos(ω↙x→ω↙x′)+isin(ω↙x→ω↙x′)
. using Euler’s formula ( 4.35)
Observe that as both kand pare real, convergence of the integral im-
plies Eω↑p
sin(ω↙x→ω↙x′)=0. Hence,
=Eω↑p
cos(ω↙x→ω↙x′)
=Eω↑pEb↑Unif([0,2π])
cos((ω↙x+b)→(ω↙x′+b))
expanding with b→b
=Eω↑pEb↑Unif([0,2π])
cos(ω↙x+b)cos(ω↙x′+b)
+sin(ω↙x+b)sin(ω↙x′+b)using the angle subtraction identity,
cos(α→β)= cosαcosβ+sinαsinβ
=Eω↑pEb↑Unif([0,2π])
2 cos(ω↙x+b)cos(ω↙x′+b)
using
Eb[cos(α+b)cos(β+b)]
=Eb[sin(α+b)sin(β+b)]
forb↑Unif([0, 2π])=Eω↑p,b↑Unif([0,2π])
zω,b(x)·zω,b(x′)
(4.40)
where zω,b(x).=∞
2 cos(ω↙x+b),
⇔1
mm
∑
i=1zω(i),b(i)(x)·zω(i),b(i)(x′) using Monte Carlo sampling to estimate
the expectation, see Example A. 6(4.41)
for independent samples ω(i)iid↑pandb(i)iid↑Unif([0, 2π]),
=z(x)↙z(x′) (4.42)
where the (randomized) feature map of random Fourier features is
z(x).=1∞m[zω(1),b(1)(x),..., zω(m),b(m)(x)]↙.( 4.43)


## Page 85

gaussian processes 75
Intuitively, each component of the feature map z(x)projects xonto a
random direction ωdrawn from the (inverse) Fourier transform p(ω)
ofk(x→x′), and wraps this line onto the unit circle in R2. After trans-
forming two points xandx′in this way, their inner product is an unbi-
ased estimator of k(x→x′). The mapping zω,b(x)=∞
2 cos(ω↙x+b)
additionally rotates the circle by a random amount band projects the
points onto the interval [0, 1].→5.0→2.5 0.0 2.5 5.0024f(x)
→5.0→2.5 0.0 2.5 5.0
x→1012
Figure 4.13: Example of random Fourier
features with where the number of fea-
tures mis 5 (top) and 10 (bottom), re-
spectively. The noise-free true function
is shown in black and the mean of the
Gaussian process is shown in blue.Rahimi et al. ( 2007 ) show that Bayesian linear regression with the fea-
ture map zapproximates Gaussian processes with a stationary kernel:
Theorem 4.13(Uniform convergence of Fourier features) .Suppose M
is a compact subset of Rdwith diameter diam (M). Then for a stationary
kernel k, the random Fourier features z, and any ε>0it holds that
P/parenleftigg
sup
x,x′↓Mz(x)↙z(x′)→k(x→x′)⇓ε
↖28/parenleftbiggσpdiam (M)
ε/parenrightbigg2
exp/parenleftbigg
→mε2
8(d+2)/parenrightbigg(4.44)
where σ2
p.=Eω↑p
ω↙ω
is the second moment of p, m is the dimension
ofz(x), and d is the dimension of x.? Problem 4.8
Note that the error probability decays exponentially fast in the dimen-
sion of the Fourier feature space.
4.5.3Data Sampling
Another natural approach is to only consider a (random) subset of the
training data during learning. The naive approach is to subsample
uniformly at random. Not very surprisingly, we can do much better.→5 0 5
x012f(x)
Figure 4.14: Inducing points uare shown
as vertical dotted red lines. The noise-
free true function is shown in black and
the mean of the Gaussian process is
shown in blue. Observe that the true
function is approximated “well” around
the inducing points.One subsampling method is the inducing points method (Quinonero-
Candela and Rasmussen, 2005 ). The idea is to summarize the data
around so-called inducing points.10For now, let us consider an arbi-
10The inducing points can be treated as
hyperparameters.trary set of inducing points,
U.={x1,..., xk}.
Then, the original Gaussian process can be recovered using marginal-
ization,
p(fω,f)=/integraldisplay
Rkp(fω,f,u)du=/integraldisplay
Rkp(fω,f|u)p(u)du, using the sum rule ( 1.7) and product
rule ( 1.11)(4.45)
where f.=[f(x1)···f(xn)]↙and fω .=f(xω)at some evaluation
point xω↓X. We use u.=[f(x1)···f(xk)]↙↓Rkto denote the pre-
dictions of the model at the inducing points U. Due to the marginaliza-
tion property of Gaussian processes ( 4.1), we have that u↑N(0,KUU).


## Page 86

76 probabilistic artificial intelligence
The key idea is to approximate the joint prior, assuming that fωand f
are conditionally independent given u,
p(fω,f)⇔/integraldisplay
Rkp(fω|u)p(f|u)p(u)du.( 4.46)
Here, p(f|u)and p(fω|u)are commonly called the training condi-
tional and the testing conditional , respectively. Still denoting the obser-
vations by A={x1,..., xn}and deﬁning ω.={xω}, we know, using
the closed-form expression for conditional Gaussians ( 1.53),
p(f|u)↑N(f;KAUK→1
UUu,KAA→QAA),( 4.47a)
p(fω|u)↑N(fω;KωUK→1
UUu,Kωω→Qωω) (4.47b)
where Qab.=KaUK→1
UUKUb. Intuitively, KAArepresents the prior co-
variance and QAArepresents the covariance “explained” by the induc-
ing points.11 11For more details, refer to section 2
of “A unifying view of sparse ap-
proximate Gaussian process regression”
(Quinonero-Candela and Rasmussen,
2005 ).Computing the full covariance matrix is expensive. In the following,
we mention two approximations to the covariance of the training con-
ditional (and testing conditional).
Example 4.14: Subset of regressors
Thesubset of regressors (SoR) approximation is deﬁned as
qSoR(f|u).=N(f;KAUK→1
UUu,0),( 4.48a)
qSoR(fω|u).=N(fω;KωUK→1
UUu,0).( 4.48b)
Comparing to Equation ( 4.47), SoR simply forgets about all vari-
ance and covariance.→5.0→2.5 0.0 2.5 5.0→10123f(x)
→5.0→2.5 0.0 2.5 5.0
x→202
Figure4.15: Comparison of SoR (top)and FITC (bottom). The inducing pointsuare shown as vertical dotted red lines.The noise-free true function is shown inblack and the mean of the Gaussian pro-cess is shown in blue.Example4.15: Fully independent training conditionalThefully independent training conditional(FITC) approximation isdeﬁned asqFITC(f|u).=N(f;KAUK→1UUu, diag{KAA→QAA}),(4.49a)qFITC(fω|u).=N(fω;KωUK→1UUu, diag{Kωω→Qωω}).(4.49b)In contrast to SoR, FITC keeps track of the variances but forgetsabout the covariance.The computational cost for inducing point methods SoR and FITC isdominated by the cost of invertingKUU. Thus, the time complexity iscubic in the number of inducing points, but only linear in the numberof data points.


## Page 87

gaussian processes 77
Discussion
This chapter introduced Gaussian processes which leverage the function-
space view on linear regression to perform exact probabilistic inference
with ﬂexible, nonlinear models. A Gaussian process can be seen as a
non-parametric model since it can represent an inﬁnite-dimensional
parameter space. Instead, as we saw with the representer theorem,
such non-parametric (i.e., “function-space”) models are directly rep-
resented as functions of the data points. While this can make these
models more ﬂexible than a simple linear parametric model in input
space, it also makes them computationally expensive as the number of
data points grows. To this end, we discussed several ways of approxi-
mating Gaussian processes.
Nevertheless, for today’s internet-scale datasets, modern machine learn-
ing typically relies on large parametric models that learn features from
data. These models can effectively amortize the cost of inference dur-
ing training by encoding information into a ﬁxed set of parameters. In
the following chapters, we will start to explore approaches to approx-
imate probabilistic inference that can be applied to such models.
Problems
4.1.Feature space of Gaussian kernel.
1.Show that the univariate Gaussian kernel with length scale h=1
implicitly deﬁnes a feature space with basis vectors
φ(x)=
φ0(x)
φ1(x)
...
with φj(x)=1
j!e→x2
2xj.
Hint: Use the Taylor series expansion of the exponential function, ex=∑∞
j=0xj
j!.
2.Note that the vector φ(x)is∞-dimensional. Thus, taking the
function-space view allows us to perform regression in an inﬁnite-
dimensional feature space. What is the effective dimension when
regressing nunivariate data points with a Gaussian kernel?
4.2.Kernels on the circle.
Consider a dataset {(xi,yi)}n
i=1with labels yi↓Rand inputs xiwhich
lie on the unit circle S⇐R2. In particular, any element of Scan
be identiﬁed with points in R2of form (cos(ϱ), sin(ϱ))or with the
respective angles ϱ↓[0, 2π).
You now want to use GP regression to learn an unknown mapping
from StoRusing this dataset. Thus, you need a valid kernel k:


## Page 88

78 probabilistic artificial intelligence
S∝S↘R. First, we look at kernels kwhich can be understood as
analogous to the Gaussian kernel.
1.You think of the “extrinsic” kernel ke:S∝S↘Rdeﬁned by
ke(ϱ,ϱ′).=exp/parenleftigg
→≃x(ϱ)→x(ϱ′)≃2
2
2κ2
,
where x(ϱ).=(cos(ϱ), sin(ϱ)). Is kepositive semi-deﬁnite for all
values of κ>0?
2.Then, you think of an “intrinsic” kernel ki:S∝S↘Rdeﬁned by
ki(ϱ,ϱ′).=exp/parenleftbigg
→d(ϱ,ϱ′)2
2κ2/parenrightbigg
where d(ϱ,ϱ′).=min(|ϱ→ϱ′|,|ϱ→ϱ′→2π|,|ϱ→ϱ′+2π|)is the
standard arc length distance on the circle S.
You would now like to test whether this kernel is positive semi-
deﬁnite. We pick κ=2 and compute the kernel matrix Kfor the
points corresponding to the angles {0,π/2,π,3π/2}. This kernel
matrix Khas eigenvectors (1, 1, 1, 1 )and(→1, 1,→1, 1).
Now compute the eigenvalue corresponding to the eigenvector
(→1, 1,→1, 1).
3.Iskipositive semi-deﬁnite for κ=2?
4.A mathematician friend of yours suggests to you yet another kernel
for points on the circle S, called the heat kernel . The kernel itself has
a complicated expression but can be accurately approximated by
kh(ϱ,ϱ′).=1
Cκ/parenleftigg
1+L→1
∑
l=1e→κ2
2l22 cos(l(ϱ→ϱ′))
,
where L↓Ncontrols the quality of approximation and Cκ>0 is
a normalizing constant that depends only on κ.
Iskhis positive semi-deﬁnite for all values of κ>0 and L↓N?
Hint: Recall that cos(a→b)= cos(a)cos(b)+sin(a)sin(b).
4.3.A Kalman ﬁlter as a Gaussian process.
Next we will show that the Kalman ﬁlter from Example 3.4can be seen
as a Gaussian process. To this end, we deﬁne
f:N0↘R,t∀↘Xt.( 4.50)
Assuming that X0↑N(0,σ2
0)and Xt+1.=Xt+εtwith independent
noise εt↑N(0,σ2
x), show that
f↑G P(0,kKF)where ( 4.51)
kKF(t,t′).=σ2
0+σ2
xmin{t,t′}.( 4.52)
This particular kernel k(t,t′).=min{t,t′}but over the continuous-time
domain deﬁnes the Wiener process (also known as Brownian motion).


## Page 89

gaussian processes 79
4.4.Reproducing property and RKHS norm.
1.Derive the reproducing property.
Hint: Use k (x,x′)=▽k(x,·),k(x′,·)̸k.
2.Show that the RKHS norm ≃·≃kis a measure of smoothness by
proving that for any f↓Hk(X)andx,y↓Xit holds that
|f(x)→f(y)|↖≃f≃k≃k(x,·)→k(y,·)≃k.
4.5.Representer theorem.
With this, we can now derive the representer theorem ( 4.20).
Hint: Recall
1.the reproducing property f (x)= ▽f,k(x,·)̸kwith k (x,·)↓H k(X)
which holds for all f ↓Hk(X)andx↓Hk(X), and
2.that the norm after projection is smaller or equal the norm before projec-
tion.
Then decompose f into parallel and orthogonal components with respect to
span{k(x1,·),..., k(xn,·)}.
4.6.MAP estimate of Gaussian processes.
Let us denote by A={x1,..., xn}the set of training points. We will
now show that the MAP estimate of GP regression corresponds to
the solution of the regularized linear regression problem in the RKHS
stated in Equation ( 4.21):
ˆf.=arg min
f↓Hk(X)→logp(y1:n|x1:n,f)+1
2≃f≃2
k.
In the following, we abbreviate K=KAA. We will also assume that
the GP has a zero mean function.
1.Show that Equation ( 4.21) is equivalent to
ˆε.=arg min
ε↓Rn≃y→Kε≃2
2+λ≃ε≃2
K(4.53)
for some λ>0 which is also known as kernel ridge regression . De-
termine λ.
2.Show that Equation ( 4.53) with the λdetermined in ( 1) is equiva-
lent to the MAP estimate of GP regression.
Hint: Recall from Equation (4.6)that the MAP estimate at a point xωis
E[fω|xω,X,y]=k↙
xω,A(K+σ2
nI)→1y.
4.7.Gradient of the marginal likelihood.
In this exercise, we derive Equation ( 4.30).
Recall that we were considering a dataset (X,y)of noise-perturbed
evaluations yi=f(xi)+εiwhere εi↑N(0,σ2
n)and fis an unknown


## Page 90

80 probabilistic artificial intelligence
function. We make the hypothesis f↑G P (0,kθ)with a zero mean
function and the covariance function kθ. We are interested in ﬁnding
the hyperparameters θthat maximize the marginal likelihood p(y|X,θ).
1.Derive Equation ( 4.30).
Hint: You can use the following identities:
(a)for any invertible matrix M,
∂
∂ϱjM→1=→M→1∂M
∂ϱjM→1and (4.54)
(b)for any symmetric positive deﬁnite matrix M,
∂
∂ϱjlog det (M)=tr/parenleftigg
M→1∂M
∂ϱj
.( 4.55)
2.Assume now that the covariance function for the noisy targets (i.e.,
including the noise contribution) can be expressed as
ky,θ(x,x′)= ϱ0˜k(x,x′)
where ˜kis a valid kernel independent of ϱ0.12 12That is, Ky,θ(i,j)= ky,θ(xi,xj).
Show that∂
∂ϱ0logp(y|X,θ)= 0 admits a closed-form solution for
ϱ0which we denote by ϱω
0.
3.How should the optimal parameter ϱω
0be scaled if we scale the
labels yby a scalar s?
4.8.Uniform convergence of Fourier features.
In this exercise, we will prove Theorem 4.13.
Lets(x,x′).=z(x)↙z(x′)and f(x,x′).=s(x,x′)→k(x,x′). Observe that
both functions are shift invariant, and we will therefore denote them
as univariate functions with argument ∆∃x→x′↓M ∆. Notice that
our goal is to bound the probability of the event sup∆↓M∆|f(∆)|⇓ε.
1.Show that for all ∆↓M ∆,P(|f(∆)|⇓ε)↖2 exp
→mε2
4
.
What we have derived in ( 1) is known as a pointwise convergence guar-
antee. However, we are interested in bounding the uniform convergence
over the compact set M∆.
Our approach will be to “cover” the compact set M∆using Tballs of
radius rwhose centers we denote by {∆i}T
i=1. It can be shown that
this is possible for some T↖(4 diam (M)/r)d. It can furthermore be
shown that
¬i.|f(∆i)|<ε
2and≃→f(∆ω)≃2<ε
2r=↗sup
∆↓M∆|f(∆)|<ε
where ∆ω=arg max∆↓M∆≃→f(∆)≃2.


## Page 91

gaussian processes 81
2.Prove P/parenleftbig≃→f(∆ω)≃2⇓ε
2r/parenrightbig↖2rσp
ε2
.
Hint: Recall that the random Fourier feature approximation is unbiased,
i.e.,E[s(∆)]=k(∆).
3.Prove P/uniontextT
i=1|f(∆i)|⇓ε
2
↖2Texp
→mε2
16
.
4.Combine the results from ( 2) and ( 3) to prove Theorem 4.13.
Hint: You may use that
(a)αr→d+βr2=2βd
d+2α2
d+2for r =(α/β)1
d+2and
(b)σpdiam (M)
ε⇓1.
5.Show that for the Gaussian kernel ( 4.14),σ2
p=d
h2.
Hint: First show σ2
p=→tr(H∆k(0)).
4.9.Subset of regressors.
1.Using an SoR approximation, prove the following:
qSoR(f,fω)=N/parenleftigg
f
fω
;0,
QAA QAω
QωAQωω
(4.56)
qSoR(fω|y)=N(fω;QωA˜Q→1
AAy,Qωω→QωA˜Q→1
AAQAω) (4.57)
where ˜Qab.=Qab+σ2
n.
2.Derive that the resulting model is a degenerate Gaussian process
with covariance function
kSoR(x,x′).=k↙
x,UK→1
UUkx′,U.( 4.58)


## Page 92


