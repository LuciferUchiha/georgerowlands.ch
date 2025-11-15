---
title: Page Rank
type: docs
weight: 18
---

very earily in times of internet how to organize web, first try was human curated web directoreis like yahoo, dmoz, look smart.

second try was web search, google. Wanted to find relevant docs from a small and trsuted set but web is huge full of untrusted documents and random pages.

Challanges wo to trust? The idea for this is that trustowrthy pages point to each other. What is best answer for query? The idea for this is that pages that know about a subject are pointing to many other pages

### Flow Formulation

Idea is that links are votes, a page is more important if it has more links, do we look at in or out links? are all links equal?, links from important pages should count more => recursive question.

Simple recursive formulation is that each links vote is proportional to the importance of its source page. so if page j has important rj with n out link each link should get rj/n votes, js own importance rj is the sum of the votes of its inlinks. A vote from an important page is worth more this also fullfills the requirement of a page is important if it is pointed to by orther importange pages. this leads to n equations for n nodes and no constants so there is no unique solution. so to get a unique solution we add a constraint that all page ranks summed together equal 1. We can solve this by using gaussian elimation but this becomes very quickly very hard for large web graphs so we need a new formulation.

matrix formulation, instead of the equations we write a column stochastic adjacency matrix, meaning each column sums up to 1. so if page i has di out links and there is a connection from i to j then Mji is 1/di otherwise 0.

we then have the rank vector r which contains all the page rankings. so ri is the score for page i and all score summed = 1 as before. we can then write the flow equaiton as r=M*r. 

In other words the rank vector is an eigenvector of the adjacency matrix m with the eigenvalue 1. because the matrix is column stochstic the largest eigenvalue is 1.

Now to solve these flow equations we can use the power iteration method. Suppose there are N pages we initialize rank vector r0 with all values 1/N. we then iterate rt+1 = M*rt and stop when we no longer change the values much, i.e we converge. rt+1 -rt < epsilon. This is a method to find the dominant eigenvector i.e the vector corresponding to the largest eigenvalue which because it is column stochastic is 1.

r1 = M * r0
r2 = M*r1 = M(M*r1) = M^2 * r0
etc.

We can also interpret this using probailities on a random walk. the pagerank vector gives the stationary distribution for a random walk on a graph i.e

## Google Formulation

There is a problem with the above definition as it might not always converge. Google defined two types of problems that can occur in a network which stops the power iteration from converging.

dead ends i.e pages with no out links so a random walk can not leave so eventually importance leaks out to 0.

Spider traps all out links are within a group so random walk get stuck in a trap. these traps then slowly absorob all the importance.

The soultion to both of these issues is teleports. To solve spider traps at each step the random walker/surfer has two options with a prob of beta it will follow a link at random with 1-beta it will teleport to some random page, common values for beta are 0.8 or 0.9.

To solve the issue of dead ends we just adjust the adjacancy matrix by keeping the colum stochastic properity, so if a node has no out links it just becomes a unfirom distribution to all nodes.

This slightly changes the page rank quation as now there is the additional probability of a teleport. this leads to the google matrix which can still be solved with the power iteration method.

But what if we have 1 billion pages we can not really store this in main memory so we need to make use of the spasity of the matrix. we can do this by just storing the degree for each source node and its destination nodes. we can then store rold and M on the disk and rnew in memory and just calcualte row by row so we do not need to load the entire matrix into memory.

But what if we can not even fit the entire rnew in memory, then we can use the so called block based update algorithm where we then just do block by block.

In the above solution we are still scanning m and rold once for each block tho so we can do even better. which leads to block stripe update algorithm. Each strip contains only destination nodes corresponding to the blocks of rnew.

Page rank does however have its limitaitons. It only measures generic popularity of a page. The solution to this is topic specific page rank.
It only uses a single measure of importance. Other models of importance are used in the hub and authorities extension.
It is susceptible to link spam, i.e artificial link topographies to boost page rank to solve the there is the trust rank.

Topic specific page rank is useful if we do not just want a generic popularity but a popularity within a topic. this allows search quieries to be answwered based on intersts of the users for example the query jaguar is it the animal or car?

The idea here is a bias random walk so when teleporting instead of going to any page uniformly we only teleport to a topic-specific set. this gives pages closer to these topic specific pages a higher rank. to make this work all we need to do is update the teleportation part and give the ones with topic specific higher probability to be teleported to.

proximity on graphs i.e the relevance closeness or similarity of pages. shortest path is not good because slow and also does not take into account hubs. network flow is also not good as it does not punish long paths? dont know netowork flow lol.

Random Walk with Restart. makes no sense

What is web spam? any deliberate action to boost web pages position. spam are web pages that are the result of spamming. Constant war, first search engines considered number of times query word appeared, prominece of word position. Add keywords hidden thousand of times, or links with high topic ranking. Googles solution is to look at what others say about u for example looking at anchor text that is linked to ur page.

spam farming. Inaccessible pages, accessible pages, owned pages. To combat spam the trsut rank was developed. which is topic specific page rank but with a teleport set of trusted pages like .edu or .gov pages.

We create a seed set by hand which contains good trusted pages. We can then use this as the teleport set. Trust attenuation is the degreee of trust which is spread across the graph. trust splitting across out links

spam mass = what fraction of a pages rank comes from spam pages. page rank p rp, rp+ page rank with trusted pages only in teleport page rank from spam pages is then the difference and the spam mass is the proportion.

HITS, hypertextinduced topic selection is a meassure of importance. we have hubs and authorities. Hubs are course bulleting, authoritizes hold usefull information. Authority each page start with hub score 1 authorities collect there votes, hubs collect authority scores. so it doesn't just infetly increase it is normalized that all hubs equal 1.

## Other

PageRank can be extended to matrix factorization. All related to graph embeddings.

2 standford students start of google search engine. PageRank is the algorithm that google uses to rank web pages in their search engine results.

web as a graph, nodes are web pages, edges are hyperlinks. PageRank is a centrality measure. It is a measure of the importance of a node in a graph.

just think of pages as static. and hyperlinks are just navigational not transnational. i.e not to add or buy something etc.

Wikipedia perfect example, directed graph.

PageRank is a centrality measure. It is a measure of the importance of a node in a graph.

link analysis algorithm. PageRank, Personalized PageRank, Random Walk with Restart.

Think of links as votes, the more links to a page the more important it is, incoming links are more important than outgoing links because they are more difficult to get.

Are all incoming links equal? No, the importance of the page casting the vote matters. A page casting a vote is more important if it has more incoming links.
This becomes a recursive question and the idea of PageRank is to solve this recursively.

if page i has importance x_i and d_i outgoing links, then the importance of each link is x_i/d_i. The importance of page i, x_i is the sum of the importance 
of the pages casting votes for it, i.e incoming links. So x_i = sum_j x_j/d_j where j are the pages casting votes for page i.

importance=rank

A small example can be shown as a system of linear equations. you might think that you can then just solve the system of linear equations. But the problem is that
this is not scalable. The matrix is very large and sparse. So we need to use an iterative method.

stochastic adjacency matrix M, each column sums to 1. Called column stochastic matrix. Think of it as a probability distribution. How can this be interpreted?

Rank vector, each page has a component in the rank vector. The rank vector is a probability distribution, i.e sums to 1. 

The flow equation can then be written as r=M*r how can this be interpreted and an example.

Can think of it as a random web surfer. At each time step the surfer is at a page and then follows a link to another page. The probability of following a link is
the importance of the page casting the vote. The surfer is at page i with probability r_i. The probability of following a link from page i to page j is M_ij.

p(t) is the probability distribution of the surfer at time t. p(t+1) is the probability distribution of the surfer at time t+1. p(t+1)=M*p(t). This is the flow equation.

p(t) is a stationary distribution, i.e it has converged. p(t+1)=p(t)=p.

Therefore r is also a stationary distribution. r=M*r. This is the PageRank equation.

This then is related to the eigenvalue problem. M*r=r. r is the eigenvector where the eigenvalue is 1. This is the Perron-Frobenius theorem???? Because 1*r=M*r=r

If you repeatly multiply M by itself, you will converge to the eigenvector with eigenvalue 1. This is the power method. And can be thought of the random surfer 
repeatly following links. Till it converges to the stationary distribution. principal eigenvector, eigenvector with largest eigenvalue which is 1 here, why?

We first assign each page a the rank 1/n where n=number of pages => r0. Then we repeatly multiply M by the rank vector. This is the power method repeated till convergence.

In general takes about 50 iterations to converge. The power method is a very simple algorithm. It is also very scalable. It is also very easy to implement.

We have two problems, dead ends/sinks and spider traps. Dead ends are pages with no outgoing links. Spider traps are all outgoing links stay within a group of pages.

Sinks lead to importance "leaking" out of the graph, i.e dissapears, random walker falls off cliff. What does this mean, visualize.

Spider traps lead to importance "trapped" in the system. What does this mean, visualize. Eventually they absorb all the importance. random walker is stuck.

To solve this we can add beta which is a teleporting factor. This is the probability of following a link. 1-beta is the probability of teleporting to a random page.
normally beta is between 0.8 and 0.9. 

If a page is a dead end we can also make the teleporting factor 1, i.e if a columns sums to 0 we can make it sum to 1 
by settings all the values to 1/n. It could theoritcally teleport again to the same page. But this is very unlikely. These dead ends anyway violate column stochasticity.

The above can be nicely written as a sum or as the "Google" matrix. The Google matrix is a column stochastic matrix. It is a linear combination of the stochastic matrix
and the teleporting matrix. The teleporting matrix is a matrix with all values 1/n. 

## Personalized PageRank / Topic-Specific PageRank

Have users and items where edges are purchases. We want to recommend items to users. We can use PageRank to find similar items. We want to find items that are similar
to the items that the user has already purchased. We can use Personalized PageRank to do this. We want a recommender system.

Doesn't teleport uniformly to all pages, but only to pages that are similar to the pages that the user has already visited i.e a subset denoted by S. 
If S only contains one page then it is also called random walk with restart.

Can find out which pages are most similiar to the page in S.

Pages are similiar because the random walker often walks over the same pages. This is the intuition. It meassures similiarity over a lot of things like:
the entire graph, direction of links and degree of pages.

## Matrix Factorization and Graph Embeddings

Instead of saying nodes are similiar if they appear on the same random walk, we just say they are similiar if they are connected.

So we try to approximate the adjacency matrix with the product of two matrices, the embedding matrix and its transpose.


