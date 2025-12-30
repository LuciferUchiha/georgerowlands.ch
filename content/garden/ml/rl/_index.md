---
title: Reinforcement Learning
type: docs
weight: 25
---

When learning from data, we are often used to the supervised learning setting where the world appears calm and well behaved: someone has already collected a dataset of input–output pairs, and our only task is to learn a mapping that imitates these known answers. But this picture of learning is deeply limited as we only have a snapshot of a static world. On the other hand, the idea of reinforcement learning is to **learn through interaction rather than instruction**. Instead of being told what the correct answer is, an agent has to discover good behaviour by "fucking around and finding out", i.e performing actions and observing the consequences, and gradually building an understanding of how its actions shape the future. In this sense, reinforcement learning is closer to the way animals and humans learn by acting in the world and learning from experience.

{{< figure 
    src="/images/ml/rlCycle.svg" 
    caption="The reinforcement learning cycle."
    alt="The reinforcement learning cycle."
    width="400"
>}}

For example imagine for a moment that you are trying to learn to ride a bicycle. Nobody hands you a table of “situations” paired with “correct actions”. Instead, you climb onto the bike, wobble, fall, climb again, and gradually figure out which movements keep you upright. The feedback the world gives you is not a label like “correct steering angle is 17 degrees”, but something far more primitive and indirect: you remain balanced (good), or you fall (bad). Reinforcement learning formalizes exactly these sorts of learning signals. Instead of labelled answers, the agent (you) receives **rewards**. Rewards do not tell you what the right action was but simply indicate how good or bad the outcome was. Your goal is then to learn to take actions that maximize the total reward it receives over time or down the road (pun intended).

{{< figure 
    src="/images/ml/rlBicycle.png"
    caption="Learning to ride a bike through trial and error."
    alt="Learning to ride a bike through trial and error."
>}}

Another key difference from supervised learning is that in reinforcement learning, **the data the agent sees is no longer independent and identically distributed**, because each action the agent takes changes the world in which future data will be generated. If you steer left on a bicycle now, the physical situation you will be in a second later depends on that choice. If you make a poor move in a game of chess, the next positions you see will reflect the consequences of that mistake. A move in Chess might seem normal now but lead to a checkmate twenty turns later. This is known as the **credit assignment problem**, determining how much each earlier action contributed to a reward that arrives much later. As the agent improves and starts behaving differently, it begins exploring different parts of the environment, which means it will see different rewards, different transitions, and different future possibilities and tries to exploit what it has learned so far to maximize its cumulative reward. This tight coupling between behaviour and data is exactly what makes reinforcement learning powerful as by **experiencing/sampling trajectories of interaction**, the agent gradually uncovers the structure of the task. But it also introduces new challenges. Because actions influence what the agent sees later, it must constantly balance two competing objectives: Trying things it already believes are good, and experimenting with actions that might lead to better strategies it has not yet discovered. This is the so called **exploration–exploitation dilemma**, which lies at the heart of reinforcement learning.

Although these ideas may seem abstract at first and were initially also heavily doubted by the research community, they underpin some of the most impressive achievements of modern AI. Systems like **AlphaGo** and **AlphaZero** by Google Deepmind were not trained on datasets of "correct" moves. Instead, they learned by interacting with the environment by playing games, receiving sparse signals such as win or lose, and gradually learning strategies. These models began with no human knowledge, no opening book, no endgame heuristics, only the rules of each game. By repeatedly playing against other oponent or against itself and learning from the outcomes, it discovered tactics, structures, and strategic principles that took human experts centuries to develop and even surpassed the best human players. 

{{< figure 
    src="/images/ml/alphaGo.jpg"
    caption="AlphaGo, developed by DeepMind, made history in 2016 by defeating the world champion Go player Lee Sedol."
    alt="AlphaGo, developed by DeepMind, made history in 2016 by defeating the world champion Go player Lee Sedol."
    width="600"
>}}
