# Task Planning with Pretrained Prior

Consider the problem of determining the next action of an agent trying to accomplish a goal, $g$, the current and previous observations $o_{1:t}$, as well as the previous actions $a_{1:t-1}$.

$$p(a_t|g, o_{1:t}, a_{1:t-1})$$

We can factor out a prior by formulating using Bayes Rule:

$$p(g, o_{1:t-1}|a_{1:t-1}, a_t, o_t)p(a_t|a_{1:t-1}, o_t)$$

where the first term is the likelihood and second prior, which can be a language model if $o_t$ is encoded using text. Observe the likelihood is discrimative i.e. it operates on binary labels/outputs probability that denote whether $a_t$ is the correct action. Assume that all goals come can unified relatively well under the same prior, and the prior training data can be aggregated from multiple goals. We can sample actions as follows: sample the top $k$ choices from the prior and reweight them using the likelihood.

In our case, we use [BabyAI](https://openreview.net/forum?id=rJeXCo0cYX) as the environment. The prior is GPT2 fine-tuned on many goals, where $a_{1:t-1}, o_t$ are represented as text. The likelihood is a classifier that involves BabyAI's 7x7 grid, the aggregated actions, observations, and the goal. We generate negative samples by changing the action of the last element of a goal sequence, which can be viewed as analogous to teacher forcing.

The prior achieves decent accuracy (~70%) and the likelihoods achieve very high accuracy (~93%) with low data (few hundred data points) only on the given goal. 

View detailed experiment logs [here](https://www.dropbox.com/sh/toemdplohy5239o/AAAjtiFgDbrUjc503eQG5WPta?dl=0).
