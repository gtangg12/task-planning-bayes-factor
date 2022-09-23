# Task Planning for Large Action Domains via Bayesian Factorization

Consider the problem of determining the next action of an agent trying to accomplish a goal, $g$, the current and previous observations $o_{1:t}$, as well as the previous actions $a_{1:t-1}$.

$$p(a_t|g, o_{1:t}, a_{1:t-1})$$

A naive solution is to train a model to directly predict the action using the above formulation. For instance, a model with a feature backbone and fine-tuned heads for each goal. However, there are several issues if the action domain is not discrete/very large e.g. position of a robot arm in space.

- The above formulation is basically generative since the action domain is large, which is usually at least as hard as the discriminative alternative (given $a_t$ and context, determine if $a_t$ is the correct next action; discussed more in depth below). 

- In the generative case, representation of the output is not consistent between heads, leading to more engineering overhead. For example, a position in 3D space may be represented as a heat map while torque of an arm joint may be representer as a scalar.

- For certain output representations, the model will potentially need to work with the entire action domain, requiring much more training data per head. 

We can instead formulate using Bayes Rule:

$$p(g, o_{1:t-1}|a_{1:t-1}, a_t, o_t)p(a_t|a_{1:t-1}, o_t)$$

where the first term is the likelihood and second prior. Observe the likelihood is discrimative i.e. it operates on binary labels/outputs probability that denote whether $a_t$ is the correct action. Assume that all goals come can unified relatively well under the same prior. We can sample actions as follows: sample the top $k$ choices from the prior and reweight them using the likelihood.

In our case, we use [BabyAI](https://openreview.net/forum?id=rJeXCo0cYX) as the environment. The prior is GPT2 fine-tuned on many goals, where $a_{1:t-1}, o_t$ are represented as text. The likelihood is a classifier that involves BabyAI's 7x7 grid, the aggregated actions, observations, and the goal. We generate negative samples by changing the action of the last element of a goal sequence, which can be viewed as analogous to teacher forcing.

The prior achieves decent accuracy (~70%) and the likelihoods achieve very high accuracy (~93%) with low data (few hundred data points) only on the given goal. 

View detailed experiment logs [here](https://www.dropbox.com/sh/toemdplohy5239o/AAAjtiFgDbrUjc503eQG5WPta?dl=0).
