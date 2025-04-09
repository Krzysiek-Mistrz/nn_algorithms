import tensorflow_probability.distributions as tfd
import tensorflow as tf

#state propabilities initials
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])
#transition propabilities
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5], [0.2, 0.8]])
#mean and standard devs for each of the states
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

#steps is the num of days we want to predict similiar to epochs
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

mean = model.mean()

with tf.compat.v1.Session() as sess:  
  print(mean.numpy())