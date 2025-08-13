import numpy as np
class ThompsonSampling:
	def __init__(self,n_actions,decay=0.99,initial_alpha=5,initial_beta=5):
		self.n_it = 0
		self.n_actions = n_actions
		self.k = 1
		self.action = None
		# Use more conservative initial parameters to encourage exploration
		self.alpha_beta = [[initial_alpha,initial_beta] for _ in range(self.n_actions)]
		self.decay = decay  # Higher decay factor to preserve more historical information

	def select_action(self):
		samples = [0] * self.n_actions
		for a in range(self.n_actions):
			for _ in range(self.k):
				samples[a]  += np.random.beta(self.alpha_beta[a][0],self.alpha_beta[a][1])
		self.action = np.argmax(samples)
		return self.action
	
	def reward(self,reward):
		if reward:
			self.alpha_beta[self.action][0] = 1 + self.alpha_beta[self.action][0] * self.decay
			self.alpha_beta[self.action][1] = 0 + self.alpha_beta[self.action][1] * self.decay
		else:
			self.alpha_beta[self.action][0] = 0 + self.alpha_beta[self.action][0] * self.decay
			self.alpha_beta[self.action][1] = 1 + self.alpha_beta[self.action][1] * self.decay
		self.n_it += 1
