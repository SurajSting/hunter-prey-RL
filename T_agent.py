import numpy as np

class FiniteHorizonTabularAgent(object):
	def __init__(self, nState, nAction, epLen, alpha0=1., mu0=0., tau0=1.,tau=1.):
		self.nState = nState
		self.nAction = nAction
		self.epLen = epLen

		# Parameters to accommodate probablistic modelling
		self.alpha0 = alpha0
		self.mu0 = mu0
		self.tau0 = tau0
		self.tau = tau

		# qVals and qMax tables
		self.qVals = {}
		self.qMax = {}

		#Now make the prior beliefs
		self.R_prior = {}
		self.P_prior = {}

		for state in range(nState):
			for action in range(nAction):
				# Initial reward knowledge - (0, 1)
				self.R_prior[state, action] = (self.mu0, self.tau0)
				
				# Initial transition dynamics knowledge 
				# either [0., 0., 0. ... nStates] I have no idea of the environment
				# or [1., 1., 1. ...nStates] Everything is possible
				self.P_prior[state, action] = (self.alpha0 * np.ones(self.nState, dtype=np.float32))

	def update_obs(self, oldState, action, reward, newState, pContinue, h):
		'''
	        Update the posterior belief based on one transition.
	        Args:
	            oldState - int
	            action - int
	            reward - double
	            newState - int
	            pContinue - 0/1
	            h - int - time within episode (not used)
	        Returns:
	            NULL - updates in place
		'''
		mu0, tau0 = self.R_prior[oldState, action]
		tau1 = tau0 + self.tau
		mu1 = (mu0 * tau0 + reward * self.tau) / tau1
		self.R_prior[oldState, action] = (mu1, tau1)

		if pContinue == 1:
			self.P_prior[oldState, action][newState] += 1


	def map_mdp(self):
		'''
		Returns the maximum a posteriori MDP from the posterior.
		Args:
			NULL
		Returns:
			R_hat - R_hat[s, a] is the MAP mean reward for (s,a)
			P_hat - P_hat[s, a] is the MAP transition vector for (s,a)
		'''
		R_hat = {}
		P_hat = {}

		for s in range(self.nState):
			for a in range(self.nAction):
				R_hat[s, a] = self.R_prior[s, a][0]
				P_hat[s, a] = self.P_prior[s, a] / np.sum(self.P_prior[s, a])
		return R_hat, P_hat

	def egreedy(self, state, timestep, epsilon = 0):
		Q = self.qVals[state, timestep]
		nAction = Q.size
		noise = np.random.rand()

		if noise < epsilon:
			action = np.random.choice(nAction)
		else:
			action = np.random.choice(np.where(Q == Q.max())[0])

		return action

	def pick_action(self, state, timestep):
		action = self.egreedy(state, timestep)
		return action

	def sample_mdp(self):
		R_samp = {}
		P_samp = {}

		for s in range(self.nState):
			for a in range(self.nAction):
				mu, tau = self.R_prior[s, a]
				R_samp[s, a] = mu + np.random.normal() * 1./np.sqrt(tau)
				P_samp[s, a] = np.random.dirichlet(self.P_prior[s, a])

		return R_samp, P_samp

	def map_mdp(self):
		'''
		        Returns the maximum a posteriori MDP from the posterior.
		        Args:
		            NULL
		        Returns:
		            R_hat - R_hat[s, a] is the MAP mean reward for (s,a)
		            P_hat - P_hat[s, a] is the MAP transition vector for (s,a)
		'''
		R_hat = {}
		P_hat = {}

		for s in range(self.nState):
			for a in range(self.nAction):
				R_hat[s, a] = self.R_prior[s, a][0]
				P_hat[s, a] = self.P_prior[s, a] / np.sum(self.P_prior[s, a])
		
		return R_hat, P_hat


	def compute_qVals(self, R, P):
		qVals = {}
		qMax = {}
		'''
			Computes the qValues dynamically timestep by timestep
			Compute the Q values for a given R, P estimates
        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions
        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        
		'''
		qMax[self.epLen] = np.zeros(self.nState, dtype=np.float32)

		for i in range(self.epLen):
			j = self.epLen - i - 1
			qMax[j] = np.zeros(self.nState, dtype=np.float32)

			for s in range(self.nState):
				qVals[s, j] = np.zeros(self.nAction, dtype=np.float32)

				for a in range(self.nAction):
					qVals[s, j][a] = R[s, a] + np.dot(P[s, a], qMax[j + 1])

					qMax[j][s] = np.max(qVals[s, j])

		return qVals, qMax


#-----------------------------------------------------------------------------
# PSRL
#-----------------------------------------------------------------------------

class PSRL(FiniteHorizonTabularAgent):
    '''
    Posterior Sampling for Reinforcement Learning
    '''

    def update_policy(self):
        '''
        Sample a single MDP from the posterior and solve for optimal Q values.
        Works in place with no arguments.
        '''
        # Sample the MDP
        R_samp, P_samp = self.sample_mdp()

        # Solve the MDP via value iteration
        qVals, qMax = self.compute_qVals(R_samp, P_samp)

        # Update the Agent's Q-values
        self.qVals = qVals
        self.qMax = qMax



#-----------------------------------------------------------------------------
# Epsilon Greedy
#-----------------------------------------------------------------------------


class EpsilonGreedy(FiniteHorizonTabularAgent):
	''' Epsilon greedy agent'''
	def __init__(self, nState, nAction, epLen, epsilon=0.1):
		'''
		As per the tabular learner, the prior effect-->0
		Args:
			epsilon - double - probablity of random action
		'''
		super(EpsilonGreedy, self).__init__(nState, nAction, epLen, alpha0=0.0001, tau0=0.0001)
		self.epsilon = epsilon

	def update_policy(self):
		# Output MAP estimate MDP
		R_hat, P_hat = self.map_mdp()

		#Solve the MDP via value iteration
		qVals, qMax = self.compute_qVals(R_hat, P_hat)

		self.qVals = qVals
		self.qMax = qMax

	def egreedy(self, state, timestep, epsilon):
		'''
        Select action according to a greedy policy
        Args:
            state - int - current state
            timestep - int - timestep *within* episode
        Returns:
            action - int
        '''
		Q = self.qVals[state, timestep]
		nAction = Q.size
		noise = np.random.rand()

		if noise < epsilon:
			action = np.random.choice(nAction)
		else:
			action = np.random.choice(np.where(Q == Q.max())[0])

		return action

	def pick_action(self, state, timestep):
		action = self.egreedy(state, timestep, self.epsilon)
		return action