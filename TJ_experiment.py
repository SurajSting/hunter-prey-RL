"""
	* These experiments consider a "Synchronous game", i.e. Tom and Jerry both choose their moves before the game board is updated with their moves.
	* The qVals considers the reward value growing in State-Action through time, dynamically, to make use of precomputed qVals in earlier stages in the game.
	* Both Tom and Jerry, learn their strategies individually based on the qVals learnt.
	* The transition dynamics of the environment assumes a deterministic setting, but allows for further expanding towards a more stochastic setting.

	Experiments:
	* Episode length/ Episodes relation - 2D random walks -> intersection time
	- need to change episode length for nondeterministic world 
"""

import sys
import os
import time
import copy
import itertools

#RL IMPORTS
import TJ_environment as env
import T_agent as tom
import J_agent as jerry

#PLOT IMPORTS
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np

def pos_to_state(Position, rows, cols):
	row, col = (Position[0], Position[1])
	state = cols * row + col
	return state

def convert_actions(action):
	if action == 0:
		action = 5
	elif action == 1:
		action = 6
	elif action == 2:
		action = 7
	elif action == 3:
		action = 8
	elif action == 4:
		action = 9
	return action

def run_finite_tabular_experiment(level, Tom, Jerry):
	# Initialize the game environment
	game = env.make_game(level)

	# Extracting Features
	rows = game.rows
	cols = game.cols
	nStates = rows * cols
	nTomActions = 5 			# North, East, West, South
	nJerryActions = 5
	epLen = 56 #56

	# Initialize the agents

	#Tom = tom.EpsilonGreedy(nStates, nTomActions, epLen, epsilon=1)
	#Jerry = jerry.EpsilonGreedy(nStates, nJerryActions, epLen, epsilon=1)

	#Tom = tom.EpsilonGreedy(nStates, nTomActions, epLen, epsilon=0.5)
	#Jerry = jerry.EpsilonGreedy(nStates, nJerryActions, epLen, epsilon=0.5)

	#Tom = tom.EpsilonGreedy(nStates, nTomActions, epLen, epsilon=0.1)
	#Jerry = jerry.EpsilonGreedy(nStates, nJerryActions, epLen, epsilon=0.1)

	#Tom = tom.PSRL(nStates, nTomActions, epLen)
	#Jerry = jerry.PSRL(nStates, nTomActions, epLen)

	nEps = 10005#3000
	TomScore = 0
	JerryScore = 0

	collisionTimestep = []
	horizon = []
	tomCollisionPosition = []
	jerryCollisionPosition = []
	start = time.time()

	for ep in range(1, nEps+2):
		# Resetting the environment
		game = env.make_game(level)
		first_obs, first_reward, first_discount = game.its_showtime()

		# Update Policy
		Tom.update_policy()
		Jerry.update_policy()

		pContinue = 1
		clear = lambda: os.system('clear')
		clear()
		# Printing to Terminal
		#for row in first_obs.board: print(row.tostring().decode('ascii'))

		while pContinue > 0:
			# Find the timestep
			h = game.the_plot.frame

			# Find where Tom and Jerry are on the grid to associate with Q-map
			Tpos = game.things['T'].position
			Ts = pos_to_state(Tpos, rows, cols)
			Jpos = game.things['J'].position
			Js = pos_to_state(Jpos, rows, cols)

			# Epsilon Greedy Action selection
			tomAction = Tom.pick_action(Ts, h)
			Ta = tomAction

			jerryAction = Jerry.pick_action(Js, h)
			# Mapping jerry's action such that: 0->5, 1->6, 2->7, 3->8, 4->9 
			Ja = convert_actions(jerryAction)

			# Applying the actions in the deterministic gridworld and printing the board
			second_obs, second_reward, second_discount = game.play([Ta, Ja])

			# Find new Tom and Jerry positions after playing their move
			newTpos = game.things['T'].position
			newTs = pos_to_state(newTpos, rows, cols)
			newJpos = game.things['J'].position
			newJs = pos_to_state(newJpos, rows, cols)

			# TERMINATE EPISODE CONDITION
			# If Tom and Jerry run out of steps for the episode
			if game.the_plot.frame == epLen:
				second_reward = [-1, 1]
				tomReward = -1
				jerryReward = 1
				JerryScore+=1
				pContinue = 0
				game.the_plot.terminate_episode()
			# In the case of Tom finding Jerry
			elif second_reward is not None:
				tomReward = second_reward[0]
				jerryReward = second_reward[1]
				TomScore+=1
				pContinue = 0
				game.the_plot.terminate_episode()

			if second_reward is None:
				tomReward = 0.0
				jerryReward = 0.0

			Tom.update_obs(Ts, tomAction, tomReward, newTs, pContinue, h)
			Jerry.update_obs(Js, jerryAction, jerryReward, newJs, pContinue, h)

			clear()
			print('Episode:', ep)
			print("Timestep: ", h)
			#print("Tom: ", TomScore, "Jerry: ", JerryScore)
			Tom_WLratio = round(TomScore / ep, 2)
			#print("RATIO: ", Tom_WLratio)
			#for row in second_obs.board: print(row.tostring().decode('ascii'))

	end = time.time()
	elapsed = round(end - start, 2)

	return Tom_WLratio, elapsed

def main(argv=()):
	nStates = 100
	nTomActions = 5
	nJerryActions = 5
	epLen = 56

	T1 = tom.EpsilonGreedy(nStates, nTomActions, epLen, epsilon=1)
	J1 = jerry.EpsilonGreedy(nStates, nJerryActions, epLen, epsilon=1)

	T2 = tom.EpsilonGreedy(nStates, nTomActions, epLen, epsilon=0.5)
	J2 = jerry.EpsilonGreedy(nStates, nJerryActions, epLen, epsilon=0.5)

	T3 = tom.EpsilonGreedy(nStates, nTomActions, epLen, epsilon=0.1)
	J3 = jerry.EpsilonGreedy(nStates, nJerryActions, epLen, epsilon=0.1)

	T4 = tom.PSRL(nStates, nTomActions, epLen)
	J4 = jerry.PSRL(nStates, nTomActions, epLen)

	T = [T1, T2, T3, T4]
	J = [J1, J2, J3, J4]

	TJ = list(itertools.product(T, J))
	test = 1
	f = open("wlTable.tex", "a")
	for T,J in TJ:
		print("Test", test)
		ratio, time = run_finite_tabular_experiment(0, T, J)
		f.write(str(ratio))
		f.write(" & ")
		f.write(str(time))	
		if(test%4 != 0):
			f.write(" & ")
		if(test%4 == 0):
			f.write(" \\\\ ")
			f.write("\n")
		test+=1
	f.close()


if __name__ == '__main__':
	main(sys.argv)
