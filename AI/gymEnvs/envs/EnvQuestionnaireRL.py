import os
import gym
import sys
import numpy as np
from gym import spaces
import copy

class EnvQuestionnaireRL(gym.Env):
	def __init__(self, args):

		# TODO: how do we deal with this ?
		# Create two tracers : one symbolic used for detecting path constraints etc, and another one less heavy used only for tracing and scoring purpose
		self.tracer = RiverTracer(symbolized=True, architecture=args.architecture, maxInputSize=args.maxLen,
								  targetAddressToReach=args.targetAddress)
		self.stateful = args.stateful

		# All possible observations from last run
		# --------------------
		# The map hash
		self.obs_map = np.zeros(shape=(OBS_MAP_EDGESIZE, OBS_MAP_EDGESIZE), dtype=np.int32) # Number of time each hashed block was found
		# The last path through the program - a list of basic block addresses from the program evaluation
		self.obs_path = None
		# Last run as above but in format {basic blocks : how many times}
		self.obs_path_stats = None
		# TODO: take it from the other side
		self.obs_embedding = None
		self.args = args
		#--------------------

		# Load the binary info into the given list of tracers. We do this strage API to load only once the binary...
		RiverTracer.loadBinary([self.tracer], args.binaryPath, args.entryfuncName)
		if args.outputType == "textual":
			outputStats = RiverStatsTextual()

		self.observation_space = spaces.Dict({'map' : spaces.Box(low=0, high=sys.maxsize, shape=(OBS_MAP_EDGESIZE, OBS_MAP_EDGESIZE)), #if args.obs_map else None,
								  				'path' : spaces.Box(low = 0, high=MAX_BBLOCK_HASHCODE, shape=(OBS_MAX_PATH_LEN,)), #if args.obs_path else None,

											   # blocks and their visit count during last run
											  'obs_path_stats' :  spaces.Tuple((spaces.Box(low = 0, high=MAX_BBLOCK_HASHCODE, shape=(OBS_MAX_PATH_LEN,)),
																				spaces.Box(low = 0, high=np.inf, shape=(OBS_MAX_PATH_LEN,)))),  #if args.obs_path_stats else None,
												'obs_embedding' : spaces.Box(low=0, high=256, shape=(OBS_PATH_EMBEDDING_SIZE,))  #if args.obs_embedding else None
								  				})

		# Action Index, Parameters
		self.action_space = spaces.Tuple((spaces.Discrete(RiverUtils.Input.getNumActionFunctors()),
										 spaces.Dict({'isSymbolic' : spaces.Discrete(2)})))

	# You can override this to fill observation and still reuse step and reset function without overriding them
	# in your experiments
	def fill_observation(self):
		obs = {}
		obs['map'] = self.obs_map
		obs['path'] = self.obs_path
		obs['obs_path_stats'] = self.obs_path_stats
		obs['obs_embedding'] = self.obs_embedding
		return obs

	# Reset the state and returns an initial observation
	def reset(self):
		self.input : RiverUtils.Input = RiverUtils.Input() 	# Array of bytes
		self.input.usePlainBuffer = True
		self.input.buffer = [0] * self.args.maxLen # np.zeros(shape=(self.args.maxLen,), dtype=np.byte)
		self.tracer.ResetMem()
		self.tracer.resetPersistentState()

		obs = self.fill_observation()
		return obs

	# Take one action and return : observation, reward, done or not, info
	def step(self, action):

		obs = self.fill_observation()
		done = crashed
		info = {'lastPathConstraints' : lastPathConstraints,
				'allBBsFound' : allBBsInThisRun}
		return (obs, numNewBlocks, crashed, done, info)

