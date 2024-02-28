# TODO:
'''
-> DEBUG the critic with a fixed policy. Implement a deterministic way to avoid exploration at all at a fixed policy
  (one dummy example for critic is to always produce the SAME probability space sorted increasingly, put seeds initial in all random variables).
-> neg loss, too much ? are we too negative in the advatange function?
-> Check standardize returns. It is better true or false ?


# TODO13 :
-> should AgentRLEpisodeInfo be unified with State env state ? Put everything inside a DS
-> tf_functions stuff...


1. Should add current clip id info when selecting a model ???
2. Put the question ids instead of indices - map output
2. Deal with the loss function
3. Add dropout features where i wrote TODO dropout

4. Add attention mechanism


'''
import tensorflow as tf
import numpy as np
from typing import Set, List, Dict
from tqdm import tqdm

from AI.AgentPathfinding import AgentPathfinding
import AI.Utils_replay as Utils_replay
from AI.DataDefinitions import *
from AI.AgentAbstract import *
from typing import Set, List, Dict, Tuple
import random
import datetime

# Ideas:
'''
1. Agent must correlate between the sequence of questions, clips and response given so far
in relation with the cluster scores and the agent real type. The agent must then identify the right questions to ask to approach to the real cluster
(show the input used, clips+question+raw deviation +  probability of clusters after each question)
We don't use the raw question's hidden factors such as attributes and categories weights because those are supposed to be learned from queID and 
probabilities, and deviations. These are impacted directly by the attributes and cateogries weights for each of the question.

2. The output neurons specify only the probabilities of the index of the questions/clips. THey are then remapped to the set of real
question/clips ids. Those are then filtered and the first maximum clip/question probability available in the current context is being used.
While the model could learn which is the correct set of questions/clip available at each context, it is easier to to a hard check for correctness purposes.


E.g. for questions. let MT = max questions count of every team in our dataset.
Then, the model output are indices between [0.....MT-1] on each team selection. When we are on a certain clip T, there are two lists:
(A). One representing all questions in T, QAll(T) organized in the validty tree
(B). One representing all questions in T which are valid in the current context, i.e. given the questions already asked and previous question, which
of them are available now ? QContext(T) inclus in QAll(T).
E.g. Consider that QAll(T) = {11, 27, 31, 45}. MT = 7. QContext = { 11,31}. The actor will produce action probability for each of the questions
in range [0..6]. In the First phase, only the first 4 indices [0...3] are valid, being mapped to QAll(T) indices. From these indices, we select only indices
{0, 2} since they correspond to QContext, valid questions in this context

3. Talk about the two rewards components. So we basically have a reward that is always negative, sustaining the agent to go closer to the real cluster
ANd another one which could be positive when the agent moves towards the desired cluster index
Talk how we average the rewards inside a question with gamma 
4. Trajectories are selection of both clips and questions according to the given rules

5. Explain how simulation of user reponses work.
    user cluster index -> connect to [cluster, attributes mean/dev gaussians] to sample values -> 

TODO:
3. The survey demo should give feedback back that agent responded to a question . In effect, the agent should compute the reward annd add the experience to a local queue
4. Multiple survey demos must be run in parallel using different agent with different cluster types behind. 
         A training supervisor must take the local batches of experiences gathered from each agent locally in step 1, and add them in a global history priority queue.
This can be done at each step since the number of survey questions is small, no need to use frame skip method.

Step 2 is done to remove correlations that would lead model weights for one cluster type only.



'''
# TODO12: fix the constants
REWARD_FACTOR_OVERALLDIST = 0.8
REWARD_FACTOR_VELOCITY = 0.2
MAX_TRAINING_EPISODES = 10000

# Size of the embeddings
GAMMA = 0.99
STANDARDIZE_RETURNS = True #False
QUESTION_PAD_VALUE = -1 # Invalid question ID
UNKNOWN_REWARD = -9999999.0


# State model that transforms an observation to a numerical representation
class AgentRLStateEmbeddingModel(tf.keras.Model):
    def __init__(self, maxQuestionsCountInsideAClip : int, listOfAllClipsIds : int, listOfAllQuestionsIds : int):
        super(AgentRLStateEmbeddingModel, self).__init__()

        self.maxNumQuestionsInsideAnyClip = maxQuestionsCountInsideAClip
        self.outFinalSize = 8 # Size of the final embedding
        self.outTimeStepSize = 8 # Size of each step input
        self.outValidQuestionsEmbSize = 8
        self.QUESTION_ID_EMBSIZE = 4
        self.CLIP_ID_EMBSIZE = 4

        self.clipsLookup = tf.keras.layers.experimental.preprocessing.IntegerLookup()
        self.clipsLookup.adapt(listOfAllClipsIds)
        clipsVocab = self.clipsLookup.get_vocabulary()
        self.clipsLookup_inverse = tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=clipsVocab, invert=True)

        self.questionsLookup = tf.keras.layers.experimental.preprocessing.IntegerLookup()
        self.questionsLookup.adapt(listOfAllQuestionsIds)
        questionsVocab = self.questionsLookup.get_vocabulary()
        self.questionsLookup_inverse = tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=questionsVocab, invert=True)

        self.embQuestionId = tf.keras.layers.Embedding(len(questionsVocab), self.QUESTION_ID_EMBSIZE)
        self.embClipId = tf.keras.layers.Embedding(len(clipsVocab), self.CLIP_ID_EMBSIZE)
        self.questionFC = tf.keras.layers.Dense(self.outTimeStepSize)
        self.gruHistory = tf.keras.layers.GRU(self.outFinalSize)

        # These will be used when selecting a question
        self.FCSetOfValidQuestionsEmb = tf.keras.layers.Dense(self.outValidQuestionsEmbSize) # Concatenate emb spaces then convert to a dense of 8 neurons
        self.history_and_currentClipEmb_and_SetOfValidQuestionsEmb = tf.keras.layers.Dense(self.outFinalSize) # combination of history + currenthememb + emb of valid questions


    # input_data is a tuple of :
    #   A. sequence of clipids and questions + their devscore so far
    # input_data[0] = [None, [(clipId_0, question_0, devscore), (clipId_0, question_1),... .]]
    #  (clipId_1, question_0), ....]
    #  B. Scores (probabilities) of agent being part of each cluster type,  at each time step :
    #  input_data[1] = [None, [scores_time0:[prob cluster_0, cluster_1], scores_time1:[...]]]
    #  Basically the len of A is equal to B

    # GRU 1 for history mapping
    # [ timestep_i :  [clipIdEmb, questionIdEmb, devScore for this question, probs after each question] => 32 neurons].
    # Input of 32 units, output of 32 units]


    # Build first an embeeding for each time step, then call the GRU layer

    # If question is asked, more things are added. checlk constructor
    def call(self, input_data, training : bool):
        clusterScoresAfterQuestion  = input_data[0][1:] # Because the first cluster scores is the initialization, all equal probability
        sequenceOfQuestions = input_data[1]
        shouldSelectClip = input_data[2]
        listOfValidQuestionIds = input_data[3]
        assert (listOfValidQuestionIds == None and shouldSelectClip == True) or isinstance(listOfValidQuestionIds, List)

        # Same timesteps size ?
        assert len(sequenceOfQuestions) == len(clusterScoresAfterQuestion)
        # Same timesteps per question
        # assert len(sequenceOfQuestions[0] == len(clustersScoresAfterQuestions[0]))

        questionAndRespHist_res = None
        numTimesteps = len(sequenceOfQuestions)
        if numTimesteps == 0:
            questionAndRespHist_res = tf.constant([[0.0]*self.outFinalSize])
        else:
            # Step 0: Build input features data for GRU
            timestepInputData = []
            lastClipIdEmbUsed = None
            for iTimeStep in range(numTimesteps):
                timestep_scoresAfterQuestion = tf.reshape(input_data[0][iTimeStep], shape=(1,-1))
                timestep_questionData = input_data[1][iTimeStep]

                clipId, questionId, devScore = timestep_questionData
                clipId = np.array([clipId])
                questionId = np.array([questionId])
                devScore = tf.reshape(np.array([devScore]), shape=(1,-1))

                clipId_lookup = self.clipsLookup(clipId)
                clipIdEmb = self.embClipId(clipId_lookup)
                lastClipIdEmbUsed = clipIdEmb

                questionId_lookup = self.questionsLookup(questionId)
                questionIdEmb = self.embQuestionId(questionId_lookup)

                questionAndResp_interm = tf.concat([clipIdEmb, questionIdEmb, devScore, timestep_scoresAfterQuestion], axis=-1)
                questionAndRespHist_res = self.questionFC(questionAndResp_interm)
                # TODO: dropout here
                timestepInputData.append(questionAndRespHist_res)

            timestepInputData = tf.stack(timestepInputData, axis=1)
            inputDataShape = timestepInputData.get_shape().as_list()
            assert inputDataShape[0] == 1 and inputDataShape[1] == numTimesteps

            # Step 1: Apply the GRU for history mapping.
            questionAndRespHist_res = self.gruHistory(timestepInputData)

        final_output = questionAndRespHist_res

        if shouldSelectClip is False:
            # Step 2.1 Emb the current clip id
            assert(lastClipIdEmbUsed)
            currentClipEmb = lastClipIdEmbUsed

            # Step 2.2 Emb the valid questions ids in this context
            # 2.2.1 Take all the questions ids in an array first
            assert len(listOfValidQuestionIds) <= self.maxNumQuestionsInsideAnyClip
            validQuestionsIds = [QUESTION_PAD_VALUE] * self.maxNumQuestionsInsideAnyClip
            for index, qId in enumerate(listOfValidQuestionIds):
                validQuestionsIds[index] = qId

            # 2.2.2. fix the emb for each possible one using the pad values emb too
            for index in range(self.maxNumQuestionsInsideAnyClip):
                validQuestionId_lookup = validQuestionsIds[index]
                validQuestionsIds[index] = self.embQuestionId(validQuestionId_lookup)

            # 2.2.3 concatenate the emb above
            concat_validQuestions_emb = tf.concat(validQuestionsIds, axis=-1)
            final_validQuestions_emb = self.FCSetOfValidQuestionsEmb(concat_validQuestions_emb)

            # 2.2.4 concatenate the history + current clip emb + final valid questions emb
            concat_context = tf.concat([questionAndRespHist_res, currentClipEmb, final_validQuestions_emb])
            final_output = self.history_and_currentClipEmb_and_SetOfValidQuestionsEmb(concat_context)

        return final_output

# A model that given a raw observation, select as requested either a clip or question
class AgentActorCriticModel(tf.keras.Model):
    def __init__(self, datastore):#, setOfAllAvailableClipIds : Set[int], setOfAllAvailableQuestionsIds : Set[int]):
        super(AgentActorCriticModel, self).__init__()

        self.datastore = datastore
        self.maxQuestionsCountInsideAClip = 0
        self.listOfAllAvailableClipsIds = [int(key) for key in self.datastore.questions_byClip.keys()]
        self.listOfAllAvailableQuestionsIds = [int(key) for key in self.datastore.questions_byId.keys()]

        for clipId in self.listOfAllAvailableClipsIds:
            questionsForClip : List[QuestionForClip] = self.datastore.questions_byClip[clipId]
            self.maxQuestionsCountInsideAClip = max(self.maxQuestionsCountInsideAClip, len(questionsForClip))

        self.embeddingStateModel = AgentRLStateEmbeddingModel(maxQuestionsCountInsideAClip = self.maxQuestionsCountInsideAClip,
                                                              listOfAllClipsIds = self.listOfAllAvailableClipsIds,
                                                              listOfAllQuestionsIds = self.listOfAllAvailableQuestionsIds)
        self.actor_clips = tf.keras.layers.Dense(len(self.listOfAllAvailableClipsIds))
        self.actor_questions = tf.keras.layers.Dense(self.maxQuestionsCountInsideAClip)
        self.critic_clips = tf.keras.layers.Dense(1)
        self.critic_questions = tf.keras.layers.Dense(1)

    # input_dat is a tuple of a raw state as defined in the embeddingStateModel defined above
    # On top of that, for now we add a simple + [Head1: Softmax(FC(numClips)) Head2: Softmax(FC(numQuestions))]
    # shouldSelectClip should be true if the mobel is required to select a clip, or false if should select a question inside the clip
    # listOfContextValid_questions and listOfContextValid_clips are the ids of the questions and clips that can be put in this current context
    def call(self, obs_data):
        # Step 0: Extract parameters from the obs data
        questionsHistoryProcessed, scoresHistory, listOfContextValid_clipsIds, listOfContextValid_questionsIds, \
                    trainingMode, shouldSelectClips, currentClipId = obs_data

        if isinstance(currentClipId, tf.Tensor):
            currentClipId = currentClipId.numpy()

        assert (shouldSelectClips and len(listOfContextValid_clipsIds) > 0) or (not shouldSelectClips and len(listOfContextValid_questionsIds) > 0),  \
                    "You requested to selected something that the context valid list is empty!"

        setOfContextValid_questionsIds = set(listOfContextValid_questionsIds) if listOfContextValid_questionsIds is not None else None
        setOfContextValid_clipsIds = set(listOfContextValid_clipsIds) if listOfContextValid_clipsIds is not None else None
        shouldSelectClip = bool(shouldSelectClips.numpy())

        allQuestionListForCurrentClip : List[QuestionForClip]= self.datastore.questions_byClip[currentClipId] if (currentClipId is not None and currentClipId != str(None)) else []
        allQuestionIdsListForCurrentClip = None if allQuestionListForCurrentClip is None else [q.id for q in allQuestionListForCurrentClip ]

        # Step 1: Create the current state embedding
        stateEmbedded = self.embeddingStateModel((scoresHistory, questionsHistoryProcessed, shouldSelectClip, listOfContextValid_questionsIds))

        # Step 2: Finally apply the head for getting the final output
        action_probs = None
        state_value = None
        if shouldSelectClips:
            action_probs, state_value = self.actor_clips(stateEmbedded), self.critic_clips(stateEmbedded)
        else:
            action_probs, state_value = self.actor_questions(stateEmbedded), self.critic_questions(stateEmbedded)
        action_probs = tf.squeeze(tf.nn.softmax(action_probs))
        state_value = tf.squeeze(state_value)

        # Now that we have the softmax, take each result in order and return the first one that is really available in the current context
        selectedQuestion = None
        selectedClip = None
        selectedItemIndex = None

        numOutputItems = action_probs.get_shape().as_list()
        numOutputItems = numOutputItems[-1]

        if shouldSelectClips:
            # READ EXAMPLE in P2 !!
            # Create a mask for action_probs and apply it
            #----
            # All invalid indices must be 0
            assert numOutputItems == len(self.listOfAllAvailableClipsIds)
            clipsIdsMask = np.ones(shape=numOutputItems, dtype=np.float)

            # All indices that represent invalid clips in this context must be 0
            for clipIndex in range(numOutputItems):
                clipId = self.listOfAllAvailableClipsIds[clipIndex] # Note indices may not be same as ids ! That's why this indirection
                if clipId not in setOfContextValid_clipsIds:
                    clipsIdsMask[clipIndex] = 0.0

            action_probs_numpy = np.copy(action_probs.numpy())
            action_probs_numpy *= clipsIdsMask
            action_probs_numpy /= action_probs_numpy.sum() # Normalize
            #----

            # Now select one according to the probabilities
            numOptions = len(action_probs_numpy)
            #print(f"Before choosing question, probabilities are {action_probs_numpy} action_probs_numpy. There are a number of {numOptions} options")
            selectedItemIndex = np.random.choice(numOptions, size=1, p=action_probs_numpy)
            #print(f"Action {selectedItemIndex} was selected")
            selectedItemIndex = selectedItemIndex[0]

            if selectedItemIndex >= len(self.listOfAllAvailableClipsIds) or \
                    self.listOfAllAvailableClipsIds[selectedItemIndex] not in setOfContextValid_clipsIds:
                selectedClip = next(iter(setOfContextValid_clipsIds))
                assert False, f"Invalid selected index {selectedItemIndex} or not in valid selection list. Selected {allQuestionIdsListForCurrentClip[selectedItemIndex]} but available {listOfContextValid_questionsIds}!"
            else:
                selectedClip = self.listOfAllAvailableClipsIds[selectedItemIndex]
        else:
            # READ EXAMPLE in P2 !!
            # Create a mask for action_probs and apply it
            #----
            # All invalid indices must be 0
            assert numOutputItems == self.maxQuestionsCountInsideAClip
            questionsCountForClip = len(allQuestionIdsListForCurrentClip)
            questionsMask = np.ones(shape=numOutputItems, dtype=np.float)
            questionsMask[questionsCountForClip : numOutputItems] = 0.0

            # All indices that represent invalid questions in this context must be 0
            for questionIndex in range(questionsCountForClip):
                questionId = allQuestionIdsListForCurrentClip[questionIndex]
                if questionId not in setOfContextValid_questionsIds:
                    questionsMask[questionIndex] = 0.0

            action_probs_numpy = np.copy(action_probs.numpy())
            action_probs_numpy *= questionsMask # Apply the mask
            action_probs_numpy /= action_probs_numpy.sum() # Normalize
            #----

            # Now select one according to the probabilities
            numOptions = len(action_probs_numpy)
            #print(f"Before choosing question, probabilities are {action_probs_numpy} action_probs_numpy. There are a number of {numOptions} options, valid are first ")
            selectedItemIndex = np.random.choice(numOptions, size=1, p=action_probs_numpy)
            #print(f"Action {selectedItemIndex} was selected")
            selectedItemIndex = selectedItemIndex[0]
            if selectedItemIndex >= len(allQuestionIdsListForCurrentClip) or \
                    allQuestionIdsListForCurrentClip[selectedItemIndex] not in listOfContextValid_questionsIds:
                selectedQuestion = listOfContextValid_questionsIds[0]
                assert False, f"Invalid selected index {selectedItemIndex} or not in valid selection list. Selected {allQuestionIdsListForCurrentClip[selectedItemIndex]} but available {listOfContextValid_questionsIds}!"
            else:
                selectedQuestion = allQuestionIdsListForCurrentClip[selectedItemIndex]

        # Check that we selected something valid as requested
        assert (shouldSelectClips == True and selectedClip) or (not shouldSelectClips and selectedQuestion), "We couldn't retrieve the item !!!"
        # If clip was requested, was a valid one selected
        assert shouldSelectClip == False or ( selectedClip in setOfContextValid_clipsIds)
        # If question was requested, was a valid one selected
        assert shouldSelectClips == True or ( selectedQuestion in setOfContextValid_questionsIds)

        # Returns the following pair: action probability for the item selected
        # value of the state as being estimated by the critic
        # the best clip, question selected (for inference)
        return action_probs[selectedItemIndex], state_value, selectedClip, selectedQuestion



# This holds and process an episode gradients info
class AgentRLEpisodeInfo:
    def __init__(self, realAgentClusterIndex : float):
        self.resetNewEpisode(realAgentClusterIndex)

    def resetNewEpisode(self, realAgentClusterIndex : float):
        self.realAgentClusterIndex = realAgentClusterIndex

        # All values along the episode path's transitions
        self.action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.rewards = [] #np.array()#tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        # This retains as temp values the last values generated along the path
        self.currentStep_actionProb = None
        self.currentStep_stateValue = None
        self.currentStep_reward = None

        self.clipSelectionSteps = []  # Indices of steps where we selected clips

        # Step counter
        self.currentStepIndex = 0

    def __len__(self):
        return self.currentStepIndex

    # Adds one step information inside the episode path: the probability of the action being taken at that point, the value as estimated by
    # the critic, and the reward value
    def addStepInfo(self, actionProb : float, value : float, reward : float):
        # store rewardself.rewards
        #self.rewards        = .write(self.currentStepIndex, reward)
        #self.rewards.mark_used()
        assert self.currentStepIndex == len(self.rewards), "Seems like you added more items to this than current step in the episode ??"
        self.rewards.append(reward)

        self.values         = self.values.write(self.currentStepIndex, value)
        self.values.mark_used()
        self.action_probs   = self.action_probs.write(self.currentStepIndex, actionProb)
        self.action_probs.mark_used()

        # Clip selection ?
        if reward == UNKNOWN_REWARD:
            self.clipSelectionSteps.append(self.currentStepIndex)

        # Increase the step
        self.currentStepIndex += 1

    def onStepMade(self):
        self.addStepInfo(actionProb=self.currentStep_actionProb,
                         value=self.currentStep_stateValue,
                         reward=self.currentStep_reward)

    def finalizeEpisodeInfo(self):
        # Transform to stacked tensors
        self.action_probs        = self.action_probs.stack()
        self.values              = self.values.stack()
        #self.rewards_numpy             = self.rewards.stack().numpy()

        # Fix the clip selection rewards. Note: the rewards will be None
        assert len(self.clipSelectionSteps) > 0, "You didn;t selected any clips in this episode ??"
        episodeLen = len(self.rewards)
        self.clipSelectionSteps.append(episodeLen) # Mark the end of the episode as a fake clip to simplify the algorithm below
        clipsSelectedLen = len(self.clipSelectionSteps)


        # Computes the estimated rewards discounted by GAMMA between [startingStep, endingStep)
        def computeEstimatedAvgRewardForClip(startingStep, endingStep):
            if startingStep + 1 == endingStep: # No question between clips ?
                assert False, "There was no question between clips.."
                return 0

            estReward = 0

            discountFactor = 1.0
            for step in range(startingStep + 1, endingStep):
                rewardAtStep = self.rewards[step]
                assert rewardAtStep != None, "Reward is none but expected to be a question not a clip in this step !"
                estReward += rewardAtStep * discountFactor
                discountFactor *= GAMMA

            return estReward / (endingStep - startingStep)

        prevClipSelectedStep = None
        for clipSelectedIndex in range(clipsSelectedLen):
            # We need to compute the reward between two consecutive clip selection indices by gamma average formula.
            # Note: in the end check
            clipSelectedStep =self.clipSelectionSteps[clipSelectedIndex]

            # Either the false step, or there is a clip here with reward not set !
            assert clipSelectedStep >= len(self.rewards) or self.rewards[clipSelectedStep] == UNKNOWN_REWARD

            if prevClipSelectedStep is None:
                prevClipSelectedStep = clipSelectedStep
                continue

            # We have a previous step and new selected. Compute the estimated reward for the questions under the previous clip selected
            prevRewardAtClip = self.rewards[prevClipSelectedStep]
            assert prevRewardAtClip is None or prevRewardAtClip == UNKNOWN_REWARD, "Expecting reward to be None because it is a clip step selection unprocessed yet !" # A sanity check first
            estRewardForClip = computeEstimatedAvgRewardForClip(prevClipSelectedStep, clipSelectedStep)
            self.rewards[prevClipSelectedStep] = estRewardForClip # Then compute and set the real value instead of None

            # Change the previous
            prevClipSelectedStep = clipSelectedStep

        # Sanity check that all are valid now
        for i in range(episodeLen):
            assert self.rewards[i] != None and self.rewards[i] != UNKNOWN_REWARD, f"Value on index {i} is still None !!!"

class AgentSurveryRL(AgentPathfinding):
    def __init__(self,
                 trainingMode,
                 # Parameters for dataset stuff
                 orgInterest: OrganizationInterestSettings, dataStore: DataStore, clustersSpec: ManualClustersSpec,
                 surveySettings : SurveyBuildSettings,
                 # Parameter for the agent itself
                 seed = None,
                 replayBufferSize = 32000,
                 gamma = 0.9, # the discount factor
                 batch_size = 1024,
                 lr = 0.001, # learning rate (alpha)
                 steps_until_sync=20,  # At how many steps should we update the target network weights
                 choose_action_frequency=1,
                 # At how many steps should we take an action decision (think about pong game: you need to keep the same action for x frames ideally...)
                 pre_train_steps=1,  # Steps to run before starting the training process,
                 train_frequency=1,
                 # At how many steps should I update the training. You could gather more data and do the training in batches more..
                 epsScheduler=  Utils_replay.LinearScheduleEpsilon()): # How to update epsilon ?

        super(AgentSurveryRL, self).__init__(orgInterest, dataStore, clustersSpec)
        self.trainingMode = trainingMode
        self.selectionModel = AgentActorCriticModel(dataStore)
        self.surveySettings = surveySettings
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        self.currentEpisodeInfo : AgentRLEpisodeInfo = None

        # Optimizer setup
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)

        # Have deterministic output at each run for easy debugging
        if seed is not None:
            self.env.seed(seed)
            tf.random.set_seed(seed)
            np.random.seed(seed)

        # Total steps of the running algorithm
        self.total_steps = 0

    def resetEnvironmentState(self):
        super().resetState()
        return self.getObservationData(trainingMode=self.trainingMode, shouldSelectClips=True)

    # Given the current context, create observation compatible with model's input
    # Note that this is somehow different than OpenAI gym env, since we need to store some obs data inside the agent itself...
    def getObservationData(self, trainingMode, shouldSelectClips):
        questionsHistory : List[QuestionResponseDeviation] = self.historyOfQuestionsAndDeviations
        scoresHistory = self.historyOfClusterProbabilityValues

        questionsHistoryProcessed = [(questionData.parentClipId, questionData.questionId, questionData.rawDeviation) for questionData in questionsHistory]

        validQuestionIds = None
        if shouldSelectClips == False:
            normalizedScoresForValidQuestions: List[Tuple[any, float]] = self.NextTheme_Questions_Valid()
            validQuestionIds = [qID for qID, _ in normalizedScoresForValidQuestions]

        # TODO11: same for clips ids
        validClipsIds = None
        if shouldSelectClips == True:
            clipIdsAndNormalizedScores: List[Tuple[int, float]] = self.getNextTheme_Clip_ValidSet()
            validClipsIds = [tID for tID, _ in clipIdsAndNormalizedScores]

        rawObsData = (questionsHistoryProcessed, scoresHistory, validClipsIds, validQuestionIds,
                      trainingMode, shouldSelectClips, self.currentTheme_clipId)

        return rawObsData

    # Init some stuff at the beggining of the survey
    def beginSurvey(self, settings : SurveyBuildSettings):
        super().beginSurvey(settings)

        # Reset state
        self.resetEnvironmentState()

        # Initialize a new episode agent person behind and episode info data structure
        self.currentEpisodeInfo = AgentRLEpisodeInfo(self.chosenAgentClusterIndex)

    def endSurvey(self, supressOutput : bool, outAgentSurveyStats:AgentSurveyStats):
        super().endSurvey(supressOutput, outAgentSurveyStats)

    ######################### INTERFACE FOR PARENT EVALUATION AS A SURVEY AGENT ###############
    # Selects the next question inside current theme according to local strategy
    def selectNextQuestionInternal(self, normalizedQuestionsScores):
        currentObservationData = self.getObservationData(trainingMode=self.trainingMode, shouldSelectClips=False)
        #currentObservationData = tf.constant(currentObservationData, shape=(-1,))
        action_prob, state_value, selectedClip, selectedQuestion = self.selectionModel(currentObservationData)
        # Fill the temp data
        self.currentEpisodeInfo.currentStep_actionProb = action_prob
        self.currentEpisodeInfo.currentStep_stateValue = state_value
        self.currentEpisodeInfo.currentStep_reward = UNKNOWN_REWARD

        return selectedQuestion

    # Get next question under a theme when the clip was already selected
    def getNextTheme_Question(self):
        return super().getNextTheme_Question()

    # Selects the next clip according to local strategy
    def selectNextClipInternal(self, clipIdsAndNormalizedScores):
        currentObservationData = self.getObservationData(trainingMode=self.trainingMode, shouldSelectClips=True)
        action_prob, state_value, selectedClipId, selectedQuestionId = self.selectionModel(currentObservationData)

        # Fill the temp data
        self.currentEpisodeInfo.currentStep_actionProb = action_prob
        self.currentEpisodeInfo.currentStep_stateValue = state_value
        self.currentEpisodeInfo.currentStep_reward = UNKNOWN_REWARD

        # We need to call the step end here because this is the point when clip decision ends
        self.currentEpisodeInfo.onStepMade()
        return selectedClipId

    def getNextTheme_Clip(self, themeId):
        return super().getNextTheme_Clip(themeId)

    def setCurrentQuestionAnswer(self, answerValue):
        # This will update the parent which in turn updates the scores statistics
        super().setCurrentQuestionAnswer(answerValue)

        # The last history scores is now the probability of each cluster
        currentProbsPerCluster = self.historyOfClusterProbabilityValues[-1]
        lastProbsPerCluster = self.historyOfClusterProbabilityValues[-2] if len(self.historyOfClusterProbabilityValues) > 1 else None
        #assert lastProbsPerCluster != None, "We should always have something in here !"

        # Reward in this moment is how far are we from the real cluster probability + velocity
        # Velocity is not applicable if the
        overallDistanceWeight = REWARD_FACTOR_OVERALLDIST #1.0 if lastProbsPerCluster is None else REWARD_FACTOR_OVERALLDIST
        velocityDistanceWeight = REWARD_FACTOR_VELOCITY #0.0 if lastProbsPerCluster is None else REWARD_FACTOR_VELOCITY
        realAgentClusterIndex = self.currentEpisodeInfo.realAgentClusterIndex

        current_prob_diffToRealCluster  = currentProbsPerCluster[realAgentClusterIndex] - 1.0
        prev_prob_diffToRealCluster = (currentProbsPerCluster[realAgentClusterIndex] - lastProbsPerCluster[realAgentClusterIndex]) #if lastProbsPerCluster is not None else 0.0

        self.currentEpisodeInfo.currentStep_reward = (current_prob_diffToRealCluster * overallDistanceWeight) + \
                                                     (prev_prob_diffToRealCluster * velocityDistanceWeight)

        # We need to call the step end here because this is the point where question decision ends
        self.currentEpisodeInfo.onStepMade()

    """
    # Receive action returns a state, reward, done
    # We wrap-up this to be compatible in a tf graph op
    def onStepMade(self, action : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state, reward, done, _ = self.env.step(action)
        return (state.astype(np.float32),
                np.array(reward, np.int32),
                np.array(done, np.int32))

    def tf_env_step(self, action:tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.onStepMade, [action], [tf.float32, tf.int32, tf.int32])

    """
    # Runs a single episode to collect training data
    # List of 3 tensors: action_probs, values, rewards
    def run_episode(self) -> List[tf.Tensor]:
        # Initialize a Survey (our environment) with this object as agent under evaluation
        # Inverse of control....basically when we run this, it will record a full episode here...
        # NOTE: the initialization of a survey will push state reset and episode setup ! Check begin survey
        environmentSurvey = SurveyFactory(dataStore=self.dataStore, agent=self, settings=self.surveySettings)
        environmentSurvey.buildSurveyDemo(supressOutput=True)

        self.currentEpisodeInfo.finalizeEpisodeInfo()

        return self.currentEpisodeInfo.action_probs, \
               self.currentEpisodeInfo.values, \
               self.currentEpisodeInfo.rewards

    def policyGradient_compute_loss(self, action_probs : tf.Tensor, values : tf.Tensor, returns : tf.Tensor) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        advantage = returns - values
        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
        critic_loss = self.huber_loss(values, returns)
        return actor_loss + critic_loss, actor_loss, critic_loss

    def policyGradient_get_expected_returns(self, rewards : List, gamma, standardize: bool = True)->tf.Tensor:
        # expected return per timestep
        n = len(rewards) # num items
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        rewards = rewards[::-1] #tf.cast(rewards[::-1], tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = tf.shape(discounted_sum)
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + np.finfo(float).eps))

        return returns

    # Runs a train episode and returns the reward
    def policyGradient_train_step(self,gamma, standardize=STANDARDIZE_RETURNS):
        with tf.GradientTape() as tape:
            # Run model for one episode and collect training data
            action_probs, values, rewards = self.run_episode()

            # Compute the expected returns
            returns = self.policyGradient_get_expected_returns(rewards, gamma, standardize)

            # Convert to the batch shapes
            action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # Compute loss values
            total_loss, actor_loss, critic_loss = self.policyGradient_compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(total_loss, self.selectionModel.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.selectionModel.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)
        return episode_reward, {'total_loss' : total_loss.numpy(), 'actor_loss' : actor_loss.numpy(), 'critic_loss' : critic_loss.numpy()}

    def train(self
              ,max_episodes
              #,reward_threshold  # Not really used but logic is written below...
              ):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/train/' + current_time
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        running_reward = 0
        running_lossDict = {'total_loss' : 0, 'actor_loss' : 0, 'critic_loss' : 0}
        with tqdm(range(max_episodes)) as t:
            for i in t:
                rewardObtained, episode_lossDict = self.policyGradient_train_step(gamma=GAMMA)
                episode_reward = rewardObtained.numpy()

                for key in running_lossDict.keys():
                    running_lossDict[key] = episode_lossDict[key] * 0.01 + running_lossDict[key] * 0.99

                running_reward = episode_reward*0.01 + running_reward*0.99

                t.set_description(f'Episode {i}')
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward,
                              totalLoss=episode_lossDict['total_loss'],
                              running_totalLoss=running_lossDict['total_loss'],
                              actorLoss=episode_lossDict['actor_loss'],
                              criticLoss=episode_lossDict['critic_loss'])

                print("\n")


                #if i % 10 == 0:
                #    print(f'Episode {i}: average reward: {running_reward}')

                with train_summary_writer.as_default():
                    tf.summary.scalar('episode_reward', episode_reward, step=i)
                    tf.summary.scalar('total_loss', episode_lossDict['total_loss'], step=i)
                    tf.summary.scalar('actor_loss', episode_lossDict['actor_loss'], step=i)
                    tf.summary.scalar('critic_loss', episode_lossDict['critic_loss'], step=i)

                #if running_reward > reward_threshold:
                #    break

        print(f"\nFinished at episode{i}: average running rewards {running_reward:.2f}!")

    def saveModel(self, modelName):
        self.dqn.save_weights(modelName)

    def loadModel(self, modelName):
        self.dqn.load_weights(modelName)
        self.dqn_target.set_weights(self.dqn.get_weights())


# Creates a random dataset for todolist

def randomDataset(num_timesteps):
    NUM_CLUSTERS = 4
    MAX_NUM_QUESTIONS = 200
    MAX_NUM_CLIPS = 20

    res = {}
    # Get some random questions and theme id
    listOfAllAvailableQuestionsIds = np.random.choice(MAX_NUM_QUESTIONS, size=(100), replace=False)
    listOfAllAvailableThemesIds = np.random.choice(MAX_NUM_CLIPS, size=(20), replace=False)

    listOfContextValid_themesIds = list(np.random.choice(listOfAllAvailableThemesIds, size=(len(listOfAllAvailableThemesIds) // 4), replace=False))
    listOfContextValid_questionsIds = list(np.random.choice(listOfAllAvailableQuestionsIds, size=(len(listOfAllAvailableQuestionsIds) // 4), replace=False))

    res['listOfAllAvailableQuestionsIds'] = listOfAllAvailableQuestionsIds
    res['listOfAllAvailableThemesIds'] = listOfAllAvailableThemesIds
    res['listOfContextValid_themesIds'] = listOfContextValid_themesIds
    res['listOfContextValid_questionsIds'] = listOfContextValid_questionsIds

    # Randomize some themes and question ids
    themes = np.random.randint(low=0, high=MAX_NUM_CLIPS, size=(num_timesteps))
    questions = np.random.randint(low=0, high=MAX_NUM_QUESTIONS, size=(num_timesteps))
    devscores = np.random.uniform(low=-7, high=7, size=[num_timesteps])

    questions_data = list(zip(themes, questions, devscores))
    clusterprobs_data = np.random.uniform(low=0, high=1, size=[num_timesteps, num_clusters])
    timestemValues = (questions_data, clusterprobs_data)

    res['timestemValues'] = timestemValues

    return res


# Wrapper function to select either a clip or a question inside current theme, given the set of context valid questions ids and clips ids
# Note that these 'context valid' must be a subset of the initial set of all questions and clips
def selectNextItemWrapper(model, data, training, shouldSelectClips,
                          listOfContextValid_questionsIds : List[int],
                          listOfContextValid_clipsIds : List[int]):

    return model(input_data=data['timestemValues'],
                 shouldSelectClips=shouldSelectClips,
                 training=training,
                 listOfContextValid_questionsIds=listOfContextValid_questionsIds,
                 listOfContextValid_clipsIds=listOfContextValid_clipsIds)


###############################################

def testRLAgent(USE_DETERMINISTIC_SURVEY_GENERATION, TRAINING_MODE):
    dataStore = loadDataStore()

    # Set org interests
    organizationInterests = OrganizationInterestSettings()
    orgAttributesSet = {"Sexual Harassment": 1.0, "Leadership": 0.8, "Personal Boundaries": 0.5}
    orgCategoriesSet = {"Awareness": 1.0, "Sanction": 0.9, "Prevalence": 0.5}
    organizationInterests.attributesInterestedIn = orgAttributesSet
    organizationInterests.categoriesInterestedIn = orgCategoriesSet

    if USE_DETERMINISTIC_SURVEY_GENERATION:
        random.seed(0)
    else:
        random.seed(datetime.now())

    # Set a pathfinding agent
    #----------------------------------
    emptyOrganizationInterests = OrganizationInterestSettings() # This must be empty since we already define the cloud of clusters and  they contain the interesting features inside already.

    # Setup a cloud of behavior type / clusters.
    # Note: A ManualSingleClusterSpec can contain many behavior types, but all must share the same feature

    surveySettings = SurveyBuildSettings(numThemes=10, minQuestionsPerTheme=3, maxQuestionsPerTheme=4,
                                         isCategoriesScoringEnabled=True, forceMaximizeNumQuestionsPerTheme=True)

    manualClustersSpec = get_ManualClusteringSpecDemo()

    baseAgent = AgentSurveryRL(orgInterest=emptyOrganizationInterests,
                               dataStore=dataStore,
                               clustersSpec=manualClustersSpec,
                               surveySettings=surveySettings,
                               trainingMode=TRAINING_MODE)
    #print(baseAgent)

    if TRAINING_MODE is False:
        # Create a survey on inference
        # TODO: load model
        surveyFactory = SurveyFactory(dataStore=dataStore, agent=baseAgent, settings=surveySettings)
        outputSurvey = surveyFactory.buildSurveyDemo(supressOutput=False)
        #print(outputSurvey)
    else:
        baseAgent.train(MAX_TRAINING_EPISODES)


def loadDataStore():
    dataStore = DataStore()
    attrDataframe = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datastore/attributes.csv"))
    catDataframe = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datastore/categories.csv"))
    dataStore.LoadAttributesAndCategories(attrDataframe=attrDataframe, catDataframe=catDataframe)
    clipsAttributesDataframe = \
        pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datastore/clips_attributes.csv"))
    clipsMetaDataframe = \
        pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datastore/clips_meta.csv"))
    dataStore.LoadClips(clipsAttributesDataframe=clipsAttributesDataframe, clipsMetaDataframe=clipsMetaDataframe)
    questionsDataframe = \
        pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datastore/questions.csv"))
    dataStore.LoadQuestions(questionsDataframe=questionsDataframe)
    return dataStore


def getNextQuestionId(
        attrDataframe,
        catDataframe,
        clipsAttributesDataframe,
        clipsMetaDataframe,
        questionsDataframe,
        attributesInterestedIn,
        categoriesInterestedIn,
        numThemes,
        minQuestionsPerTheme,
        maxQuestionsPerTheme,
        isCategoriesScoringEnabled,
        forceMaximizeNumQuestionsPerTheme,
        themeId,
        lastSelectedQuestionId,
        lastSelectedClipId,
        setOfAlreadySelectedClips,
        prevSelectedClipId,
        setOfAlreadySelectedQuestionsForCurrentClip,
):
    def loadDataStore():
        random.seed(datetime.now())
        result = DataStore()
        result.LoadAttributesAndCategories(attrDataframe=attrDataframe, catDataframe=catDataframe)
        result.LoadClips(clipsAttributesDataframe=clipsAttributesDataframe, clipsMetaDataframe=clipsMetaDataframe)
        result.LoadQuestions(questionsDataframe=questionsDataframe)
        return result

    def getAgent():
        def getOrganizationInterests():
            result = OrganizationInterestSettings()
            result.attributesInterestedIn = attributesInterestedIn
            result.categoriesInterestedIn = categoriesInterestedIn
            return result

        return AgentBasic.AgentBasic(orgInterest=getOrganizationInterests(), dataStore=dataStore)

    def setSurveySettings():
        surveySettings = SurveyBuildSettings(
            numThemes=numThemes,
            minQuestionsPerTheme=minQuestionsPerTheme,
            maxQuestionsPerTheme=maxQuestionsPerTheme,
            isCategoriesScoringEnabled=isCategoriesScoringEnabled,
            forceMaximizeNumQuestionsPerTheme=forceMaximizeNumQuestionsPerTheme,
        )
        agent.beginSurvey(surveySettings)

    def setClipToBePresentedOnAgent():
        def setCurrentClipSettingsOnAgent():
            def setNumberOfQuestionsToBeAsked():
                if forceMaximizeNumQuestionsPerTheme:
                    agent.currentTheme_targetNumQuestionsToSelect = maxQuestionsPerTheme
                else:
                    totalNumberOfQuestionsInSelectedClip = len(dataStore.questions_byClip[idOfClipToBeSelected])
                    agent.currentTheme_targetNumQuestionsToSelect = min(
                        random.randint(minQuestionsPerTheme, maxQuestionsPerTheme),
                        totalNumberOfQuestionsInSelectedClip
                    )

            agent.currentTheme_id = themeId
            if lastSelectedQuestionId is not None:
                idOfClipToBeSelected = agent.currentTheme_clipId = lastSelectedClipId
                agent.prevQuestionIdSelected = lastSelectedQuestionId
            else:
                idOfClipToBeSelected = agent.currentTheme_clipId
            agent.setOfAlreadySelectedClips = setOfAlreadySelectedClips
            agent.prevSelectedClipId = prevSelectedClipId
            agent.setOfAlreadySelectedQuestionsForClip[agent.currentTheme_clipId] = \
                setOfAlreadySelectedQuestionsForCurrentClip
            setNumberOfQuestionsToBeAsked()

        theme = Theme(attributesFlattened=dataStore.attributesFlattened)
        theme.id = themeId
        if lastSelectedQuestionId is None and nextClip() is None:
            return False
        else:
            setCurrentClipSettingsOnAgent()
            return True

    def nextClip():
        try:
            return agent.getNextTheme_Clip(themeId)
        except AssertionError:
            return None

    def nextQuestionId():
        suggestedQuestionId = agent.getNextQuestionId()
        if suggestedQuestionId == INVALID_ID:
            if nextClip() is None:
                return None
            return agent.getNextQuestionId()
        else:
            return suggestedQuestionId

    dataStore = loadDataStore()
    agent = getAgent()
    setSurveySettings()
    if not setClipToBePresentedOnAgent():
        return None

    question_id = nextQuestionId()
    return question_id if question_id != INVALID_ID else None

##############################################


if __name__ == "__main__":
    # Unit todolist 0 :todolist if we can produce a next clip /question when requested
    RUN_TEST_0 = False
    if RUN_TEST_0:
        datasetSample = randomDataset(num_timesteps=10)

        # The model is instatiated with all set of available questions ids and clips ids in the dataset
        agentRLStateModel = AgentRLStateModel(setOfAllAvailableQuestionsIds=datasetSample['setOfAllAvailableQuestionsIds'],
                                            setOfAllAvailableClipsIds=datasetSample['setOfAllAvailableClipsIds'])


        nextClip = selectNextItemWrapper(agentRLStateModel, datasetSample, training=True, shouldSelectClips=True,
                                              listOfContextValid_questionsIds=datasetSample['listOfContextValid_questionsIds'],
                                              listOfContextValid_themesIds=datasetSample['listOfContextValid_clipsIds'])
        #print(nextClip.shape)
        nextQuestion = selectNextItemWrapper(agentRLStateModel, datasetSample, training=True, shouldSelectClip=False,
                                              listOfContextValid_questionsIds=datasetSample['listOfContextValid_questionsIds'],
                                              listOfContextValid_clipsIds=datasetSample['listOfContextValid_clipsIds'])
        #print(nextQuestion.shape)

    # Unit todolist 1: train a full agent
    RUN_TEST_1 = False
    if RUN_TEST_1:
        testRLAgent(USE_DETERMINISTIC_SURVEY_GENERATION=True, TRAINING_MODE = True)

    # Unit todolist 2: inference a trained agent
    RUN_TEST_2 = True
    if RUN_TEST_2:
        testRLAgent(USE_DETERMINISTIC_SURVEY_GENERATION=True, TRAINING_MODE = False)

