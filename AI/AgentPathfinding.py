# TODO:
# Compare with AgentBasic
# Add doc

from AI.AgentUtils import *
from AI.AgentBasic import *
from AI.AgentAbstract import *
from AI.DataDefinitions import *
from AI.AgentAbstract import *
from typing import Set, List, Dict, Tuple
import random
import numpy as np

# Decay rate of the explorations
EXPLORATION_EPSILON_START = 1.0
EXPLORATION_EPSILON_END = 0.1
EXPLORATION_DECAY_PER_QUESTION = 0.85

from AI.DataDefinitions import ManualClusterFeature, QuestionResponseDeviation, ManualClusterScores


class AgentPathfinding(AgentBasic):
    def __init__(self, orgInterest: OrganizationInterestSettings, dataStore: DataStore,
                 clustersSpec : ManualClustersSpec, verbose=False):

        self.clustersSpec : ManualClustersSpec = clustersSpec
        assert orgInterest.isEmpty(), "For this agent you should pass an empty organization setting because you \
                                    already define the things in the clusters spec"

        super(AgentPathfinding, self).__init__(orgInterest, dataStore, verbose=verbose, clustersSpec=clustersSpec)
        self.resetState()

    def getAgentId(self):
        return PATHFINDING_AGENT_ID

    def resetState(self):
        super().resetState()

        self.numSelectedRandomly_exploration = 0 # How many times did the agent selected something because of the exploration rate prob
        self.numSelectedRandomly_roulette = 0 # Same as above, but this time random by the roulette

    # Callback executed in the beggining of a survey
    def beginSurvey(self, settings : SurveyBuildSettings):
        super().beginSurvey(settings)

    # Gets the set of questions available in the current state and theme scored by cosine similarity
    def NextTheme_Questions_Valid(self) -> List[Tuple[any, float]]:
        # Gets the most important features for classifying the agent now and compute the score based on them
        featuresInterestedIn = self.getCurrentAgentStateInterestSettings()
        normalizedScores = AgentUtils.cosineSim_questionSelection(orgCategoriesInterest=featuresInterestedIn.categoriesInterestedIn,
                                                                  prevQuestionIdSelected=self.prevQuestionIdSelected,
                                                                    dataStore=self.dataStore,
                                                                    parentClipId=self.currentTheme_clipId,
                                                                    setOfAlreadySelectedQuestionsForClip=self.setOfAlreadySelectedQuestionsForClip,
                                                                    surveySettings = self.settings)

        return normalizedScores

    # Selects the next question inside current theme according to local strategy
    def selectNextQuestionInternal(self, normalizedQuestionsScores):
        return AgentUtils.selectRoulette(normalizedQuestionsScores, elitismPercent=ELITISM_PERCENT_QUESTIONS)

    # Get next question under a theme when the clip was already selected
    def getNextTheme_Question(self):
        normalizedScores = self.NextTheme_Questions_Valid()

        if len(normalizedScores) == 0:
            return None

        queId = self.selectNextQuestionInternal(normalizedScores)
        return queId

    # Get the normalized scored set of available themes ids
    def getNextTheme_Clip_ValidSet(self) -> List[Tuple[int, float]]:
        # Gets the most important features for classifying the agent now and compute the score based on them
        featuresInterestedIn = self.getCurrentAgentStateInterestSettings()
        clipIdsAndNormalizedScores: List[Tuple[int, float]] = AgentUtils.cosineSim_clipsSelection(
            orgAttributesInterest=featuresInterestedIn.attributesInterestedIn,
            orgCategoriesInterested=featuresInterestedIn.categoriesInterestedIn,
            minQuestionsForClip=1,  # self.settings.minQuestionsPerTheme,
            dataStore=self.dataStore,
            alreadySelectedClips=self.setOfAlreadySelectedClips,
            prevSelectedClipId=self.prevSelectedClipID)

        return clipIdsAndNormalizedScores

    # Selects the next clip according to local strategy
    def selectNextClipInternal(self, clipIdsAndNormalizedScores):
        # Scale by a factor containing only the elite part
        return AgentUtils.selectRoulette(clipIdsAndNormalizedScores, elitismPercent=ELITISM_PERCENT_CLIPS)

    # Get clip id for the next theme, the best according to score and state
    def getNextTheme_Clip(self, themeId):
        self.resetCurrentThemeState()

        clipIdsAndNormalizedScores : List[Tuple[int, float]] = self.getNextTheme_Clip_ValidSet()
        if len(clipIdsAndNormalizedScores) == 0:
            assert False, "I couldn't select a clip anymore for this theme"
            return None

        clipIdSelected = self.selectNextClipInternal(clipIdsAndNormalizedScores)
        assert clipIdSelected is not None

        self.currentTheme_id = themeId
        self.onClipSelectedForTheme(themeId=themeId, clipId=clipIdSelected)

        self.currentTheme_questionsUsed = [] # init the series of questions for this clip as empty
        numTotalQuestionsInClipSelected = len(self.dataStore.questions_byClip[clipIdSelected])

        # Decide how many questions to select the questions for this clip
        self.currentTheme_targetNumQuestionsToSelect = 0
        if self.settings.forceMaximizeNumQuestionsPerTheme:
            self.currentTheme_targetNumQuestionsToSelect = self.settings.maxQuestionsPerTheme
        else:
            self.currentTheme_targetNumQuestionsToSelect = min(random.randint(self.settings.minQuestionsPerTheme, self.settings.maxQuestionsPerTheme),
                                                           numTotalQuestionsInClipSelected)

        return clipIdSelected

    # Gets the most important attributes and categories which should help the AI reviewer to eliminate entropy from the user's behavior classification
    def getCurrentAgentStateInterestSettings(self) -> OrganizationInterestSettings:
        # Get the e-best cluster to use in this context
        nextBestClusterIndex = self.selectCluster(self.clustersSpec)
        currentInterest = AgentUtils.createOrganizationInterestSettings_byIndividualClusterSpec(self.clustersSpec.clusters[nextBestClusterIndex], self.dataStore)
        return currentInterest

    def getCurrentExplorationEpsilon(self):
        index = self.getCurrentQuestionIndex()
        res = max(EXPLORATION_EPSILON_END, (EXPLORATION_EPSILON_START * (EXPLORATION_DECAY_PER_QUESTION ** index)))
        return res

    # This selects the cluster index to use next for our clip / question selection
    def selectCluster(self, clustersSpec : ManualClustersSpec):
        numClusters = len(clustersSpec.clusters)
        scores, detailedScores = self.scoreClusters(clustersSpec)
        probResults = scores.result
        assert probResults.shape[0] == 1  and  probResults.shape[1] == (numClusters + 1) # Expected: One user | user id + probability for each cluster
        probabilitiesPerCluster = probResults[0][1:]
        assert probabilitiesPerCluster.shape[0] == numClusters

        clusterIndexToUse = None
        if DebugSettings.DEBUG_KNOWING_PERSON_BEHAVIOR is False:
            # if P < eps => choose each action with equal probability
            currEps = self.getCurrentExplorationEpsilon()
            if currEps > np.random.rand():
                clusterIndexToUse = np.random.choice(a = numClusters, size=1)[0]
                self.numSelectedRandomly_exploration += 1

                if self.verbose:
                    print(f"Following random cluster {clusterIndexToUse}. Exploration Epsilon is {currEps} Current probabilities per cluster: {self.currentProbabilityPerCluster}. "
                          f"Random expl rate so far (#randomselections (clip and question id) at question index): {self.numSelectedRandomly_exploration}/{self.getCurrentQuestionIndex() + 1}")
            else:
                # Roulette well selection with the found probabilities distribution - trying to get the most trustable set of clusters
                clusterIndexToUse = np.random.choice(a = numClusters, size = 1, p=probabilitiesPerCluster)[0]
                bestCluster = int(np.argmax(probabilitiesPerCluster))
                if bestCluster != clusterIndexToUse:
                    self.numSelectedRandomly_roulette += 1

                if self.verbose:
                    print(f"Following cluster {clusterIndexToUse} based on random roullete selection. Best was {bestCluster}. Random Roulette rate: {self.numSelectedRandomly_roulette}/{self.getCurrentQuestionIndex() + 1}")
        else:
            clusterIndexToUse = self.chosenAgentClusterIndex
            if self.verbose:
                print(f"Following cluster {clusterIndexToUse} FORCED FOR DEBUG")

        assert clusterIndexToUse is not None, "Failed to select a target cluster"

        return clusterIndexToUse

