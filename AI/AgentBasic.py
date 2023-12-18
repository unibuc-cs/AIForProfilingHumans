from AgentUtils import *
from AgentAbstract import *
from DataDefinitions import *
from typing import Set, List, Dict
import random
from VisualizationUtils_2 import ClusterPlotterHelp

# No memory agent, respecting the organization interest.
# Optionally it can include a clusterSpecification too, but that is used only for testing purposes - there is no algorithm to approach one of them
class AgentBasic(AbstractAgent):
    def __init__(self, orgInterest: OrganizationInterestSettings, dataStore: DataStore, verbose=False, clustersSpec=None):
        self.clustersSpec : ManualClustersSpec = clustersSpec
        self.currentProbabilityPerCluster = np.array([1.0/self.clustersSpec.numClusters for i in range(self.clustersSpec.numClusters)]) # By default, equal probability for each cluster
        super(AgentBasic, self).__init__(orgInterest, dataStore, verbose=verbose)
        self.chosenAgentClusterIndex = None # This is the real cluster behind the person ; Used for testing purposes to check performance of the algorithms

    def getAgentId(self):
        return BASE_AGENT_ID

    def resetState(self):
        super().resetState()

        # For a theme (clip under evaluation, store some temporary values
        self.resetCurrentThemeState()
        self.setOfAlreadySelectedClipsIds = set() # Set of clip ids already shown
        self.prevSelectedClipID = NO_DEPENDENCY_CONST

        # The set of asked question by clip id. This is useful to be kept as global because requrest might change and someone could say, ask the same clip between different themes !
        self.setOfAlreadySelectedQuestionsForClip : Dict[any, Set[int]] = {}

        # Mean deviation for each category and attribute name
        self.agentDeviations_byCategory_raw : Dict[any, float] = {}
        self.agentDeviations_byCategory_biased : Dict[any, float] = {}
        self.attributesDeviations_raw : Dict[any, float] = {}
        self.attributesDeviations_biased : Dict[any, float] = {}

        # Same as above but now the first index is the category
        self.attributesDeviations_byCategory_raw: Dict[any, Dict[str, float]] = {}
        self.attributesDeviations_byCategory_biased: Dict[any, Dict[str, float]] = {}

        self.numQuestionsContainingAttribute = {}
        self.numQuestionsContainingAttribute_perCategory = {}

        # Initialize the empty dictionaries above
        # ==============================================
        for attr in self.dataStore.attributesFlattened:
            self.attributesDeviations_raw[attr]  = 0.0
            self.attributesDeviations_biased[attr]  = 0.0
            self.numQuestionsContainingAttribute[attr] = 0

        for cat in self.dataStore.categoriesList:
            self.agentDeviations_byCategory_raw[cat] = 0.0
            self.agentDeviations_byCategory_biased[cat] = 0.0

        # Create empty templates for cache for attributes
        for catId in self.dataStore.categoriesList:
            self.attributesDeviations_byCategory_raw[catId] = {}
            self.attributesDeviations_byCategory_biased[catId] = {}
            self.numQuestionsContainingAttribute_perCategory[catId] = {}

            for attrId in self.dataStore.attributesFlattened:
                self.attributesDeviations_byCategory_raw[catId][attrId] = 0.0
                self.attributesDeviations_byCategory_biased[catId][attrId] = 0.0
                self.attributesDeviations_raw[attrId] = 0.0
                self.attributesDeviations_biased[attrId] = 0.0
                self.numQuestionsContainingAttribute_perCategory[catId][attrId] = 0
        # ==============================================

        # Probabilities of this agent to be on each of the clusters. Updated after each question response
        self.historyOfClusterProbabilityValues: List[List[float]] = []  # List of numpy arrays of float...

        # Initial set of probabilities is equal for each cluster
        numClusters = 1 if self.clustersSpec is None else len(self.clustersSpec)
        assert numClusters > 0
        initProbs = np.array([1.0 / numClusters] * numClusters)
        self.historyOfClusterProbabilityValues.append(initProbs)

    # Better estimation of user answer given the fact that we know which cluster is this user behind
    def simulateUserAnswer(self, questionId):
        answer = None
        if self.chosenAgentClusterIndex != None and self.clustersSpec != None:
            assert self.chosenAgentClusterIndex != None, "There is no real agent cluster index selected ! check what happened"
            realAgentClusterSpec : ManualSingleClusterSpec = self.clustersSpec.clusters[self.chosenAgentClusterIndex]

            questionData = self.dataStore.questions_byId[questionId]
            answer = realAgentClusterSpec.simulateUserAnswerToQuestion(questionData)
        else:
            answer = random.randint(MIN_QUESTION_RESPONSE, MAX_QUESTION_RESPONSE)
        return answer

    # This function could work on a running sum as O(1) instead of O(N), where N is  the number of questions for the user
    # But since N is so small, it doesn't worth the effort for debugging and maintenance
    def updateCurrentAgentStates(self):
        #currentClip     = self.dataStore.clips[self.currentTheme_clipId]
        #currentQuestion = self.dataStore.questions_byId[self.prevQuestionIdSelected]

        # Step 1: Computes the deviations by category
        for cat in self.agentDeviations_byCategory_raw:
            # TODO Ciprian : see other comments, we should compute for all other stats like MEDIAN, MIN, MAX
            res : QuestionResponseDeviation = SurveyResponses.reduceResponsesDeviations(self.questionDeviationsByCat[cat], ReduceOpType.MEAN)
            self.agentDeviations_byCategory_raw[cat] = res.rawDeviation
            self.agentDeviations_byCategory_biased[cat] = res.biasedDeviation

        # For normalization issues, we count how many questions contain a given attribute
        #-----
        self.numQuestionsContainingAttribute.clear()
        for attr in self.dataStore.attributesFlattened:
            self.numQuestionsContainingAttribute[attr] = 0

        for cat in self.dataStore.categoriesList:
            self.numQuestionsContainingAttribute_perCategory[cat] = {}
            for attr in self.dataStore.attributesFlattened:
                self.numQuestionsContainingAttribute_perCategory[cat][attr] = 0
        #------

        # Step 2:Computes the current stats based on attributes
        # ---------------------
        # Step 2.1 Summing up attributes step
        for questionAskedDevData in self.historyOfQuestionsAndDeviations:
            clipUsed = self.dataStore.clips[questionAskedDevData.parentClipId]
            questionData = self.dataStore.questions_byId[questionAskedDevData.questionId]
            questionCategory = questionData.category

            attributesUsed = clipUsed.attributes_nonzeroKeys
            for attributeId in attributesUsed:
                scoreOfAttributeInTheUsedClip = clipUsed.attributes.scores[attributeId]
                isGoodCond = (scoreOfAttributeInTheUsedClip >= 0.0 and scoreOfAttributeInTheUsedClip <= 1.0)
                assert isGoodCond, "The attributes factors in the dataset should be between 0-1 !"

                if isGoodCond:
                    self.numQuestionsContainingAttribute[attributeId] += 1
                    self.numQuestionsContainingAttribute_perCategory[questionCategory][attributeId] += 1

                # Multiply the deviation observed on  the question response by how important is the attribute for the clip presented
                rawDeviation_atrrValue = questionAskedDevData.rawDeviation * scoreOfAttributeInTheUsedClip
                biasedDeviation_attrValue = questionAskedDevData.biasedDeviation * scoreOfAttributeInTheUsedClip
                self.attributesDeviations_raw[attributeId] += rawDeviation_atrrValue
                self.attributesDeviations_biased[attributeId] += biasedDeviation_attrValue
                self.attributesDeviations_byCategory_raw[questionCategory][attributeId] += rawDeviation_atrrValue
                self.attributesDeviations_byCategory_biased[questionCategory][attributeId] += biasedDeviation_attrValue

        # Step 2.2 Normalization step
        #------------------
        for attributeId in self.attributesDeviations_raw:
            if self.numQuestionsContainingAttribute[attributeId] > 0:
                self.attributesDeviations_raw[attributeId] /= self.numQuestionsContainingAttribute[attributeId]
                self.attributesDeviations_biased[attributeId] /= self.numQuestionsContainingAttribute[attributeId]
            else:
                assert self.attributesDeviations_raw[attributeId] == 0.0 and \
                       self.attributesDeviations_biased[attributeId] == 0.0, \
                    "Attribute used but not counted ???"

        for categoryId in self.attributesDeviations_byCategory_raw:
            for attributeId in self.attributesDeviations_byCategory_raw[categoryId]:
                if self.numQuestionsContainingAttribute_perCategory[categoryId][attributeId] > 0:
                        self.attributesDeviations_byCategory_raw[categoryId][attributeId] /= self.numQuestionsContainingAttribute_perCategory[categoryId][attributeId]
                        self.attributesDeviations_byCategory_biased[categoryId][attributeId] /= self.numQuestionsContainingAttribute_perCategory[categoryId][attributeId]
                else:
                        assert self.attributesDeviations_byCategory_raw[categoryId][attributeId] == 0.0 and \
                            self.attributesDeviations_byCategory_biased[categoryId][attributeId] == 0.0,\
                                "Attribute used but not counted ???"

                        # Sanity check to see that all values are in range
                        assert -MAX_QUESTION_RESPONSE <= self.attributesDeviations_byCategory_raw[categoryId][attributeId] <= MAX_QUESTION_RESPONSE

                # Sanity check to see that all values are in range
                #assert -MAX_QUESTION_RESPONSE <= self.attributesDeviations_raw[attributeId] <= MAX_QUESTION_RESPONSE
        # -----------

    def resetCurrentThemeState(self):
        self.currentTheme_id = INVALID_ID  # The id of the theme
        self.currentTheme_clipId = INVALID_ID # The id of the clip selected for the theme

        self.currentTheme_targetNumQuestionsToSelect  = 0 # The number of questions to select for this theme
        self.currentTheme_numQuestionsAsked = 0 # THe number of questions already asked for this theme

    def getNextTheme_Clip_ValidSet(self) -> List[Tuple[int, float]]:
        clipIdsAndNormalizedScores : List[Tuple[int, float]] = AgentUtils.cosineSim_clipsSelection(orgAttributesInterest=self.orgAttrInterest,
                                                                                                   orgCategoriesInterested=self.orgCategoriesInterest,
                                                                                                   minQuestionsForClip = 1,  # self.settings.minQuestionsPerTheme,
                                                                                                   dataStore=self.dataStore,
                                                                                                   alreadySelectedClips=self.setOfAlreadySelectedClips,
                                                                                                   prevSelectedClipId = self.prevSelectedClipID)
        return clipIdsAndNormalizedScores

    # Get clip id for the next theme, the best according to score and state
    def getNextTheme_Clip(self, themeId):
        self.resetCurrentThemeState()

        clipIdsAndNormalizedScores: List[Tuple[int, float]] = self.getNextTheme_Clip_ValidSet()

        if len(clipIdsAndNormalizedScores) == 0:
            assert False, "I couldn't select a clip anymore for this theme"
            return None

        clipIdSelected = AgentUtils.selectRoulette(clipIdsAndNormalizedScores, elitismPercent=ELITISM_PERCENT_CLIPS)
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


    # Get next question under a theme when the clip was already selected
    def getNextTheme_Question(self):
        normalizedScores = AgentUtils.cosineSim_questionSelection(orgCategoriesInterest=self.orgCategoriesInterest,
                                                                  prevQuestionIdSelected=self.prevQuestionIdSelected,
                                                                    dataStore=self.dataStore,
                                                                    parentClipId=self.currentTheme_clipId,
                                                                    setOfAlreadySelectedQuestionsForClip=self.setOfAlreadySelectedQuestionsForClip,
                                                                    surveySettings = self.settings)

        if len(normalizedScores) == 0:
            return None

        queId = AgentUtils.selectRoulette(normalizedScores, elitismPercent=ELITISM_PERCENT_QUESTIONS)
        return queId

    # The public function to get a question under a theme
    def getNextQuestionId(self):
        # If we already selected the target number of questions to show, then that's it !
        if self.currentTheme_numQuestionsAsked >= self.currentTheme_targetNumQuestionsToSelect:
            return INVALID_ID

        queId = self.getNextTheme_Question()
        if queId == None:
            pass
            #assert self.currentTheme_numQuestionsAsked >= self.settings.minQuestionsPerTheme, \
            #    f"I couldn't select a question anymore for this theme and i should have selected a minimum of {self.settings.minQuestionsPerTheme} but i selected only {self.currentTheme_numQuestionsAsked}.Check the graph of question dependencies, add more"
            return INVALID_ID

        self.__onQuestionSelectedForClip(questionId=queId)
        return queId

    # Commit a clip selection for a given theme id
    def onClipSelectedForTheme(self, themeId: any, clipId: any):
        self.currentTheme_clipId = clipId
        self.setOfAlreadySelectedClips.add(clipId)
        self.prevQuestionIdSelected = NO_DEPENDENCY_CONST
        self.prevSelectedClipID = clipId

        if clipId not in self.setOfAlreadySelectedQuestionsForClip:
            self.setOfAlreadySelectedQuestionsForClip[clipId] = set()

    # Commit a question for a given theme id and question
    def __onQuestionSelectedForClip(self, questionId : any):
        if not self.currentTheme_clipId in self.setOfAlreadySelectedQuestionsForClip:
            self.setOfAlreadySelectedQuestionsForClip[self.currentTheme_clipId] = set()
        self.setOfAlreadySelectedQuestionsForClip[self.currentTheme_clipId].add(questionId)
        self.prevQuestionIdSelected = questionId
        self.currentTheme_numQuestionsAsked += 1

    # Assign a probability score for the agent to be in the given cluster specified as parameter
    def scoreClusters(self, clustersSpec : ManualClustersSpec) -> ManualClusterScores:
        numClusters = len(clustersSpec.clusters)
        scoresRes = ManualClusterScores(1, numClusters)
        scoresRes.result[0][0] = self.getAgentId()

        #  Compute the current score for each cluster
        res : UserScoresSurveyResults = SurveyResponses.scoreUserSurveyToClustersSpec(self.getAgentId(),
                                                          userDeviationsByCategory=self.agentDeviations_byCategory_raw,
                                                          userAttributesDeviations= self.attributesDeviations_raw,
                                                            userAttributesPerCategoryDeviations=self.attributesDeviations_byCategory_raw,
                                                          userResponse=None,
                                                          clustersSpec=clustersSpec)


        # DEMO DEBUG
        #SurveyResponses.outputSurveyResponses(res, outPrefixPath=os.path.join("dataout", "surveysDemo"))

        # Some sanity checks.
        assert len(res.outProbabilityPerCluster_normalized) == len(clustersSpec)
        sumProbs = np.sum(res.outProbabilityPerCluster_normalized)
        assert (sumProbs - 1.0) < 0.0001, "The normalized final probability is not correct"

        # Fill in the scores then finalize
        for clusterIndex in range(numClusters):
            scoresRes.result[0][1 + clusterIndex] = res.outProbabilityPerCluster_normalized[clusterIndex]

        scoresRes.finalize()
        return scoresRes, res

    # Callback executed in the beggining of a survey
    def beginSurvey(self, settings : SurveyBuildSettings):
        super().beginSurvey(settings)

        if self.clustersSpec != None:
            # Choose a default person behind this survey
            self.chosenAgentClusterIndex = np.random.choice(
                len(self.clustersSpec)) if DebugSettings.DEBUG_FIXED_CLUSTER is None else DebugSettings.DEBUG_FIXED_CLUSTER
            if self.verbose:
                if self.chosenAgentClusterIndex is not None: # This is used only when doing
                    print(f"For this Survey, in tests, the agent is behind Cluster {self.chosenAgentClusterIndex}")
                #print(f"Current cluster probs: {self.currentProbabilityPerCluster}")

    def endSurvey(self, supressOutput : bool, outStats : AgentSurveyStats):
        if outStats:
            clusterPredicted = np.argmax(self.currentProbabilityPerCluster)
            clusterPredicted_prob = self.currentProbabilityPerCluster[clusterPredicted]
            clusterReal_prob = self.currentProbabilityPerCluster[self.chosenAgentClusterIndex]
            isCorrect = clusterPredicted == self.chosenAgentClusterIndex

            outStats.addNewSurveyStats(isSurveyCorrect=isCorrect, groundTruth_prob=clusterReal_prob, predicted_prob=clusterPredicted_prob)
            if not supressOutput:
                print(f"End survey. C:{isCorrect} Pred:{clusterPredicted} Correct:{self.chosenAgentClusterIndex} Probs={clusterPredicted_prob:.2f} vs {clusterReal_prob:.2f}; Stats:{outStats}")

            self.chosenAgentClusterIndex = None

        super().endSurvey(supressOutput, outStats)

    def setCurrentQuestionAnswer(self, answerValue):
        #  Step 1: Append the new answers to the local agent hjistory
        questionData = self.dataStore.questions_byId[self.prevQuestionIdSelected]
        rawDeviation = abs(answerValue - questionData.baseline)
        biasedDeviation = rawDeviation # TODO Ciprian , after proto: bring the algorithms here from DataDefinitions regarding biases !

        # questionData.category
        questionDeviations = QuestionResponseDeviation()
        questionDeviations.rawDeviation = rawDeviation
        questionDeviations.biasedDeviation = rawDeviation
        questionDeviations.questionId = self.prevQuestionIdSelected
        questionDeviations.parentClipId = self.currentTheme_clipId

        self.historyOfQuestionsAndDeviations.append(questionDeviations)
        if questionData.category not in self.questionDeviationsByCat:
            self.questionDeviationsByCat[questionData.category] = []
        self.questionDeviationsByCat[questionData.category].append(questionDeviations)

        # Step 2: update the local statistics for deviations pe attributes and categories
        self.updateCurrentAgentStates()

        # Step 3: update the local probabilities per cluster for this agent
        if self.clustersSpec is None:
            self.currentProbabilityPerCluster = [1.0] # Single cluster simulation if none...
        else:
            scoreClusterRes : ManualClusterScores = None
            detailedScores : UserScoresSurveyResults = None
            scoreClusterRes, detailedScores = self.scoreClusters(self.clustersSpec)
            self.currentProbabilityPerCluster = scoreClusterRes.result[0][1:] # First (and single user), first column is the user id rest are probs
            assert (len(self.currentProbabilityPerCluster) == len(self.clustersSpec))

            if self.verbose:
                prefixOutputFile = f"dataout/AgentDebugging/Agent_{self.getAgentId()}_step_{self.getCurrentQuestionIndex()}"
                clusterPlortterHelper = ClusterPlotterHelp(clustersSpec=self.clustersSpec,
                                                           dataStore=self.dataStore,
                                                           results=detailedScores,
                                                           userDeviationsByCategory=self.agentDeviations_byCategory_raw,
                                                           userAttributesDeviations=self.attributesDeviations_raw,
                                                           outFile_clusterScoreCurves=f"{prefixOutputFile}_clusterScores",
                                                           outFile_contributions=f"{prefixOutputFile}_clusterContribs")


        self.historyOfClusterProbabilityValues.append(self.currentProbabilityPerCluster)
        assert (len(self.historyOfClusterProbabilityValues) - 1 == len(self.historyOfQuestionsAndDeviations)) # We have always the 0-init probabilities in the history of scores

