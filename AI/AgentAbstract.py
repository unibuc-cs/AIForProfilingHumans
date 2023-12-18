import sys

from DataDefinitions import *
import random
from typing import Dict, List, Tuple, Set
from DataDefinitions import Theme, INVALID_ID, QuestionResponseDeviation


class SurveyBuildSettings:
    def __init__(self,  numThemes : int,
                        maxQuestionsPerTheme : int,
                        minQuestionsPerTheme : int,
                        isCategoriesScoringEnabled : bool,
                        forceMaximizeNumQuestionsPerTheme : bool):
        self.numThemes = numThemes
        self.maxQuestionsPerTheme = maxQuestionsPerTheme
        self.minQuestionsPerTheme = minQuestionsPerTheme
        self.isCategoriesScoringEnabled = True              # If categories scoring is enabled
        self.forceMaximizeNumQuestionsPerTheme = False


# A collecting stats for getting feedback for how agents performs during interview tests
class AgentSurveyStats:
    def __init__(self):
        self.resetStats()

    def resetStats(self):
        self.numSurveys = 0.0  # Num surveys being taken
        self.correctSurveys = 0.0 # How many times the agent was correct in the classification
        self.lossTotal = 0.0  # When doing wrong, this will compute the linear absolute difference between the ground truth and the real cluster

    # Called when a new survey ended and we have new data to collect
    def addNewSurveyStats(self, isSurveyCorrect : bool, groundTruth_prob, predicted_prob ):
        self.numSurveys += 1
        if isSurveyCorrect == True:
            self.correctSurveys += 1

        self.lossTotal += abs(groundTruth_prob - predicted_prob)

    def __str__(self):
        if self.numSurveys == 0:
            return "No survey was done on this todolist"
        currentAvgLoss = self.lossTotal / self.numSurveys if self.numSurveys != 0 else 0
        outStr = f"Correctness={self.correctSurveys/self.numSurveys if self.numSurveys > 0 else 0:.2f} {self.correctSurveys}/{self.numSurveys:.2f}. Loss={currentAvgLoss:.5f}"
        return outStr

    def printStats(self):
        StatsStr = self.__str__()
        print(StatsStr)

# Abstract agent class. All agents should derive from  this
class AbstractAgent:
    def __init__(self, orgInterest : OrganizationInterestSettings, dataStore : DataStore, verbose : bool = False):
        self.orgAttrInterest : OrgAttributesSet = orgInterest.attributesInterestedIn
        self.orgCategoriesInterest : OrgCategoriesSet = orgInterest.categoriesInterestedIn
        self.dataStore = dataStore
        self.settings = None
        self.verbose = verbose

        self.resetState()

    def getAgentId(self):
        raise NotImplementedError
        return INVALID_AGENT_ID

    def resetState(self):
        # This set  stores the set of selected clips already
        # clip id is in setOfAlreadySelectedClips if it was already selected
        self.setOfAlreadySelectedClips : Set[any] = set()

        # The previous clip selected
        self.prevSelectedClipID : int = NO_DEPENDENCY_CONST

        # This dict stores for each selected clip, which questions where already selected
        # setOfAlreadySelectedQuestionsForClip[]
        self.setOfAlreadySelectedQuestionsForClip : Dict[any, Set[int]] = {}

        # The list of questions asked and the answers deviations following organization baselines
        self.historyOfQuestionsAndDeviations : List[QuestionResponseDeviation] = []
        self.questionDeviationsByCat: Dict[any, List[QuestionResponseDeviation]] = {} # Question deviations asked until now indexed by the category id
        for cat in self.dataStore.categoriesList:
            self.questionDeviationsByCat[cat] = []

    def getCurrentQuestionIndex(self):
        return len(self.historyOfQuestionsAndDeviations)

    # Init some stuff at the beggining of the survey
    def beginSurvey(self, settings : SurveyBuildSettings):
        self.settings = settings

    def endSurvey(self, supressOutput:bool, outAgentSurveyStats:AgentSurveyStats):
        pass

    # Get clip for the next theme
    def getNextTheme_Clip(self, themeId : int):
        raise NotImplementedError()

    # Get next question under a theme when the clip was already selected
    def getNextQuestionId(self):
        raise NotImplementedError()

    # Sets the answer value for the current question being asked
    def setCurrentQuestionAnswer(self, answer):
        raise NotImplementedError()

    # Get next question under a theme when the clip was already selected
    def __getNextTheme_Question(self, parentClipId: any):
        raise NotImplementedError()

    def __onQuestionSelectedForClip(self, questionId : any):
        raise NotImplementedError()

    def __onClipSelectedForTheme(self, themeId: any, clipId: any):
        raise NotImplementedError()


class SurveyFactory:
    def __init__(self, dataStore: DataStore, agent: AbstractAgent, settings: SurveyBuildSettings):
        self.settings = settings
        self.agent = agent
        self.dataStore = dataStore

    # Builds a survey by pushing an agent and some parameters explained in the doc
    def buildSurveyDemo(self, supressOutput=False, outSurveyStats : AgentSurveyStats=None,
                        doSanityChecks=False, useLogFilePath=None) -> Survey: # TODO agent as abstract class type strong
        dataStore = self.agent.dataStore
        self.supressOutput = supressOutput # TODO: use logging library !

        # We build this survey as a reference / bug purposes. Probably not necessarily in production environment
        outSurvey = Survey()

        # Check if we need to redirect output log to a file
        self.logFileRedirect = None
        if self.supressOutput is False:
            self.originalStdout = sys.stdout # Save a reference to the original stdout
            if useLogFilePath :
                redirectFilePath = os.path.join(useLogFilePath, f"_Agent{self.agent.getAgentId()}_log.txt")
                self.logFileRedirect = open(redirectFilePath, "w")
                sys.stdout = self.logFileRedirect

        # Inform the agent that survey is starting
        if self.supressOutput is False:
            print("# Starting survey !")
        self.agent.beginSurvey(self.settings)

        for themeId in range(self.settings.numThemes):
            theme = Theme(self.dataStore.attributesFlattened)
            theme.id = themeId

            # Select a clip
            clipIdSelected = self.agent.getNextTheme_Clip(themeId)

            if self.supressOutput is False:
                print(f"============== For theme {themeId} agent selected clip {clipIdSelected} =============== ")

            # Select questions one by one using the agent and store them locally for history tracking
            # Select the next question id. If this happens to be INVALID_ID, it means that this is the end of the theme
            questionsIdsAsked = []
            nextQuestionId = None
            while nextQuestionId is None or nextQuestionId != INVALID_ID:
                nextQuestionId = self.agent.getNextQuestionId()

                if nextQuestionId == INVALID_ID or nextQuestionId == None:
                    continue

                # Simulate a user response here to this question to give back to the agent
                answer = self.agent.simulateUserAnswer(nextQuestionId)
                self.agent.setCurrentQuestionAnswer(answer) # VERY IMPORTANT TO CALL THIS also !

                questionsIdsAsked.append(nextQuestionId)

                if self.supressOutput is False:
                    print(f"Agent selected question id {nextQuestionId}")
                    print(f"Current cluster probs: {self.agent.currentProbabilityPerCluster}")

            # Fill the theme in the history tracking output - maybe not needed in production but very usefull for debugging and data science metrics purposes
            theme.clip = dataStore.clips[clipIdSelected]
            theme.questions = [None] * len(questionsIdsAsked)

            for questionIter, questionId in enumerate(questionsIdsAsked):
                theme.questions[questionIter] = dataStore.questions_byId[questionId]

            outSurvey.themes.append(theme)

        assert len(outSurvey.themes) == self.settings.numThemes, "Incorrect number of themes added to the survery!"

        self.agent.endSurvey(self.supressOutput, outSurveyStats)

        if self.supressOutput is False:
            print("# Ending survey !")
            if self.logFileRedirect is not None:
                # flush.close the file
                self.logFileRedirect.flush()
                self.logFileRedirect.close()

                # Restore back the stdout
                sys.stdout = self.originalStdout


        if doSanityChecks:
            isCorrect = outSurvey.doSanityChecks(dataStore, self.settings)

        return outSurvey








