import AgentBasic
from AgentAbstract import *
import random
from datetime import datetime
import AgentPathfinding
from AgentUtils import *
from DataDefinitions import *

USE_DETERMINISTIC_SURVEY_GENERATION = False  # Put it true to avoid randoms and debug properly


def test():
    dataStore = loadDataStore()


    ##### STEP 0 : Define the organization interest - NEEDED by AgentBasic only !, then the clusters NEEDED for pathfdining agent
    # Set org interests - For comparison purposes we put here the union of all attributes and categories used in the clusters definitions
    orgAttributesSet = {"Leadership": 1.0, "Sexual Harassment": 1.0, "Mental Health": 1.0, "Team Interaction":1.0, "Personal Boundaries":1.0}
    orgCategoriesSet = {"Awareness": 1.0, "Sensitivity": 1.0}
    organizationInterests = AgentUtils.buildOrganizationSettingsByDicts(attrsAndScores=orgAttributesSet, catAndScores=orgCategoriesSet, dataStore=dataStore)

    if USE_DETERMINISTIC_SURVEY_GENERATION:
        random.seed(0)
    else:
        random.seed(datetime.now())

    # Set a pathfinding agent
    #----------------------------------
    emptyOrganizationInterests = OrganizationInterestSettings() # This must be empty since we already define the cloud of clusters and  they contain the interesting features inside already.

    manualClustersSpec = get_ManualClusteringSpecDemo()

    VERBOSE = False # TODO Would be nice to use the logging library in the future, but too lazy now
    DO_SANITY_CHECKS = True # Snould we do sanity checks for all reviews ?
    NUM_AUTOMATED_TESTS = 100

    ##### STEP 1: Create the agents and the survey factories that use them
    agentBasicInstance = AgentBasic.AgentBasic(orgInterest=organizationInterests, dataStore=dataStore, verbose=VERBOSE, clustersSpec=manualClustersSpec)
    pathfindingAgentInstance = AgentPathfinding.AgentPathfinding(orgInterest=emptyOrganizationInterests, dataStore=dataStore,
                                                  clustersSpec=manualClustersSpec, verbose=VERBOSE)

    surveySettings = SurveyBuildSettings(numThemes=10, minQuestionsPerTheme=1, maxQuestionsPerTheme=4,
                                         isCategoriesScoringEnabled=True, forceMaximizeNumQuestionsPerTheme=True)
    surveyFactory_agentPathfinding = SurveyFactory(dataStore=dataStore, agent=pathfindingAgentInstance, settings=surveySettings)
    surveyFactory_agentBasicTest = SurveyFactory(dataStore=dataStore, agent=agentBasicInstance, settings=surveySettings)

    ##### Step 2: Run multiple tests , see the doc !
    # Test 0: Create a survey for demo and show output
    #========================
    print("@@@@@ Running a real kind of demo to show a survey with output !")
    pathfindingAgentInstance_realdemo = AgentPathfinding.AgentPathfinding(orgInterest=emptyOrganizationInterests, dataStore=dataStore,
                                                                            clustersSpec=manualClustersSpec, verbose=True)
    pathfindingAgentInstance_realdemo.resetState()
    surveyFactory_agentPathfinding_realdemo = SurveyFactory(dataStore=dataStore, agent=pathfindingAgentInstance_realdemo, settings=surveySettings)
    outputSurvey_pathfinding = surveyFactory_agentPathfinding_realdemo.buildSurveyDemo(supressOutput=False, outSurveyStats=None,
                                                                                        doSanityChecks=DO_SANITY_CHECKS,
                                                                                        useLogFilePath="dataout/AgentDebugging")
    #=========================


    # Test 1: Create a functional todolist to see performance of the agent, random cluster individual selected
    # ========================
    print("@@@@@ PERFORMANCE TEST - Pathfinding agentStarting a bunch of todolist surveys ")
    if DebugSettings.DEBUG_FIXED_SEED is not None:
        random.seed(DebugSettings.DEBUG_FIXED_SEED)
        np.random.seed(DebugSettings.DEBUG_FIXED_SEED)
        # TODO set tensorflow as well..

    DebugSettings.DEBUG_KNOWING_PERSON_BEHAVIOR = False
    DebugSettings.DEBUG_FIXED_CLUSTER = None
    stats_pathfindingAgent = AgentSurveyStats()
    stats_basicAgent = AgentSurveyStats()
    for i in range(NUM_AUTOMATED_TESTS):
        if i % 30 == 0:
            print(f"##### Starting suvery todolist {i}")

        # Run todolist with patfinding agent
        pathfindingAgentInstance.resetState()
        outputSurvey_pathfinding = surveyFactory_agentPathfinding.buildSurveyDemo(supressOutput=not VERBOSE, outSurveyStats=stats_pathfindingAgent, doSanityChecks=DO_SANITY_CHECKS)

        # Same with the basic agent
        agentBasicInstance.resetState()
        outputSurvey_basic = surveyFactory_agentBasicTest.buildSurveyDemo(supressOutput=not VERBOSE, outSurveyStats=stats_basicAgent, doSanityChecks=DO_SANITY_CHECKS)
    print("Final TEST stats result for PATHFINDING AGENT: ", stats_pathfindingAgent)
    print("Final TEST stats result for BASIC AGENT: ", stats_basicAgent)
    # ========================
    # Test 1.2: Compare with base agent


    # Test 2: Create a functional todolist to see performance of the agent, random cluster individual selected
    print("@@@@@ Prove capabilities - Pathfinding agentsStarting a bunch of todolist surveys to see the provness capability of each cluster using the actual questions ")
    clustersToTest = list(range(len(manualClustersSpec)))
    for clusterIndex in clustersToTest:
        print(f"### Testing cluster {clusterIndex}")
        if DebugSettings.DEBUG_FIXED_SEED is not None:
            random.seed(DebugSettings.DEBUG_FIXED_SEED)
        np.random.seed(DebugSettings.DEBUG_FIXED_SEED)
        # TODO set tensorflow as well..

        DebugSettings.DEBUG_KNOWING_PERSON_BEHAVIOR = True
        DebugSettings.DEBUG_FIXED_CLUSTER = clusterIndex
        stats = AgentSurveyStats()
        for i in range(NUM_AUTOMATED_TESTS):
            if i % 30 == 0:
                print(f"##### Starting suvery todolist {i}")

            pathfindingAgentInstance.resetState()
            outputSurvey_pathfinding = surveyFactory_agentPathfinding.buildSurveyDemo(supressOutput=not VERBOSE, outSurveyStats=stats, doSanityChecks=DO_SANITY_CHECKS)
        print(f"Final stats for proving cluster {clusterIndex} result: ", stats)

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


if __name__ == "__main__":
    test()
