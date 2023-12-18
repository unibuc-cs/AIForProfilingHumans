from AI.DataDefinitions import *
from typing import Dict, List, Tuple, Set
from AI.AgentAbstract import *
import random

# Some utility for agents
from AgentAbstract import SurveyBuildSettings


class AgentUtils:
    # Currently just a simple DAG dependency
    @staticmethod
    def checkClipsOrderCompatibility(dataStore : DataStore, proposedClipId : any, prevSelectedClipId : any) -> bool:
        if prevSelectedClipId == NO_DEPENDENCY_CONST:
            return True

        assert proposedClipId != NO_DEPENDENCY_CONST

        proposedClipInstance = dataStore.clips[proposedClipId]
        prevClipInstance = dataStore.clips[prevSelectedClipId]

        # if no dependencies list, then it means that it's clean
        if len(proposedClipInstance.setOfClipIdsDependencies) == 0:
            return True

        # Prev clip should NOT appear in the dependencies
        if proposedClipInstance.dependencyType == ClipDependencyType.DEP_EXCLUDING_PREVIOUS:
            return prevSelectedClipId not in proposedClipInstance.setOfClipIdsDependencies
        # Prev clip SHOULD appear in the dependencies list
        elif proposedClipInstance.dependencyType == ClipDependencyType.DEP_PREVIOUS:
            return prevSelectedClipId in proposedClipInstance.setOfClipIdsDependencies
        else:
            raise NotImplementedError("Unknown case please solve")

        # TODO check
        return True

    # Computes the raw score value between a clip and organization attributes interest, knowing the already selected clips also..
    # Returns a list of [(clip id, score)] sorted by score
    @staticmethod
    def cosineSim_clipsSelection(orgAttributesInterest,
                                 orgCategoriesInterested,
                                 minQuestionsForClip : int,
                                 dataStore : DataStore,
                                 alreadySelectedClips : Set[any],
                                 prevSelectedClipId : int) -> List[Tuple[int, float]]:
        allClipsAvailable : Dict[any, Clip] = dataStore.clips

        # The if/else is because some agents send directly the dictionaries but others send only the wrapper data structure around these dictionaries !
        attributesInterested = orgAttributesInterest.attributesInterested #if not isinstance(orgAttributesInterest,Dict) else orgAttributesInterest
        categoriesInterested = orgCategoriesInterested.categoriesInterested #if not isinstance(orgCategoriesInterested,Dict) else orgCategoriesInterested

        outputList : List[Tuple[int, float]] = []
        for _, clip in allClipsAvailable.items():
            if clip.id in alreadySelectedClips:
                continue

            if AgentUtils.checkClipsOrderCompatibility(dataStore=dataStore, proposedClipId=clip.id, prevSelectedClipId=prevSelectedClipId) is False:
                continue

            # Does the clip contain the minimum number of questions ?
            if (clip.id not in dataStore.questions_byClip) or len(dataStore.questions_byClip[clip.id]) < minQuestionsForClip:
                continue

            # Take the attributes score
            clipAttributes = clip.attributes.scores
            totalScoreAttributes = 0.0
            for attrKey in attributesInterested.keys:
                totalScoreAttributes += (attributesInterested.scores[attrKey]  * clipAttributes[attrKey]) # TODO 1: Store these as numpy arrays for speed !!

            # Take the categories score
            questionsOnThisClip = dataStore.questions_byClip[clip.id]
            # A simple algorithm to score all. Need to do on the tree
            # TODO
            #========
            totalCategoriesScore = 0.0
            for que in questionsOnThisClip:
                for categoryKey in categoriesInterested.keys():
                    totalCategoriesScore += que.categoriesArray.scores[categoryKey]
            # Normalize by the number of questions otherwise it will be unfair for attributes
            if len(questionsOnThisClip) > 0:
                totalCategoriesScore /= len(questionsOnThisClip)
            #========

            totalScore = totalScoreAttributes + totalCategoriesScore

            outputList.append((clip.id, totalScore))
        outputList = sorted(outputList, key=lambda x : x[1], reverse=True)

        outputList = AgentUtils.normalizeCosineScores(outputList)
        return outputList

    # Computes the raw score value between a question (based on its category) and organization interest in questions,
    # knowing the parent clipId selected
    @staticmethod
    def cosineSim_questionSelection(parentClipId : any,
                                    prevQuestionIdSelected,
                                    orgCategoriesInterest,
                                    dataStore : DataStore,
                                    setOfAlreadySelectedQuestionsForClip : Dict[any, Set[int]],
                                    surveySettings : SurveyBuildSettings) -> List[Tuple[int, float]]:

        # The if/else is because some agents send directly the dictionaries but others send only the wrapper data structure around these dictionaries !
        categoriesInterested = orgCategoriesInterest.categoriesInterested #if not isinstance(orgCategoriesInterest, Dict) else orgCategoriesInterest
        questionsAvailableForClip = dataStore.questions_byClip[parentClipId]
        outputList : List[Tuple[int, float]] = []

        # Establish the set of valid follow up questions are: the ones that have no dependency, the one allowed by the DAG  -  MINUS - the ones already asked
        setOfAllValidFollowupQuestions = set()
        if NO_DEPENDENCY_CONST in dataStore.acceptableQuestionsIdAfter[parentClipId]:
            for qId in dataStore.acceptableQuestionsIdAfter[parentClipId][NO_DEPENDENCY_CONST]:
                setOfAllValidFollowupQuestions.add(qId)

        if prevQuestionIdSelected in dataStore.acceptableQuestionsIdAfter[parentClipId]:
            for qId in dataStore.acceptableQuestionsIdAfter[parentClipId][prevQuestionIdSelected]:
                setOfAllValidFollowupQuestions.add(qId)

        setOfAllValidFollowupQuestions.difference_update(setOfAlreadySelectedQuestionsForClip[parentClipId])

        # Take all valid questions and score them
        for questionId in setOfAllValidFollowupQuestions:
            # Take the question instance data
            question = dataStore.questions_byId[questionId]
            totalScore = 0.0

            # Add the score for categories first
            if surveySettings.isCategoriesScoringEnabled:
                for catInterestedKey in categoriesInterested:
                    totalScore += categoriesInterested[catInterestedKey] * question.categoriesArray.scores[catInterestedKey] # TODO 1: Store these as numpy arrays for speed !!
            else:
                totalScore = 1.0 # if categories are not enabled, maximum score for everyone.

            outputList.append((question.id, totalScore))

        if len(outputList) > 0:
            AgentUtils.normalizeCosineScores(outputList)
            outputList = sorted(outputList, key=lambda x : x[1], reverse=True)
        return outputList

    # Given a list of (key, score), normalize all to represent probabilities between 0-1 normalized for all scores
    @staticmethod
    def normalizeCosineScores(cosineScores : List[Tuple[any, float]]):
        cosineNormalizedScores = []
        totalScores = 0.0
        for item in cosineScores:
            totalScores += item[1]

        if totalScores > 0.0:
            for item in cosineScores:
                cosineNormalizedScores.append((item[0], item[1] / totalScores))
        else:
            cosineNormalizedScores = cosineScores


        return cosineNormalizedScores

    @staticmethod
    def selectRoulette(normalizedSortedScores : Tuple[any, float], elitismPercent = 1.0):
        elitismSelection = normalizedSortedScores[:max(1, int(len(normalizedSortedScores)*elitismPercent))]
        eliteSumScores = sum(map(lambda x : x[1] , elitismSelection))
        assert eliteSumScores <= 1.0

        rvalue = random.random() * eliteSumScores
        # print(rvalue)
        currSum = 0.0
        for _, value in enumerate(normalizedSortedScores):
            currSum += value[1]
            if currSum >= rvalue:
                return value[0]

        return normalizedSortedScores[0][0]
        assert False, "I didn't returned as expected...."


    # Given two dictionaries containing attributes and their score, and another one categories and scores, this function builds the organization's interest data structure
    @staticmethod
    def buildOrganizationSettingsByDicts(attrsAndScores : Dict[any, float], catAndScores : Dict[any, float], dataStore):
        attributesInterestedIn = OrgAttributesSet(attrsAndScores, dataStore.attributesFlattened)
        categoriesInterestedIn = OrgCategoriesSet(catAndScores, dataStore.categoriesList)

        orgIntSettings = OrganizationInterestSettings()
        orgIntSettings.attributesInterestedIn = attributesInterestedIn
        orgIntSettings.categoriesInterestedIn = categoriesInterestedIn
        return orgIntSettings

    # Given a specification of clusters that share the same features, try to get a fictive interest settings that respects the cluster
    @staticmethod
    def createOrganizationInterestSettings_byIndividualClusterSpec(individualCluster: ManualSingleClusterSpec,
                                                                   dataStore: DataStore):
        # Fill in data from clusters
        categories_names = set()
        attributes_names = set()

        for feature in individualCluster.features:
            assert isinstance(feature, ManualClusterFeature)
            categories_names.add(feature.name)

            for attrInCat in feature.listOfAttributes:
                attributes_names.add(attrInCat)

        attrsAndScores = {attrName: 1.0 for attrName in attributes_names}
        catAndScores = {catName: 1.0 for catName in categories_names}

        orgIntSettings = AgentUtils.buildOrganizationSettingsByDicts(attrsAndScores=attrsAndScores,
                                                                     catAndScores=catAndScores, dataStore=dataStore)
        return orgIntSettings

