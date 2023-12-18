import os
import statistics
import pandas as pd
pd.options.mode.chained_assignment = None
from colorhash import ColorHash

from enum import Enum
from scipy import interpolate
from typing import Dict, List, Tuple, Set
import numpy as np
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import metrics
from sklearn.metrics import silhouette_score
import copy
import matplotlib.pyplot as plt
import scipy.stats
import ast

import matplotlib.pyplot as plt

from memory_profiler import profile
from memory_profiler import memory_usage
from memory_profiler import LogFile

MIN_AMBIGUITY = 1
MAX_AMBIGUITY = 7
AVG_AMBIGUITY = (MIN_AMBIGUITY + MAX_AMBIGUITY) / 2
MIN_SEVERITY = 1
MAX_SEVERITY = 7
AVG_SEVERITY = (MIN_SEVERITY + MAX_SEVERITY) / 2
MIN_BASELINE = 1
MAX_BASELINE = 7
AVG_BASELINE = (MIN_BASELINE + MAX_BASELINE) / 2
MIN_QUESTION_RESPONSE = 1
MAX_QUESTION_RESPONSE = 7

UNSET_CATEGORY = -1
INVALID_ID = "None"
NO_DEPENDENCY_CONST = -1

# Some debug constants for testing purposes
class DebugSettings:
    DEBUG_PERFECT_ANSWERS = True  # No deviation in answering
    DEBUG_KNOWING_PERSON_BEHAVIOR = False  # Pretending to know the user behavior before
    DEBUG_FIXED_SEED = None # 10
    DEBUG_FIXED_CLUSTER = None # The fixed agent cluster that we want to prove

# Randomness parameters in agent decisions
ELITISM_PERCENT_QUESTIONS = 0.35 # When evaluating the bunch of sorted by score questions (and clips), use only these percent part for exploration with selection roulette instead of exploring full list
ELITISM_PERCENT_CLIPS = 0.25

# Some playground agent ids to know which one is
PATHFINDING_AGENT_ID = 9999999
BASE_AGENT_ID = 1111111

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

class ClipDependencyType(Enum):
    DEP_PREVIOUS = 1   # This means a dependence on a set
    DEP_EXCLUDING_PREVIOUS = 2  # This means a dependence EXCLUDING a set

# Read global values: TODO maybe we should read the above one in the same way from a csv file
class ReduceOpType(Enum):
    MIN = 1
    MAX = 2
    MEAN = 3
    MEDIAN = 4


STATS_KEYS = [ReduceOpType.MIN, ReduceOpType.MAX, ReduceOpType.MEAN, ReduceOpType.MEDIAN]

def mysign(x):
    return 1.0 if x >= 0.0 else -1.0

# THESE are used for MANUAL clustering specification algorithms
#========================================================================
class ManualClusterFeature:
    # Mean and dev are specifications for a gaussian distribution probability distribution
    def __init__(self, catName : str, listOfAttributes : List[str], mean : float, dev : float):
        self.name = catName  # This is one the category of questions interesting for this cluster
        self.mean = mean
        self.dev = dev

        self.listOfAttributes = listOfAttributes # This is the attributes interesting for this question category and cluster specifically

        self.gaussianDist = scipy.stats.norm(self.mean, self.dev)

    # Given a value, compute the PDF of the gaussian using the recorded mean and deviation
    def scoreDeviation(self, val) -> float:
        return self.gaussianDist.pdf(val)

    def sampleValue(self) -> float:
        if DebugSettings.DEBUG_PERFECT_ANSWERS == False:
            return self.gaussianDist.rvs(1)[0]
        else:
            return self.mean

class ManualSingleClusterSpec:
    # Input are specifications for categories and attributes
    def __init__(self, features : List[ManualClusterFeature],  name):
        self.features = features
        self.numFeatures = len(features)
        self.centroidValue = self.__extractAsCentroidValue()
        self.name = name

        self.allAttributesListUsedInClusterSpec = set()
        for feature in self.features:
            for attr in feature.listOfAttributes:
                self.allAttributesListUsedInClusterSpec.add (attr)

    # Checks if the feature names and numbers are the same in both, such that these are comparable
    @staticmethod
    def areClustersCompatible(clusterA, clusterB) -> bool:
        if len(clusterA.features) != len(clusterB.features):
            return False

        for catIndex in range(len(clusterA.features)):
            # Check category feature names
            if clusterA.features[catIndex].name != clusterB.features[catIndex].name:
                return False

            # Check attributes features list per category
            if clusterA.features[catIndex].listOfAttributes != clusterB.features[catIndex].listOfAttributes:
                return False

        return True

    # This returns the centroid -  (means of the specified features)
    def __extractAsCentroidValue(self):
        # The convention is to put categories first then attributes ! Check the code in getResponsesFeaturesData function
        categories_mean = [cat.mean for cat in self.features]
        #attributes_mean = [attr.mean for attr in self.attributes_features]
        centroid_value = categories_mean #+ attributes_mean
        return centroid_value

    # Extract feature names, same convention as above,
    def getFeatureNames(self) -> List[str]:
        categories_names = [cat.name for cat in self.features]
        #attributes_names = [attr.name for attr in self.attributes_features]
        featureNames = categories_names #+ attributes_names
        return featureNames

    # For a given question data, returns the response according to the question's metadata and this cluster's metada
    # THis is how a user under this cluster would answer to the given question !
    def simulateUserAnswerToQuestion(self, question): #: QuestionForClip):
        # sample a value for each cat and attribute under this cluster definition
        # then do a weighted average by the scores in the question metadata
        # MIN_QUESTION_RESPONSE, MAX_QUESTION_RESPONSE)

        totalWeightedDeviation = 0.0
        totalWeightedScores = 0.0

        # Category scores
        for feature in self.features:
            sampledDeviation = feature.sampleValue() #abs(feature.sampleValue()  - feature.mean) - The features mean represent actually deviations not raw mean values !!!
            questionScoreForFeature = question.categoriesArray.scores.get(feature.name, 0.0)
            sampledDeviationWeighted = sampledDeviation * questionScoreForFeature

            totalWeightedDeviation += sampledDeviationWeighted
            totalWeightedScores += questionScoreForFeature

        deviationResponseMean = 0.0 if totalWeightedScores == 0.0 else (totalWeightedDeviation / totalWeightedScores)
        answer = question.baseline + deviationResponseMean
        return answer


# Note that this clusters must have the same features OR
# sharedFeaturesTest  must be  False if the clusters inside span along different set of features
class ManualClustersSpec:
    def __init__(self, clusters = List[ManualSingleClusterSpec], sharedFeaturesTest=True):
        self.clusters = clusters
        self.numClusters = len(self.clusters)
        self.maxNumFeatures = max([singleCluster.numFeatures for singleCluster in self.clusters])

        # Sanity check. Are all clusters given compatible ?
        if sharedFeaturesTest == True:
            assert len(self.clusters) > 0
            firstCluster = self.clusters[0]
            for cluster in self.clusters[1:]:
                if not ManualSingleClusterSpec.areClustersCompatible(firstCluster, cluster):
                    assert False, "The given clusters are not compatible !!!"

        # Ignore these when sharedFeaturesTest is False !
        # Extract the feature names used in this cluster list specification:
        self.featureNames = firstCluster.getFeatureNames() if sharedFeaturesTest == True else None

        # Get the centroids, this is reliable when using automatic clustering method behavior
        self.centroids = np.array([cluster.centroidValue for cluster in self.clusters])

        self.clustersNames = [cluster.name for cluster in self.clusters]

    def __len__(self):
        return len(self.clusters)

    def getClustersNames(self):
        return self.clustersNames


class ManualClusterScores:
    def __init__(self, numUsers, numClusters):
        # an numpy array basically of size 1+NumClusters. On first component we store the user id, on the rest the probability of being assigned on each cluster
        self.result = np.zeros(shape=(numUsers, 1 + numClusters), dtype=np.float32)

        # the mean probability for each cluster in this group
        self.meanProbabilityPerCluster = np.zeros(numClusters)

        # the type label - highest probability for each individual
        self.predictions = np.zeros(numUsers, dtype=np.int)

        # number of clusters
        self.numClusters = numClusters
        self.numUsers = numUsers

    def finalize(self):
        # Compute the mean probability of each cluster for this group
        for i in range(self.numClusters):
            self.meanProbabilityPerCluster[i] = np.mean(self.result[:, i + 1])

        for userIndex in range(self.numUsers):
            self.predictions[userIndex] = np.argmax(self.result[userIndex, 1:]) # 1:end, because on pos 0 we have the userId

        self.predictions = self.predictions
    # Write a CSV containing
    # - first row, userid =-1 always, showing the means of probabilities for each class inside the analyzed group
    # - second row to end: probability of each user in the group (e.g. either team or global)
    def writeStats(self, outputFilePath):
        import csv
        csv_columns = ['UserId']
        for i in range(self.numClusters):
            csv_columns.append(f'P of Cluster_{i}')
        csv_columns.append("Label")

        try:
            with open(outputFilePath, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()

                first_line = '-1,' + ','.join([str(x) for x in self.meanProbabilityPerCluster])
                csvfile.write(first_line)
                csvfile.write("\n")

                for idx, val in enumerate(self.result):
                    user_line = ','.join([str(x) for x in val])
                    user_line += "," + str(self.predictions[idx])
                    csvfile.writelines(user_line)
                    csvfile.write("\n")


        except IOError:
            print("I/O error")
#========================================================================

class CorrelationOptions:
    def __init__(self, itemsToShow, teamId, threshold=1.0, saveFigurePath = None,
                    questionFilteredId = None, userFilteredId = None, useBiasedValues=True):

        self.itemsToShow = itemsToShow   # This can be None. If it is none, then treshold must be something valid between
        self.teamId = teamId
        self.saveFigurePath = saveFigurePath
        self.threshold = threshold

        self.questionFilteredId = questionFilteredId
        self.userFilteredId = userFilteredId
        self.useBiasedValues = useBiasedValues

# Informations about a GMM trained model
class CustomGaussianMixtureMeta():
    def __init__(self):
        self.numClasses = None # The number of classes
        self.centroids = None # The centroid positions
        self.weights = None # weights of each gaussian
        self.covariances = None # The covariances (num_classes, nfeatures, nfeatures) array
        self.refToTrainingData = None # A reference to the training data, warning: might be garbage at some point

        #self.PCATransformUsed = None # The Principal Component ANalysis transform used to covert the original dataset

# Informations about predicting a data sample to a GMM model
class PredictionGaussianMixtureMeta():
    def __init__(self):
        self.numClasses = None
        self.numUsers = None # The number of users that give responses for this sample prediction
        self.predictions = None # An array that represents the class for each individual in the data sample
        self.meanConfidence = None # mean of the above value
        self.confidencePerUser = None # Posterior probability for the selected component (class) given the data
        self.percentPerClass = None # How many individuals are for each class category, in percentage
        self.numbersPerClass = None # Same as above, but in numbers not percentage.
        self.centroids = None # The  centroids on the GMM that this prediction is tested against

# This is for storing the 0-1 values of the Attributes scores, flattened on leafs
class AttributesArray:
    def __init__(self, attributesFlattened):
        self.keys = attributesFlattened
        self.scores = {key: 0 for key in self.keys}  # default is 0
    def __len__(self):
        return len(self.keys)

class  CategoriesArray:
    def __init__(self, categoriesList):
        self.keys = categoriesList
        self.scores = {key: 0 for key in self.keys}  # default is 0

# A collection  of statistics cross-surveys
class GlobalStatistics:
    def __init__(self):
        self.avgDeviationByQuestion : Dict[int, float] = {} # questionId ->  avg deviation

class DataStoreSettings:
    def __init__(self):
        self.useConstantBiasRemoval = False # If true, this will remove the constant biases identified from global means per each question. See technical doc

# Storing a question type following a clip
class QuestionForClip:
    def __init__(self, categoriesList):
        self.id = INVALID_ID
        self.parentClipId = INVALID_ID  #
        self.severity = (MIN_SEVERITY + MAX_SEVERITY) * 0.5
        self.ambiguity = (MIN_AMBIGUITY + MAX_AMBIGUITY) * 0.5
        self.baseline = (MIN_BASELINE + MAX_BASELINE) * 0.5
        self.category = UNSET_CATEGORY
        self.categoriesArray = CategoriesArray(categoriesList)

        self.parentClipAtttributes_Array = None
        self.parentClipAttributes_nonZeroKeys = None

    def setCategories(self, categoryIds : List[int]):
        self.category = categoryIds[0] # Currently there is a single category per question, but probably there will be more
        self.categoriesArray.scores[self.category] = 1.0

class Clip:
    def __init__(self, attributesFlattened):
        self.attributes = AttributesArray(attributesFlattened)  # Attributes of the clip being shown
        self.id = INVALID_ID  # The name of the clip, id, something..
        self.severity = 0
        self.dependencyType : ClipDependencyType = ClipDependencyType.DEP_PREVIOUS
        self.setOfClipIdsDependencies : Set[any] = set()

        # Cached !
        self.attributes_nonzeroKeys: List[str] = []

        def isCompatibleWithPrevClipId(self, prevClipId : any) ->bool:
            compatible = False
            if self.dependencyType == ClipDependencyType.DEP_PREVIOUS:
                compatible = prevClipId in self.setOfClipIdsDependencies
            else:
                compatible = prevClipId not in self.setOfClipIdsDependencies


# This is an instance  inside survey containing the clip selected and its question from the survey.
# These were selected from the datastore and its rules
class Theme:
    def __init__(self, attributesFlattened):
        self.id = INVALID_ID
        self.clip: Clip = Clip(attributesFlattened)  # The clip for this theme
        self.questions: List[QuestionForClip] = []  # The list of questions following the clip


class Survey:
    def __init__(self):
        self.themes: List[Theme] = []
        #self.questionsCategories: List[str] = []

    def getAttributesUsedInTheme(self, themeIndex):
        themeClip = self.themes[themeIndex].clip
        return themeClip.attributes_nonzeroKeys

    # This function performs basic sanity checks over the generated Survey
    def doSanityChecks(self,  dataStore, surveySettings): #dataStore: DataStore, settings: SurveyBuildSettings
        # Is the right number of themes asked ?
        assert len(self.themes) == surveySettings.numThemes, f"Incorrect number of themes. Requested {surveySettings.themes}, got {self.themes}"

        # Check if the number of questions asked per clip is within desired range
        # Check if clips and questions are different
        # Check order of questions
        setOfAskedClipIds = set()
        setOfAskedQuestionIds = set()
        for themeIndex, themeData in enumerate(self.themes):
            assert themeIndex == themeData.id, "Seems like indices do not correspond !"
            themeClipId = themeData.clip.id
            themeQuestionIds = [que.id for que in themeData.questions]

            assert (surveySettings.minQuestionsPerTheme <= len(themeQuestionIds) <= surveySettings.maxQuestionsPerTheme) or surveySettings.forceMaximizeNumQuestionsPerTheme, \
                        f"We requested to ask between {surveySettings.minQuestionsPerTheme} and {surveySettings.maxQuestionsPerTheme} questions " \
                        f"but got {len(themeQuestionIds)} questions"

            assert themeClipId not in setOfAskedClipIds, f"Clip {themeClipId} is selected again in the same interview !! {setOfAskedClipIds}"

            setOfAskedClipIds.add(themeClipId)

            acceptableQuestionsMap = dataStore.acceptableQuestionsIdAfter[themeClipId]
            prevQueId = NO_DEPENDENCY_CONST
            for queId in themeQuestionIds:
                assert queId not in setOfAskedQuestionIds, f"Questions {queId} is selected again in the same interview !! {setOfAskedQuestionIds}"

                acceptableQuestionsSet_no_dependency = acceptableQuestionsMap[NO_DEPENDENCY_CONST]
                acceptableQuestionsSet_precDep = acceptableQuestionsMap[prevQueId] if prevQueId in acceptableQuestionsMap else None

                assert queId in acceptableQuestionsSet_no_dependency or \
                       (acceptableQuestionsSet_precDep is not None and queId in acceptableQuestionsSet_precDep),\
                    f"The question asked {queId} is not in the set of acceptable " \
                                                        f"questions list {acceptableQuestionsSet_precDep} "
                prevQueId = queId
                setOfAskedQuestionIds.add(queId)

# The deviations computed for each response on a clip question
class QuestionResponseDeviation:
    def __init__(self):
        self.rawDeviation: float = None  # valueGivenByUser - baseline
        self.biasedDeviation: float = None  # The raw deviation biased by severity and other factors
        self.questionId: int = None
        self.parentClipId : int = None # The clip used to show this question on


class UserResponses:
    def __init__(self):
        # self.surveyId = INVALID_ID
        self.userId = INVALID_ID
        self.teamId = INVALID_ID

        # For each question index, what was the user response ? Following same indexing rule as
        # Survey.clipsAndQuestionsSequence, Themes and questions
        self.responses: List[List[int]] = []

        # CACHED THINGS THAT NEED CALLING FUNCTIONS TO STORE
        # --------------------------
        # For each theme, question per theme, store the Deviations value for the responses given above
        self.deviations: List[List[QuestionResponseDeviation]] = []

        # Deviations stats by category: First key is category id
        # Second key is STATS_KEY list
        self.deviationsStats_byCategory: Dict[any, Dict[ReduceOpType, QuestionResponseDeviation]] = {}

        # All deviation of responses by category. Basically the above is the aggregate of this
        self.allResponseDeviationsByCategory: Dict[any, List[QuestionResponseDeviation]] = {}

        # For a given attribute key, what is the deviation value ?
        self.attributesDeviations_raw: Dict[str, float] = {}
        self.attributesDeviations_biased: Dict[str, float] = {}
        # Same as above but now the first index is the category
        self.attributesDeviations_byCategory_raw: Dict[any, Dict[str, float]] = {}
        self.attributesDeviations_byCategory_biased: Dict[any, Dict[str, float]] = {}

        self.attributesDeviations_byQuestion_raw: Dict[any, Dict[str, float]] = {}
        self.attributesDeviations_byQuestion_biased: Dict[any, Dict[str, float]] = {}

    def resetCache(self):
        self.deviations = []
        self.deviationsStats_byCategory = {}
        self.allResponseDeviationsByCategory = {}
        self.attributesDeviations_raw = {}
        self.attributesDeviations_biased = {}
        self.attributesDeviations_byCategory_raw = {}
        self.attributesDeviations_byCategory_biased = {}
        self.attributesDeviations_byQuestion_raw = {}
        self.attributesDeviations_byQuestion_biased = {}

class UserScoresSurveyResults:
    # A. two list: one containing the probability for the user to be in each of the given clusters, the second
    # B. two dicts: for each cluster dicts mapping name of category/attribute to its individual score
    #  contains the normalized probabilities (sum in 0-1)
    def __init__(self, userId : any, clustersNames : List[str]):
        self.outProbabilityPerCluster = None
        self.outProbabilityPerCluster_normalized = None

        # For each of the clusters given in spec, in order compute the score contribution of each category and attribute
        self.detailedScoresPerCluster_categories: List[Dict[any, float]] = []
        self.detailedScoresPerCluster_attributes : List[Dict[any, float]] = []

        self.userId : any = userId
        self.clustersNames : List[str] = clustersNames
        self.numMerged = 1

    # Merges results from the other into self
    def merge(self, otherResult):
        # Add probabilities per cluster
        numClusters = len(self.outProbabilityPerCluster)
        assert numClusters == len(otherResult.outProbabilityPerCluster)
        assert self.outProbabilityPerCluster_normalized is None, "Normalize it with finalize after, please"
        for clusterIndex in range(len(self.outProbabilityPerCluster)):
            self.outProbabilityPerCluster[clusterIndex] += otherResult.outProbabilityPerCluster[clusterIndex]

        assert numClusters == len(otherResult.detailedScoresPerCluster_categories)
        assert numClusters == len(otherResult.detailedScoresPerCluster_attributes)
        for clusterIndex in range(numClusters):
            # Add results per categories
            # ------------------------------
            self_categoriesPerCluster : Dist[any, float] = self.detailedScoresPerCluster_categories[clusterIndex]
            other_categoriesPerCluster: Dist[any, float] = otherResult.detailedScoresPerCluster_categories[clusterIndex]

            for cat in self_categoriesPerCluster.keys():
                assert cat in other_categoriesPerCluster
                self_categoriesPerCluster[cat] += other_categoriesPerCluster[cat]
            # ------------------------------

            # Add results per attribute
            # ------------------------------
            self_attrPerCluster : Dist[any, float] = self.detailedScoresPerCluster_attributes[clusterIndex]
            other_attrPerCluster: Dist[any, float] = otherResult.detailedScoresPerCluster_attributes[clusterIndex]

            for attr in self_attrPerCluster.keys():
                assert attr in other_attrPerCluster
                self_attrPerCluster[attr] += other_attrPerCluster[attr]
            # ------------------------------


        self.numMerged += 1


    # Given a cluster index, returns two dictionaries giving the percent contribution of each feature (attribute or cat) to the cluster score
    def getContributionOfAttrAndCats(self, clusterIndex : int) -> Dict[str, Dict[any, float]]:
        assert clusterIndex < len(self.detailedScoresPerCluster_categories) \
            ,"Seems like you didn't compute these stats before"

        assert self.outProbabilityPerCluster[clusterIndex] > 0.0, "You have a problem with the final probability. IT should never be 0"

        constribs_attr = {}
        constribs_cat = {}
        clusterDetailed_perAttr = self.detailedScoresPerCluster_attributes[clusterIndex]
        clusterDetailed_perCat = self.detailedScoresPerCluster_categories[clusterIndex]

        for attrName,attrVal in clusterDetailed_perAttr.items():
            constribs_attr[attrName] = attrVal / self.outProbabilityPerCluster[clusterIndex]

        for catName,catVal in clusterDetailed_perCat.items():
            constribs_cat[catName] = catVal / self.outProbabilityPerCluster[clusterIndex]

        return {'attr' : constribs_attr, 'cat' : constribs_cat}

    # Postcomputations
    def finalize(self):
        numClusters = len(self.outProbabilityPerCluster)
        self.outProbabilityPerCluster_normalized = np.zeros(shape=self.outProbabilityPerCluster.shape,
                                                       dtype=self.outProbabilityPerCluster.dtype)

        # Step 0: make the mean if multiple values were actually merged
        if self.numMerged > 1:
            for i in range(numClusters):
                self.outProbabilityPerCluster[i] /= self.numMerged

            for clusterIndex in range(numClusters):
                # Mean results per categories
                # ------------------------------
                self_categoriesPerCluster : Dist[any, float] = self.detailedScoresPerCluster_categories[clusterIndex]

                for cat in self_categoriesPerCluster.keys():
                    self_categoriesPerCluster[cat] /= self.numMerged

                # Mean results per attributes
                # ------------------------------
                self_attributesPerCluster : Dist[any, float] = self.detailedScoresPerCluster_attributes[clusterIndex]

                for attr in self_attributesPerCluster.keys():
                    self_attributesPerCluster[attr] /= self.numMerged
        #--------------------


        # Step 1: do the normalization filling
        # Normalize the scores and get the final results out
        # Then normalize the probabilities such that they sum to 1
        sumProbs = np.sum(self.outProbabilityPerCluster)
        if sumProbs > 0:
            self.outProbabilityPerCluster_normalized = self.outProbabilityPerCluster / sumProbs
        else:
            assert False, "What the hell ?"
        sumProbs = np.sum(self.outProbabilityPerCluster_normalized)
        assert (sumProbs - 1.0) < 0.0001, "The normalized final probability is not correct"


class SurveyResponses:
    def __init__(self, categoriesList, globalStatistics : GlobalStatistics):
        self.categoriesList = categoriesList
        self.surveyTemplate: Survey = None
        self.globalStatistics = globalStatistics
        self.allKnownQuestionIds = set()

        # indexed by team id
        # allUsersResponses[teamId] = list of UserResponses for each individual from teamId
        self.allUsersResponses: Dict[int, List[UserResponses]] = {}
        self.userIdToTeamId : Dict[int, int] = {} # In this survey, what is the team of a user id
        self.invidualUserResponses : Dict[int, UserResponses] = {} # The reponses given by this user
        #self.userIdResponses : Dict[int, List[UserResponses]] = {} # Responses to one or more questionaires given by an user id
        self.totalUsersCount : int = None # The total number of users that responded to the survey
        self.totalUsersCountByTeam : Dict[int, int] = None # Same as above but indexed by team id

        # CACHED THINGS THAT NEED CALLING FUNCTIONS TO STORE
        # --------------------------
        # A list of flattened attributes that are really used in the template dataset
        self.allAttributesUsedInDataset = []

        # Create the output cache - DETAILED AT QUESTION LEVEL
        # For each team, for each theme, for each question, all responses aggregated as specified in the option
        # raw or biased
        self.aggregatedDeviations_PerTeam_raw: Dict[int, Dict[int, Dict[int, List[float]]]] = {}
        self.aggregatedDeviations_PerTeam_biased: Dict[int, Dict[int, Dict[int, List[float]]]] = {}

        self.aggregatedDeviations_PerTeam_raw: Dict[int, Dict[int, Dict[int, List[float]]]] = {}
        self.aggregatedDeviations_PerTeam_biased: Dict[int, Dict[int, Dict[int, List[float]]]] = {}

        # Same as above but not indexed by team
        self.aggregatedDeviations_Global_raw: Dict[int, Dict[int, List[float]]] = {}
        self.aggregatedDeviations_Global_biased: Dict[int, Dict[int, List[float]]] = {}

        # Create the output template - DETAILED AS AN OVERVIEW OF DEVIATIONS PER CATEGORY
        # These are global, key is category
        self.overviewDeviationsByCategory_Global_raw: Dict[int, List[float]] = {}
        self.overviewDeviationsByCategory_Global_biased: Dict[int, List[float]] = {}
        # These are per team indexed by category, then teamId
        self.overviewDeviationsByCategory_PerTeam_raw: Dict[int, Dict[int, List[float]]] = {}
        self.overviewDeviationsByCategory_PerTeam_biased: Dict[int, Dict[int, List[float]]] = {}

        # Global and per team output caches for attributes values
        self.overviewAttributes_Global_biased: Dict[str, List[float]] = {}
        self.overviewAttributes_Global_raw: Dict[str, List[float]] = {}
        self.overviewAttributes_PerTeam_biased: Dict[int, Dict[str, List[float]]] = {}  # First index is team as always
        self.overviewAttributes_PerTeam_raw: Dict[int, Dict[str, List[float]]] = {}

        # This is a statistics matrices composed with the following things:
        # On columns: [For all categories - min, max, mean, median of deviations, all attributes used in the dataset]
        # On rows: each row contains an entry for each user
        self.columnsInStatsTable = []  # All column names used in the data frame
        self.statsTable_raw: pd.DataFrame = None
        self.statsTable_perTeam_raw: Dict[int, pd.DataFrame] = {}
        self.statsTable_perQuestion_raw : Dict[any, pd.DataFrame] = {} # Same as above, at global level, but include only the data filtered by each question id !
        self.statsTable_perQuestionAndTeam_raw : Dict[any, Dict[any, pd.DataFrame]] = {}# Same as above, but at team level
        self.statsTable_perUser_raw: Dict[int, pd.DataFrame] = {} # Same as above but indexed by user id instead

        self.columnsInStatsTable = []  # All column names used in the data frame
        self.statsTable_biased: pd.DataFrame = None
        self.statsTable_perTeam_biased: Dict[int, pd.DataFrame] = {}
        self.statsTable_perQuestion_biased : Dict[any, pd.DataFrame] = {} # Same as above, at global level, but include only the data filtered by each question id !
        self.statsTable_perQuestionAndTeam_biased : Dict[any, Dict[any, pd.DataFrame]] = {}# Same as above, but at team level
        self.statsTable_perUser_biased: Dict[int, pd.DataFrame] = {} # Same as above but indexed by user id instead

    # Overview functions for getting deviations per questions.
    # biasedResults: True if you want to get biased results, False for raw
    # Returns a dict["questionId" : avg deviation]
    def getAvgQuestionDeviationsPerSurvey(self, biasedResults : bool) -> Dict[int, float]:
        dataToUse = self.aggregatedDeviations_Global_raw if biasedResults == False else self.aggregatedDeviations_Global_biased

        results : Dict[int, float] = {}
        counts : Dict[int, int] = {}
        themeKeys = self.aggregatedDeviations_Global_raw.keys()
        allQuestionsUsedKeys = set()
        for themeIndex, deviationsPerTheme in dataToUse.items():
            themeData = self.surveyTemplate.themes[themeIndex]
            for questionIndex, deviationsPerQuestion in deviationsPerTheme.items():
                questionId = themeData.questions[questionIndex].id

                if results.get(questionId) == None:
                    results[questionId] = 0.0
                    counts[questionId] = 0

                meanDevPerQuestion = np.abs(np.array(deviationsPerQuestion)).mean()

                results[questionId] += meanDevPerQuestion
                counts[questionId] += 1

        for questionId, count in counts.items():
            results[questionId] /= count

        return results

    def funcinterp_ambiguity(self, x):
        if x <= AVG_AMBIGUITY:
            return 1.0
        else:
            func = interpolate.interp1d(x=[MIN_AMBIGUITY, AVG_AMBIGUITY, MAX_AMBIGUITY], y=[1.0, 1, 0.5],
                                        kind='quadratic')
            return func(x).item()

    def funcinterp_severity(self, x):
        func = interpolate.interp1d(x=[MIN_SEVERITY, AVG_SEVERITY, MAX_SEVERITY], y=[0.5, 1, 2],
                                    kind='quadratic')
        return func(x).item()

    # Compute  deviations for each individual response, from attributes, per question, raw or biased
    #  Use forcedUserId if you w ant to do this only for a certain given user
    #@profile
    def ComputeDeviations_userLevel(self, forcedUserId = None):
        if forcedUserId == None:
            # First, Compute all the attributes used in the interview
            surveyTemplate: Survey = self.surveyTemplate
            self.allAttributesUsedInDataset = []
            allUniqueAttributes = set()
            for themeIndex, themeData in enumerate(surveyTemplate.themes):
                usedAttributes = surveyTemplate.getAttributesUsedInTheme(themeIndex)
                for x in usedAttributes:
                    allUniqueAttributes.add(x)
            self.allAttributesUsedInDataset = list(allUniqueAttributes)

            # Then, count the number of users and responses , total and per team
            self.totalUsersCount = 0
            self.totalUsersCountByTeam = {}
            for teamId, teamResponses in self.allUsersResponses.items():
                numResponsesInTeam = len(teamResponses)
                self.totalUsersCountByTeam[teamId] = numResponsesInTeam
                self.totalUsersCount += numResponsesInTeam



        # Calculate and cache the deviations for each of the user responses
        # Iterate over the user responses in each team
        for teamId, teamResponses in self.allUsersResponses.items():
            for singleUserResponses in teamResponses:
                if forcedUserId is not None and singleUserResponses.id != forcedUserId:
                    continue

                assert singleUserResponses.teamId == teamId

                singleUserResponses.resetCache()

                # Create empty templates for cache inside the single user data
                singleUserResponses.deviations = [None] * len(surveyTemplate.themes)
                # Here we aggregate all responses deviations by certain categories of questions
                singleUserResponses.allResponseDeviationsByCategory = {}
                for catId in self.categoriesList:
                    singleUserResponses.allResponseDeviationsByCategory[catId] = []


                numQuestionsContainingAttribute : Dict[any, int] = {} # THe number of questions that uses a certain attribute id
                numQuestionsContainingAttribute_perCategory : Dict[any, Dict[any,  int]] = {} # Same as above, but first key is the category id

                # Create empty templates for cache for attributes
                for catId in self.categoriesList:
                    singleUserResponses.attributesDeviations_byCategory_raw[catId] = {}
                    singleUserResponses.attributesDeviations_byCategory_biased[catId] = {}
                    numQuestionsContainingAttribute_perCategory[catId] = {}

                    for attrId in self.allAttributesUsedInDataset:
                        singleUserResponses.attributesDeviations_byCategory_raw[catId][attrId] = 0.0
                        singleUserResponses.attributesDeviations_byCategory_biased[catId][attrId] = 0.0
                        singleUserResponses.attributesDeviations_raw[attrId] = 0.0
                        singleUserResponses.attributesDeviations_biased[attrId] = 0.0
                        numQuestionsContainingAttribute[attrId] = 0
                        numQuestionsContainingAttribute_perCategory[catId][attrId] = 0

                # Compute the user mean and std dev from global statistics deviations average to the user deviations on the questions he responded on
                # See the technical document for more details. This helps us removing the biases from users who  are coleric or something and give very deviated  responses each time
                # The output will be in constantBiasForUser variable
                diffDeviationsFromUserToGlobalStats = []
                globalDeviationsByQuestion = self.globalStatistics.avgDeviationByQuestion
                for themeIndex, themeData in enumerate(surveyTemplate.themes):
                    for questionIndex, questionData in enumerate(themeData.questions):
                        questionId = questionData.id
                        userResponseValueOnThisQuestion = singleUserResponses.responses[themeIndex][questionIndex]
                        rawDeviation = userResponseValueOnThisQuestion - questionData.baseline

                        avgRawDeviationInGlobalStat = 0 if questionId not in globalDeviationsByQuestion else globalDeviationsByQuestion[questionId]
                        diffDeviationsFromUserToGlobalStats.append(abs(rawDeviation - avgRawDeviationInGlobalStat))

                diffDeviationsFromUserToGlobalStats = np.array(diffDeviationsFromUserToGlobalStats)
                diffDeviations_mean =  np.mean(diffDeviationsFromUserToGlobalStats)
                diffDeviations_std  =  np.std(diffDeviationsFromUserToGlobalStats)
                constantBiasForUser = diffDeviations_mean / max(1.0, diffDeviations_std)
                #----------


                for themeIndex, themeData in enumerate(surveyTemplate.themes):
                    deviationsForTheme: List[QuestionResponseDeviation] = [None] * len(themeData.questions)
                    clipUsedOnTheme = themeData.clip

                    attributesUsed = clipUsedOnTheme.attributes_nonzeroKeys

                    # Compute and fill statistics for  the individual users for the raw and biased deviation
                    for questionIndex, questionData in enumerate(themeData.questions):
                        questionId = questionData.id
                        userResponseValueOnThisQuestion = singleUserResponses.responses[themeIndex][questionIndex]

                        rawDeviation = userResponseValueOnThisQuestion - questionData.baseline

                        # Bias by ambiguity
                        biasFactorByAmbiguity   = self.funcinterp_ambiguity(questionData.ambiguity)
                        # Bias by severity
                        biasFactorBySeverity    = self.funcinterp_severity(questionData.severity)
                        biasedDeviation = rawDeviation * biasFactorByAmbiguity * biasFactorBySeverity

                        # Adjust the estimated constant bias for the user using the same ratio that drives from raw to biased deviation (because these are factors which also influences this)
                        ratio_rawOverBiasedDeviation = 0.0 if biasedDeviation == 0.0 else (rawDeviation / biasedDeviation)
                        adjusted_constantBiasForUser  = constantBiasForUser if ratio_rawOverBiasedDeviation == 0.0 else constantBiasForUser / ratio_rawOverBiasedDeviation
                        biasedDeviation =  biasedDeviation - (mysign(biasedDeviation) * adjusted_constantBiasForUser)

                        # questionData.category
                        questionDeviations = QuestionResponseDeviation()
                        questionDeviations.rawDeviation = rawDeviation
                        questionDeviations.biasedDeviation = biasedDeviation
                        questionDeviations.questionId = questionId

                        # print("User {0} deviations: raw={1:0.2f} biased={2:0.2f} constantBias={3:0.2f}, adjusted={3:0.2f}".format(singleUserResponses.userId, rawDeviation, biasedDeviation, constantBiasForUser, adjusted_constantBiasForUser))

                        deviationsForTheme[questionIndex] = questionDeviations
                        singleUserResponses.allResponseDeviationsByCategory[questionData.category]\
                            .append(questionDeviations)

                        # For each attribute used in this evaluated question, propagate  the deviations  to attributes
                        for attributeId in attributesUsed:
                            scoreOfAttributeInTheUsedClip = clipUsedOnTheme.attributes.scores[attributeId]
                            isGoodCond = (scoreOfAttributeInTheUsedClip >= 0.0 and scoreOfAttributeInTheUsedClip <= 1.0)
                            assert isGoodCond, "The attributes factors in the dataset should be between 0-1 !"

                            if isGoodCond:
                                numQuestionsContainingAttribute[attributeId] += 1
                                numQuestionsContainingAttribute_perCategory[questionData.category][attributeId] += 1


                                # Multiply the deviation observed on  the question response by how important is the attribute for the clip presented
                                rawDeviation_atrrValue = rawDeviation * scoreOfAttributeInTheUsedClip
                                biasedDeviation_attrValue = biasedDeviation * scoreOfAttributeInTheUsedClip

                                singleUserResponses.attributesDeviations_raw[attributeId] += rawDeviation_atrrValue
                                singleUserResponses.attributesDeviations_biased[attributeId] += biasedDeviation_attrValue
                                singleUserResponses.attributesDeviations_byCategory_raw[questionData.category][attributeId] += rawDeviation
                                singleUserResponses.attributesDeviations_byCategory_biased[questionData.category][attributeId] += biasedDeviation

                                # Same stats as above but per question index
                                if questionId not in singleUserResponses.attributesDeviations_byQuestion_raw:
                                    singleUserResponses.attributesDeviations_byQuestion_raw[questionId] = {}
                                    singleUserResponses.attributesDeviations_byQuestion_biased[questionId] = {}
                                    for newAttrId in self.allAttributesUsedInDataset:
                                        singleUserResponses.attributesDeviations_byQuestion_raw[questionId][newAttrId] = 0.0
                                        singleUserResponses.attributesDeviations_byQuestion_biased[questionId][newAttrId] = 0.0

                                singleUserResponses.attributesDeviations_byQuestion_raw[questionId][attributeId]     += rawDeviation
                                singleUserResponses.attributesDeviations_byQuestion_biased[questionId][attributeId]  += biasedDeviation
                                self.allKnownQuestionIds.add(questionData.id)

                    singleUserResponses.deviations[themeIndex] = deviationsForTheme

                # Normalize the attributes scores by dividing with the number of questions on which appeared.
                # This way they will have all scores between mindeviation - max deviations (0-7 currently)
                # Think that you have 10 questions with the same attribute 'LEADERSHIP'. This will potentially contain a value in the end between 0-70 that doesn't represent too much

                for attributeId in self.allAttributesUsedInDataset:
                    if numQuestionsContainingAttribute[attributeId] > 0:
                        singleUserResponses.attributesDeviations_raw[attributeId] /= numQuestionsContainingAttribute[attributeId]
                        singleUserResponses.attributesDeviations_biased[attributeId] /= numQuestionsContainingAttribute[attributeId]
                    else:
                        assert singleUserResponses.attributesDeviations_raw[attributeId] == 0.0 and singleUserResponses.attributesDeviations_biased[attributeId] == 0.0,\
                                    "Attribute used but not counted ???"

                    # Same as above normalization but this time including the categories....
                    for categoryId in self.categoriesList:
                        if numQuestionsContainingAttribute_perCategory[categoryId][attributeId] > 0:
                            singleUserResponses.attributesDeviations_byCategory_raw[categoryId][attributeId] /= numQuestionsContainingAttribute_perCategory[categoryId][attributeId]
                            singleUserResponses.attributesDeviations_byCategory_biased[categoryId][attributeId] /= numQuestionsContainingAttribute_perCategory[categoryId][attributeId]
                        else:
                            assert singleUserResponses.attributesDeviations_byCategory_raw[categoryId][attributeId] == 0.0 and \
                                   singleUserResponses.attributesDeviations_byCategory_biased[categoryId][attributeId] == 0.0,\
                                        "Attribute used but not counted ???"

                        # Sanity check to see that all values are in range
                        assert -MAX_QUESTION_RESPONSE <= singleUserResponses.attributesDeviations_byCategory_raw[categoryId][attributeId] <= MAX_QUESTION_RESPONSE

                    # Sanity check to see that all values are in range
                    assert -MAX_QUESTION_RESPONSE <= singleUserResponses.attributesDeviations_raw[attributeId] <= MAX_QUESTION_RESPONSE

                # Reduce the response deviations by categories to find statistic deviation for each stat
                for catId in self.categoriesList:
                    singleUserResponses.deviationsStats_byCategory[catId] = {}
                    if len(singleUserResponses.allResponseDeviationsByCategory[catId]) > 0:
                        for op in STATS_KEYS:
                            singleUserResponses.deviationsStats_byCategory[catId][op] = \
                                self.reduceResponsesDeviations(
                                    singleUserResponses.allResponseDeviationsByCategory[catId],
                                    op
                                )
                    else:
                        for op in STATS_KEYS:
                            res = QuestionResponseDeviation()
                            res.rawDeviation = 0
                            res.biasedDeviation = 0
                            singleUserResponses.deviationsStats_byCategory[catId][op] = res

                    # Sanity check that for each category the raw deviations (at least) are in the normal range..
                    for op in STATS_KEYS:
                        assert -MAX_QUESTION_RESPONSE <= singleUserResponses.deviationsStats_byCategory[catId][op].rawDeviation <= MAX_QUESTION_RESPONSE

    @staticmethod
    def reduceResponsesDeviations(
            responsesDeviations: List[QuestionResponseDeviation],
            op: ReduceOpType
    ) -> QuestionResponseDeviation:
        res = QuestionResponseDeviation()
        allRawValues = []  # These are for mode and mean only
        allBiasedValues = []
        for responseDeviation in responsesDeviations:
            if op == ReduceOpType.MIN:
                if res.rawDeviation is None or res.rawDeviation > responseDeviation.rawDeviation:
                    res.rawDeviation = responseDeviation.rawDeviation

                if res.biasedDeviation is None or res.biasedDeviation > responseDeviation.biasedDeviation:
                    res.biasedDeviation = responseDeviation.biasedDeviation

            elif op == ReduceOpType.MAX:
                if res.rawDeviation is None or res.rawDeviation < responseDeviation.rawDeviation:
                    res.rawDeviation = responseDeviation.rawDeviation

                if res.biasedDeviation is None or res.biasedDeviation < responseDeviation.biasedDeviation:
                    res.biasedDeviation = responseDeviation.biasedDeviation

            elif op == ReduceOpType.MEAN or op == ReduceOpType.MEDIAN:
                assert isinstance(responseDeviation.rawDeviation, float)
                allRawValues.append(responseDeviation.rawDeviation)
                allBiasedValues.append(responseDeviation.biasedDeviation)

        isEmpty = len(allRawValues) == 0
        if op == ReduceOpType.MEAN:
            res.rawDeviation = statistics.mean(np.abs(np.array(allRawValues))) if isEmpty is False else 0# It makes no sense to have average of deviations ! What would happen with 2 questions with deviations -3 and 3 ?
            res.biasedDeviation = statistics.mean(np.abs(np.array(allBiasedValues))) if isEmpty is False else 0
        elif op == ReduceOpType.MEDIAN:
            res.rawDeviation = statistics.median(np.abs(np.array(allRawValues))) if isEmpty is False else 0# Same comment as above
            res.biasedDeviation = statistics.median(np.abs(np.array(allBiasedValues))) if isEmpty is False else 0

        return res

    # Same as above but aggregate results by team or global/organization level
    #@profile
    def ComputeDeviations_Aggregated(self):
        allTeamsKeys = self.allUsersResponses.keys()

        # Step 1: Create the empty output caches first
        for teamId in allTeamsKeys:
            self.aggregatedDeviations_PerTeam_raw[teamId] = {}
            self.aggregatedDeviations_PerTeam_biased[teamId] = {}

            for themeIndex, themeData in enumerate(self.surveyTemplate.themes):
                self.aggregatedDeviations_PerTeam_raw[teamId][themeIndex] = {}
                self.aggregatedDeviations_PerTeam_biased[teamId][themeIndex] = {}
                self.aggregatedDeviations_Global_raw[themeIndex] = {}
                self.aggregatedDeviations_Global_biased[themeIndex] = {}

                for questionIndex, questionData in enumerate(themeData.questions):
                    self.aggregatedDeviations_PerTeam_raw[teamId][themeIndex][questionIndex] = []
                    self.aggregatedDeviations_PerTeam_biased[teamId][themeIndex][questionIndex] = []
                    self.aggregatedDeviations_Global_raw[themeIndex][questionIndex] = []
                    self.aggregatedDeviations_Global_biased[themeIndex][questionIndex] = []

        # Overviews of deviations by team and categories
        for teamId in allTeamsKeys:
            self.overviewDeviationsByCategory_PerTeam_raw[teamId] = {}
            self.overviewDeviationsByCategory_PerTeam_biased[teamId] = {}

            for catId in self.categoriesList:
                self.overviewDeviationsByCategory_Global_raw[catId] = []
                self.overviewDeviationsByCategory_Global_biased[catId] = []
                self.overviewDeviationsByCategory_PerTeam_raw[teamId][catId] = []
                self.overviewDeviationsByCategory_PerTeam_biased[teamId][catId] = []

        # Overviews of attributes scores
        for teamId in allTeamsKeys:
            self.overviewAttributes_PerTeam_biased[teamId] = {}
            self.overviewAttributes_PerTeam_raw[teamId] = {}
            for attrId in self.allAttributesUsedInDataset:
                self.overviewAttributes_Global_biased[attrId] = []
                self.overviewAttributes_Global_raw[attrId] = []
                self.overviewAttributes_PerTeam_biased[teamId][attrId] = []
                self.overviewAttributes_PerTeam_raw[teamId][attrId] = []

        # Step 2: Populate the caches created above
        # For each team we are interested in
        for teamId in allTeamsKeys:
            responsesByTeam = self.allUsersResponses[teamId]

            for userResponses in responsesByTeam:
                # For each theme in the survey
                for themeIndex, themeQuestions in enumerate(userResponses.deviations):
                    # For each question in the theme
                    for questionIndex, questionResponse in enumerate(themeQuestions):
                        self.aggregatedDeviations_PerTeam_raw[teamId][themeIndex][questionIndex]\
                            .append(questionResponse.rawDeviation)
                        self.aggregatedDeviations_PerTeam_biased[teamId][themeIndex][questionIndex]\
                            .append(questionResponse.biasedDeviation)

                        self.aggregatedDeviations_Global_raw[themeIndex][questionIndex]\
                            .append(questionResponse.rawDeviation)
                        self.aggregatedDeviations_Global_biased[themeIndex][questionIndex]\
                            .append(questionResponse.biasedDeviation)

                        # Get the category and fill in the overviews too
                        templateQuestionData = \
                            self.surveyTemplate.themes[themeIndex].questions[questionIndex]
                        category = templateQuestionData.category
                        self.overviewDeviationsByCategory_Global_raw[category]\
                            .append(questionResponse.rawDeviation)
                        self.overviewDeviationsByCategory_Global_biased[category]\
                            .append(questionResponse.biasedDeviation)
                        self.overviewDeviationsByCategory_PerTeam_raw[teamId][category]\
                            .append(questionResponse.rawDeviation)
                        self.overviewDeviationsByCategory_PerTeam_biased[teamId][category]\
                            .append(questionResponse.biasedDeviation)

                # Gather ATTRIBUTES from individuals to global and categories attributes values
                for attrId, attrValue in userResponses.attributesDeviations_raw.items():
                    self.overviewAttributes_Global_raw[attrId].append(attrValue)
                    self.overviewAttributes_PerTeam_raw[teamId][attrId].append(attrValue)

                for attrId, attrValue in userResponses.attributesDeviations_biased.items():
                    self.overviewAttributes_Global_biased[attrId].append(attrValue)
                    self.overviewAttributes_PerTeam_biased[teamId][attrId].append(attrValue)

    # Constructs the pandas dataframe
    #@profile
    def ComputeAllStatsTable(self):
        # Populate the columns order: Categories then attributes
        self.columnsInStatsTable = []
        self.columnsInStatsTable_attrOnly = []
        for catId in self.categoriesList:
            for statOp in STATS_KEYS:
                self.columnsInStatsTable.append(f'{catId}_{statOp.name}')

        for attrId in self.allAttributesUsedInDataset:
            self.columnsInStatsTable.append(attrId)
            self.columnsInStatsTable_attrOnly.append(attrId)


        dataframes_indicesAllUsers = index=np.arange(self.totalUsersCount)

        self.statsTable_raw = pd.DataFrame(index=dataframes_indicesAllUsers, columns=self.columnsInStatsTable)
        self.statsTable_biased = pd.DataFrame(index=dataframes_indicesAllUsers, columns=self.columnsInStatsTable)

        allTeamsKeys = self.allUsersResponses.keys()
        for teamId in allTeamsKeys:
            dataframes_indicesAllUsersInThisTeam = index=np.arange(self.totalUsersCountByTeam[teamId])
            self.statsTable_perTeam_raw[teamId] = pd.DataFrame(index=dataframes_indicesAllUsersInThisTeam, columns=self.columnsInStatsTable)
            self.statsTable_perTeam_biased[teamId] = pd.DataFrame(index=dataframes_indicesAllUsersInThisTeam, columns=self.columnsInStatsTable)

        userIndex = -1
        for teamId, responsesByTeam in self.allUsersResponses.items():
            userPerTeamIndex = -1
            for userResponse in responsesByTeam:
                userIndex += 1
                userPerTeamIndex += 1

                # Populate values for row userIndex
                values_raw = {}
                values_biased = {}

                # Categories stats first
                for catId in self.categoriesList:
                    for statOp in STATS_KEYS:
                        values_raw[f'{catId}_{statOp.name}'] = userResponse.deviationsStats_byCategory[catId][statOp].rawDeviation
                        values_biased[f'{catId}_{statOp.name}'] = userResponse.deviationsStats_byCategory[catId][statOp].biasedDeviation

                # Then attributes
                for attrId in self.allAttributesUsedInDataset:
                    values_raw[attrId] = userResponse.attributesDeviations_raw[attrId]
                    values_biased[attrId] = userResponse.attributesDeviations_biased[attrId]

                self.statsTable_raw.loc[userIndex] = values_raw
                self.statsTable_perTeam_raw[teamId].loc[userPerTeamIndex] = values_raw
                self.statsTable_biased.loc[userIndex] = values_biased
                self.statsTable_perTeam_biased[teamId].loc[userPerTeamIndex] = values_biased

                if userResponse.userId not in self.statsTable_perUser_raw:
                    self.statsTable_perUser_raw[userResponse.userId] = pd.DataFrame(columns=self.columnsInStatsTable)
                    self.statsTable_perUser_biased[userResponse.userId] = pd.DataFrame(columns=self.columnsInStatsTable)

                self.statsTable_perUser_raw[userResponse.userId].loc[0] = values_raw
                self.statsTable_perUser_biased[userResponse.userId].loc[0] = values_biased

        # Stat table for attributes filtered by question instead
        for questionId in self.allKnownQuestionIds:
            self.statsTable_perQuestionAndTeam_biased[questionId] = {}
            self.statsTable_perQuestionAndTeam_raw[questionId] = {}
            self.statsTable_perQuestion_biased[questionId] = pd.DataFrame(index=dataframes_indicesAllUsers, columns=self.columnsInStatsTable_attrOnly)
            self.statsTable_perQuestion_raw[questionId] = pd.DataFrame(index=dataframes_indicesAllUsers, columns=self.columnsInStatsTable_attrOnly)

            for teamId in allTeamsKeys:
                dataframes_indicesAllUsersInThisTeam = index=np.arange(self.totalUsersCountByTeam[teamId])
                self.statsTable_perQuestionAndTeam_biased[questionId][teamId] = pd.DataFrame(index=dataframes_indicesAllUsersInThisTeam, columns=self.columnsInStatsTable_attrOnly)
                self.statsTable_perQuestionAndTeam_raw[questionId][teamId] = pd.DataFrame(index=dataframes_indicesAllUsersInThisTeam, columns=self.columnsInStatsTable_attrOnly)

            userIndex = -1
            for teamId, responsesByTeam in self.allUsersResponses.items():
                userPerTeamIndex = -1
                for userResponse in responsesByTeam:
                    userIndex += 1
                    userPerTeamIndex += 1
                    values_raw = {}
                    values_biased = {}

                    for attrId in self.allAttributesUsedInDataset:
                        values_raw[attrId] = userResponse.attributesDeviations_byQuestion_raw[questionId][attrId]
                        values_biased[attrId] = userResponse.attributesDeviations_byQuestion_biased[questionId][attrId]

                    self.statsTable_perQuestion_raw[questionId].loc[userIndex] = values_raw
                    self.statsTable_perQuestionAndTeam_raw[questionId][teamId].loc[userPerTeamIndex] = values_raw
                    self.statsTable_perQuestion_biased[questionId].loc[userIndex] = values_biased
                    self.statsTable_perQuestionAndTeam_biased[questionId][teamId].loc[userPerTeamIndex] = values_biased


    # Given options, return the correlation matrix processed
    def getCorrelationsBetweenStats(self, options : CorrelationOptions):


        # Select the data to correlate depending on the requested options
        dataSelected = None

        # Is the data filtered by a certain question id ?
        # Then select as data the ATTRIBUTES only for that particular question. Categories are not shown since each question is already part of a single category
        # This is at global level, probably doesn't make sense to put it at team level too ?
        if options.questionFilteredId is not None:
            if options.teamId is None:
                dataSelected = self.statsTable_perQuestion_raw[options.questionFilteredId] if options.useBiasedValues == False \
                    else self.statsTable_perQuestion_biased[options.questionFilteredId]
            else:  # Select at team level no filtering...
                dataSelected = self.statsTable_perQuestionAndTeam_raw[options.questionFilteredId][options.teamId] if options.useBiasedValues == False \
                    else self.statsTable_perQuestionAndTeam_biased[options.questionFilteredId][options.teamId]

            if options.itemsToShow != None:
                dataSelected = dataSelected[options.itemsToShow]
            else:
                options.itemsToShow = dataSelected.columns
        else:
            if options.itemsToShow == None:
                options.itemsToShow = list(self.statsTable_raw.columns)

            # Select at global level everything, no other filtering...
            if options.teamId is None:
                dataSelected = self.statsTable_raw[options.itemsToShow] if options.useBiasedValues == False \
                    else self.statsTable_biased[options.itemsToShow]
            else: # Select at team level no other filtering...
                dataSelected = self.statsTable_perTeam_raw[options.teamId][options.itemsToShow] if options.useBiasedValues == False \
                    else self.statsTable_perTeam_biased[options.teamId][options.itemsToShow]

        """
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(dataSelected['Sexual Harassment'])
            print(dataSelected['Personal Boundaries'])
        """

        #print(dataSelected.dtypes)
        dataSelected = dataSelected.astype('float32')
        #print(dataSelected.dtypes)

        corr_all = dataSelected.corr()
        selected_pairs : List[Tuple[str, str, float]] = [] # Between A,B we have correlation value X
        for statIndex_1 in range(len(options.itemsToShow)):
            for statIndex_2 in range(statIndex_1 + 1, len(options.itemsToShow)):
                strStat_1 = options.itemsToShow[statIndex_1]
                strStat_2 = options.itemsToShow[statIndex_2]

                # are the stat coming from the same stat ?
                if strStat_1.split("_")[0] == strStat_2.split("_")[0]:
                    continue

                corrValue = corr_all[strStat_1][strStat_2]
                if abs(corrValue) >= options.threshold:
                    selected_pairs.append((strStat_1, strStat_2, corrValue))

        # Sort the list now by highest corrvalue
        selected_pairs = sorted(selected_pairs, key=lambda tup : tup[2], reverse=True)
        return selected_pairs, corr_all



    # This handles the preprocessing of data gathered in a way that we can process efficiently for visualization and AI
    # TODO: put all data in pandas dataframe instead of custom dictionaries...An example is the self.statsTable which
    #  is just the beginning
    def preprocessData(self):
        self.ComputeDeviations_userLevel()
        self.ComputeDeviations_Aggregated()
        self.ComputeAllStatsTable()


    # Gets a panda table filled with either global or team level data and filtered by categories and attributes deviations
    # Columns: [userId, Category 1, Category 2, ..... , Atrr 1, Attr 2]
    # statsType is the one used to get the results for individuals on categories, like mean, median, min/max etc.. BY default we put mean
    #   listOfCategoryIds, listOfAttributesIds = the list of Ids to consider for clustering
    #   teamId = None if you want global level, or per team else
    #   statType = the reduce operation to aggregating deviations of responses
    #   useRawData = True if you want raw, false For Biased
    #   saveFeaturesAsCsv = name of the file where to write out the features extracted according to your input
    def getResponsesFeaturesData(self, listOfCategoryIds, listOfAttributesIds, teamId = None,
                       statType : ReduceOpType = ReduceOpType.MEAN,
                       useRawData : bool = True) -> pd.DataFrame: # leave none for teamId to get global level operation
        # Step 1: Preallocate the panda dataframe
        # 1.1 but first need to know how many rows there are
        teamIdsToExtractData = self.allUsersResponses.keys() if teamId is None else [teamId]

        numTotalUsersRequested = 0
        for teamId in teamIdsToExtractData:
            responsesByTeam = self.allUsersResponses[teamId]
            numTotalUsersRequested += self.totalUsersCountByTeam[teamId]

        columns = ['UserId']
        for requestedCatId in listOfCategoryIds:
            assert requestedCatId in self.categoriesList, f"The category {requestedCatId} requested is not recorded in the dataset !"
            columns.append(requestedCatId)
        for requestedAttrId in listOfAttributesIds:
            assert requestedAttrId in self.allAttributesUsedInDataset, f"The attribute {requestedAttrId} requested is not recorded in the dataset !"
            columns.append(requestedAttrId)

        resDataFrame = pd.DataFrame(index=np.arange(numTotalUsersRequested), columns=columns)

        # Step 2: fill the table
        userIndex = -1
        for teamId in teamIdsToExtractData:
            responsesByTeam = self.allUsersResponses[teamId]
            for userResponse in responsesByTeam:
                userIndex += 1
                resDataFrame.iloc[userIndex]['UserId'] = userResponse.userId

                # Put all categories stats in the output dataframe
                for requestedCatId in listOfCategoryIds:
                    statsDeviations : QuestionResponseDeviation  = userResponse.deviationsStats_byCategory[requestedCatId][statType]
                    devVal = statsDeviations.rawDeviation if useRawData else statsDeviations.biasedDeviation
                    resDataFrame.iloc[userIndex][requestedCatId] = devVal

                # Now put all the attributes
                deviationsForAttributes = userResponse.attributesDeviations_raw if useRawData else userResponse.attributesDeviations_biased
                for requestedAttrId in listOfAttributesIds:
                    attrDevVal = deviationsForAttributes[requestedAttrId]
                    resDataFrame.iloc[userIndex][requestedAttrId] = attrDevVal


        # Step 3: sanity check to see if table has no NAs then return it
        assert resDataFrame.isnull().any().any() == False, "Your table contains NAs !! Please check this again and see what was not completed !!"
        return resDataFrame

    # Scores each individual in the organization part requested for each cluster type
    #   teamId = None if you want global level, or per team else
    #   statType = the reduce operation to aggregating deviations of responses
    #   useRawData = True if you want raw, false For Biased
    # Returns a tuple of (feature and score) for each individual user in the team requested for evaluation in the parameters list
    def getManualClusteringScoresDistribution(self, clustersSpec : ManualClustersSpec,
                                        teamId, statType : ReduceOpType = ReduceOpType.MEAN,
                                        useRawData : bool = True) -> Tuple[np.array, ManualClusterScores]:
        numClusters = len(clustersSpec.clusters)

        # TODO Ciprian: REFACTORING - feature extraction from the code below should be done with getResponsesFeaturesData instead to have it unified with the other method !

        # Step 1: Preallocate output data structure
        teamIdsToExtractData = self.allUsersResponses.keys() if teamId is None else [teamId]

        numTotalUsersRequested = 0
        for teamId in teamIdsToExtractData:
            responsesByTeam = self.allUsersResponses[teamId]
            numTotalUsersRequested += self.totalUsersCountByTeam[teamId]

        scoresRes = ManualClusterScores(numTotalUsersRequested, numClusters)
        usersFeatures = np.zeros(shape=(numTotalUsersRequested, clustersSpec.clusters[0].numFeatures + 1)) # First feature is the user id

        userIndex = -1
        for teamId in teamIdsToExtractData:
            responsesByTeam = self.allUsersResponses[teamId]
            for userResponse in responsesByTeam:
                userIndex += 1

                # Put the user id
                scoresRes.result[userIndex][0] = userResponse.userId
                usersFeatures[userIndex][0] = userResponse.userId

                # For each cluster gather some data needed like the devations and feature names
                userDeviationsByCategory : Dict[any, float] = {}
                userDeviationsForAttributes : Dict[any, float] = {}
                for clusterId, clusterSpec in enumerate(clustersSpec.clusters):
                    category_features : List[ManualClusterFeature] = clusterSpec.category_features
                    attributes_features : List[ManualClusterFeature] = clusterSpec.attributes_features

                    # Iterate over the cluster features, get the individual response to them and sum up
                    feature_index = 0 # First feature was the user id
                    for cat_feature in category_features:
                        feature_index += 1
                        statsDeviations: QuestionResponseDeviation = userResponse.deviationsStats_byCategory[cat_feature.name][statType]
                        devVal = statsDeviations.rawDeviation if useRawData else statsDeviations.biasedDeviation
                        userDeviationsByCategory[cat_feature.name] = devVal
                        usersFeatures[userIndex][feature_index] = devVal

                    deviationsForAttributes = userResponse.attributesDeviations_raw if useRawData else userResponse.attributesDeviations_biased
                    for attr_feature in attributes_features:
                        feature_index += 1
                        attrDevVal = deviationsForAttributes[attr_feature.name]
                        userDeviationsForAttributes[attr_feature.name] = attrDevVal
                        usersFeatures[userIndex][feature_index] = attrDevVal

                #  Compute the score for each cluster
                userResponse.statType = statType
                userResponse.useRawData = useRawData
                res:UserScoresSurveyResults = SurveyResponses.scoreUserSurveyToClustersSpec(
                    userId = userResponse.userId,
                    userDeviationsByCategory=None, #userDeviationsByCategory,
                    userAttributesDeviations=None, #userDeviationsForAttributes,
                    userAttributesPerCategoryDeviations= None, # attributesDeviations_byCategory,
                    userResponse=userResponse,
                    clustersSpec=clustersSpec)

                # Some sanity checks...
                assert len(res.outProbabilityPerCluster_normalized) == len(clustersSpec)
                sumProbs = np.sum(res.outProbabilityPerCluster_normalized)
                assert (sumProbs - 1.0) < 0.0001, "The normalized final probability is not correct"

                # Fill in the normalized scores
                for clusterIndex in range(numClusters):
                    scoresRes.result[userIndex][1 + clusterIndex] = res.outProbabilityPerCluster_normalized[clusterIndex]

        scoresRes.finalize()
        return usersFeatures, scoresRes

# TODO: finish this function
# apply it correctly to the other sides without doing PCA each time
# check the bug with centers on 2D and 3D case when PCA is used
# -1 for numclusters

    # This functions prepare data for clusterization if more dimensions than needed are in the features dataset
    def BehaviorsAutoClusterization_prepareData(self, featuresData : pd.DataFrame, numMaxComponentsDesired : int):
        featuresData = np.asarray(featuresData.values[:, 1:], dtype=np.float)  # First column is user id, note interesting for us
        dimensionalityWasReduced = False

        # If the number of columns in the features Data is > 3, then we apply Pricipal Component Analysis (PCA) we reduce the dimensionality:
        if featuresData.shape[1] > numMaxComponentsDesired:
            NUM_PRINCIPAL_AXES = numMaxComponentsDesired  # target
            initialFeaturesShape = copy.copy(featuresData.shape)

            # First, scale the data to substracting the mean and reducing by the standard deviation
            scaler = StandardScaler()
            featuresData = scaler.fit_transform(featuresData)

            # Normalize..
            featuresData = normalize(featuresData)

            # Apply PCA
            pca = PCA(n_components=NUM_PRINCIPAL_AXES)
            # TODO: log about PCA here
            #metadata.PCATransformUsed = pca

            featuresData = pca.fit_transform(featuresData)

            # Check the correctness of the transform
            assert featuresData.shape[0] == initialFeaturesShape[0], featuresData.shape[1] == NUM_PRINCIPAL_AXES

        return featuresData, dimensionalityWasReduced

    # This function gets you the automated clustering method
    # Inputs :
    #   numClusters = in how many cluster would you like to split the features ?
    #           If -1, then it will do auto selection. Warning: this is costly..
    #           minClusters and maxCluster are the range of clusters to search from
    #   featuresDataFrame = the features extracted (see above function)
    # Outputs:
    # A. the Gaussian multi mixture (GMM) distribution trained with Expectation maximization (EM) algorithm that describes the data
    # as best as possible within the number of clusters requested as input, and the features requested in the input.
    # B. the metadata for the gaussian fit containig the means and stanfard deviations of the fitted guassian , their importance, etc.
    def BehaviorsAutoClusterization_fit(self, numClusters, featuresData : np.array, minClusters : int = None, maxClusters : int = None, plot=True):
        np.random.seed(0) # To have deterministic results

        metadata = CustomGaussianMixtureMeta()
        selected_gmmDist =None

        # If client wants to find the number of optimal clusters automatically, find it...but it's costly :)
        if numClusters < 0:
            assert (minClusters != None and maxClusters != None), "Cluster ranges not specified ! When searching automatiucally the optimal number, please provide these"
            cluster_range = np.arange(minClusters, maxClusters)

            # Since the EM algorithm is not deterministic, we run 8 iterations for proposal and we select the 3 best of these when we do the final comparison
            num_local_iterations = 8

            silhouette_means = []
            silhouette_err = []
            for cluster_proposal in cluster_range:
                local_results = np.zeros(shape=(num_local_iterations))

                for iter in range(num_local_iterations):
                    gmmLocal = mixture.GaussianMixture(n_components=cluster_proposal, n_init=2, covariance_type='full').fit(featuresData)
                    labels = gmmLocal.predict(featuresData)
                    sil_score = metrics.silhouette_score(featuresData, labels, metric='euclidean')
                    local_results[iter] = sil_score

                # Select the 3 best runs
                best_runs_indices = np.argsort(local_results)[:3]
                best_runs_values = local_results[best_runs_indices]
                local_mean = np.mean(best_runs_values)
                local_err = np.std(local_results)

                # Add aggregate to the statistics
                silhouette_means.append(local_mean)
                silhouette_err.append(local_err)

            if plot:
                plt.errorbar(cluster_range, silhouette_means, yerr=silhouette_err)
                plt.title("Silhouette Scores", fontsize=20)
                plt.xticks(cluster_range)
                plt.xlabel("N. of clusters")
                plt.ylabel("Score")

            # Finally, put the number of clusters on the best run and run the gaussian fit using this number
            numClusters =  cluster_range[np.argmax(silhouette_means)]


        # Create a Gaussian mixture distribution
        gmmDist = mixture.GaussianMixture(n_components=numClusters, covariance_type='full')

        # Fit features and get medians, stds and correlations using Expectation Maximization algorithm
        gmmDist.fit(featuresData)

        metadata.numClasses = numClusters
        metadata.centroids = gmmDist.means_
        assert len(metadata.centroids) == numClusters == metadata.numClasses
        metadata.weights = gmmDist.weights_
        metadata.covariances = gmmDist.covariances_

        return gmmDist, metadata


    # Given a trained Gaussian Mixture model, our custom metadata  about it and the input data (featuresData),
    # We return a prediction data structure  that holds labels and trust probabilities
    def BehaviorsAutoClusterization_predict(self, trainedGMM : mixture.GaussianMixture,
                                            trainedGMM_meta : CustomGaussianMixtureMeta,
                                            featuresData : np.array) -> PredictionGaussianMixtureMeta:

        # Create a Gaussian mixture distribution
        # Fit features and get medians, stds and correlations using Expectation Maximization algorithm
        predictionRes = PredictionGaussianMixtureMeta()
        gmmLabels = trainedGMM.predict(featuresData)
        gmmProbs = trainedGMM.predict_proba(featuresData)
        predictionRes.gmmProbs = gmmProbs

        #uniqueClasses, countsPerUniqueClass = np.unique(gmmLabels, return_counts=True)

        predictionRes.numUsers = featuresData.shape[0]
        predictionRes.numClasses = trainedGMM_meta.numClasses
        predictionRes.predictions = gmmLabels
        predictionRes.centroids = trainedGMM_meta.centroids

        predictionRes.confidencePerUser = gmmProbs[np.arange(len(gmmProbs)), gmmLabels]
        predictionRes.meanConfidence = np.mean(predictionRes.confidencePerUser)

        numUsers = featuresData.shape[0]
        predictionRes.numbersPerClass = np.zeros(predictionRes.numClasses, dtype=np.int32)
        predictionRes.percentPerClass = np.zeros(predictionRes.numClasses)

        assert numUsers > 0
        for classIndex in range(predictionRes.numClasses):
            predictionRes.numbersPerClass[classIndex] = np.sum(gmmLabels == classIndex)
            predictionRes.percentPerClass[classIndex] = (predictionRes.numbersPerClass[classIndex] + 0.0) / numUsers

        return predictionRes

    def outputSurveyResponses(results : UserScoresSurveyResults, outPrefixPath : str):
        if not os.path.exists(outPrefixPath):
            os.makedirs(outPrefixPath, exist_ok=True)

        numClusters = len(results.clustersNames)
        for indexCluster in range(numClusters):
            cluster_name        = results.clustersNames[indexCluster]
            cluster_rawProb     = results.outProbabilityPerCluster[indexCluster]
            cluster_normProb    = results.outProbabilityPerCluster_normalized[indexCluster]
            cluster_detailedCat = results.detailedScoresPerCluster_categories[indexCluster]
            cluster_detailedAtt = results.detailedScoresPerCluster_attributes[indexCluster]

            # category labels
            plt.xticks(plt.xticks()[0], sorted_keys)

            if options.saveFigurePath is not None:
                plt.savefig(f"{options.saveFigurePath}_{themeIndex}.png")

            plt.show()

        plt.clf()



    # Given an userID (fictive or not, doesn't make any connection to database),
    # the users deviations by category, his attributes deviations and a set of clusters specification,
    # Returns a UserScoresSurveyResults (see its comments)
    # NOTE1: deviations can be both raw or biased as you wish.
    # NOTE2: This can be used for any static user answer or agent based. You have the needed data for deviations
    # inside, check UserResponses class, or AgentBase !
    @staticmethod
    def scoreUserSurveyToClustersSpec(userId: any,
                                      userDeviationsByCategory: Dict[any, float],
                                      userAttributesDeviations: Dict[any, float],
                                      userAttributesPerCategoryDeviations: Dict[any, Dict[str, float]],
                                      userResponse : UserResponses,
                                      clustersSpec: ManualClustersSpec,
                                      finalizeInEnd=True) -> UserScoresSurveyResults:

        # Extract data from user responses if not provided already
        # For each cluster gather some data needed like the devations and feature names
        assert ((userDeviationsByCategory != None and userAttributesDeviations != None and userAttributesPerCategoryDeviations != None and userResponse == None) or
                (userResponse != None and userDeviationsByCategory == None and userAttributesDeviations == None)),  \
                    "Incorrect input specification. One or another variant must be used !"

        if userResponse != None:
            if userAttributesPerCategoryDeviations is None:
                userAttributesPerCategoryDeviations = {}

            userDeviationsByCategory: Dict[any, float] = {}
            userAttributesDeviations: Dict[any, float] = {}
            userAttributesDeviations_byCategory : Dict[any, Dict[any, float]] = {} # first key index is category, then attribute

            for clusterId, clusterSpec in enumerate(clustersSpec.clusters):
                features: List[ManualClusterFeature] = clusterSpec.features

                deviationsForAttributes = userResponse.attributesDeviations_raw if userResponse.useRawData else userResponse.attributesDeviations_biased
                deviationsForAttributes_byCategory = {}

                # Iterate over the cluster features, get the individual response to them and sum up
                feature_index = 0  # First feature was the user id
                for feature in features:
                    statsDeviations: QuestionResponseDeviation = userResponse.deviationsStats_byCategory[feature.name][userResponse.statType]
                    devVal = statsDeviations.rawDeviation if userResponse.useRawData else statsDeviations.biasedDeviation
                    userDeviationsByCategory[feature.name] = devVal

                    if feature.name not in userAttributesPerCategoryDeviations:
                        userAttributesPerCategoryDeviations[feature.name] = {}

                    allUserDeviationsForAttributes_byCategory = userResponse.attributesDeviations_byCategory_raw[feature.name] if userResponse.useRawData \
                                                                else userResponse.attributesDeviations_byQuestion_biased[feature.name]


                    for attr_feature in feature.listOfAttributes:
                        userAttributesPerCategoryDeviations[feature.name][attr_feature] = allUserDeviationsForAttributes_byCategory[attr_feature]

        clustersNames = clustersSpec.getClustersNames()
        res = UserScoresSurveyResults(userId, clustersNames=clustersNames)
        res.outProbabilityPerCluster = np.zeros(shape=(len(clustersSpec, )), dtype=np.float32)

        # Invididual scores for this cluster's features
        for clusterId, clusterSpec in enumerate(clustersSpec.clusters):
            features: List[ManualClusterFeature] = clusterSpec.features

            outScorePerCategory: Dict[any, float] = {}
            outScorePerAttribute: Dict[any, float] = {}

            # Iterate over the cluster features, get the individual response to them and sum up
            feature_index = 0  # First feature was the user id
            totalProbScore = 0.0
            for feature in features:
                feature_index += 1
                # TODO Ciprian: Param for biased

                weight_for_category_dev = 0.5
                weight_for_attributesMean_Dev = 0.5
                assert (weight_for_category_dev + weight_for_attributesMean_Dev == 1.0), "Equal to 1 sum should be given"

                # Add the category deviation score as TERM1
                #-----
                categoryAnswerDeviation = userDeviationsByCategory[feature.name]
                featureUserScore = feature.scoreDeviation(categoryAnswerDeviation)
                totalProbScore += featureUserScore * weight_for_category_dev
                #----

                # Then add the average of the attributes scores as TERM2
                #----
                attributesScoreForThisFeature = 0
                for attr_feature in feature.listOfAttributes:
                    attrAnswerDeviation = userAttributesPerCategoryDeviations[feature.name][attr_feature]
                    attrUserScore = feature.scoreDeviation(attrAnswerDeviation) # Note: Same deviations as for the category as a whole !
                    attributesScoreForThisFeature += attrUserScore

                    # Add the statistics, i.e. how this attribute influences the scoring behavior at all.
                    if attr_feature not in outScorePerAttribute:
                        outScorePerAttribute[attr_feature] = 0.0
                    outScorePerAttribute[attr_feature] += attrUserScore

                # Compute the mean of the attributes used inside this category feature
                if len(feature.listOfAttributes) > 0:
                    attributesScoreForThisFeature /= len(feature.listOfAttributes)

                totalProbScore += attributesScoreForThisFeature * weight_for_attributesMean_Dev
                #----

                assert feature.name not in outScorePerCategory, "Duplicated feature scored !"
                outScorePerCategory[feature.name] = featureUserScore

            # Then finally normalize the summed result to get a proper probability value for that user
            totalProbScore /= clusterSpec.numFeatures
            assert totalProbScore >= 0.0 and totalProbScore <= 1.0, "The probability value is not normalized !!!"
            assert (clusterSpec.numFeatures == len(features)) ,"The number of features is incorrect" # \ # + len(attributes_features)),

            res.outProbabilityPerCluster[clusterId] = totalProbScore

            # Some sanity checks...
            # Check if the list of categories and attributes are filled correctly in the output
            assert sorted([catFeature.name for catFeature in features]) == sorted(outScorePerCategory.keys())
            #assert sorted([attrFeature.name for attrFeature in attributes_features]) == sorted(outScorePerAttribute.keys())

            res.detailedScoresPerCluster_categories.append(outScorePerCategory)
            res.detailedScoresPerCluster_attributes.append(outScorePerAttribute)

        if finalizeInEnd == True:
            res.finalize()
        return res




# These  two  classes are used to handle the input of  the organization for setting the list of attributes
# and categories that  they are interested in, and how much on each (see technical design doc)
# For a list of attributes given by key, how interested are in  that one between 0-1
class OrgAttributesSet:
    def __init__(self, attributesInterestedIdAndScore : Dict[str, float], attributesFlattened):
        self.attributesInterested = AttributesArray(attributesFlattened=attributesFlattened)

        for item in attributesInterestedIdAndScore.items():
            attrName = item[0]
            attrScore = item[1]

            # Check
            assert (attrName in attributesFlattened), (f"You given an attribute key {attrName} that doesn't exist in the database")
            assert 0.0 <= attrScore <= 1.0, (f"You given an attribute score of {attrScore} but it has to be normalized between [0,1]")
            self.attributesInterested.scores[attrName] = attrScore


# For a list of categories given by key, how interested are in  that one between 0-1
class OrgCategoriesSet:
    def __init__(self, categoriesInterestedIdAndScore : Dict[str, float], categoriesList):
        self.categoriesInterested = categoriesInterestedIdAndScore

        for item in categoriesInterestedIdAndScore.items():
            catName = item[0]
            catScore = float(item[1])

            # Check
            assert (catName in categoriesList), (f"You given a category key {catName} that doesn't exist in the database")
            assert 0.0 <= catScore <= 1.0, (f"You given an attribute score of {catScore} but it has to be normalized between [0,1]")
            self.categoriesInterested[catName] = catScore

# Parameters given by the organization that says what they are interested on
class OrganizationInterestSettings:
    def __init__(self):
        self.attributesInterestedIn : OrgAttributesSet = None
        self.categoriesInterestedIn : OrgCategoriesSet = None

    def isEmpty(self) -> bool:
        return (self.attributesInterestedIn is None or len(self.attributesInterestedIn) == 0) and \
               (self.categoriesInterestedIn is None  or len(self.categoriesInterestedIn) == 0)

class DataStore:
    def __init__(self):
        # Simulating the vortexplore datastore - These are all objects defined and stored in the vortexplore library
        # that clients can chose and create templates.
        self.attributesFlattened = []
        self.categoriesList = []
        self.clips: Dict[any, Clip] = {}  # indexed by id
        self.questions_byId: Dict[int, QuestionForClip] = {}  # indexed by id
        self.questions_byClip: Dict[int, List[QuestionForClip]] = {}  # Same as above, same instances,
        self.acceptableQuestionsIdAfter: Dict[int, List[
            int]]  # If i put a question X, which are the following list of question ids that I can put in continuation ?
        # but indexed by parent clip (-1 key means it is generic)

        self.colorByFeatureName: Dict[str, Tuple[int, int, int]] = {}

    def LoadAttributesAndCategories(self, attrDataframe, catDataframe):
        self.attributesFlattened = attrDataframe['Name'].tolist()
        self.categoriesList = catDataframe['Name'].tolist()

        def addColorForFeature(featureName):
            rgbColor = np.array(ColorHash(featureName).rgb, dtype=np.float32)
            rgbColor *= (1.0/255)
            self.colorByFeatureName[featureName] = rgbColor

        for name in self.attributesFlattened:
            addColorForFeature(name)

        for name in self.categoriesList:
            addColorForFeature(name)

    def LoadClips(self, clipsAttributesDataframe, clipsMetaDataframe):
        allClipsUniqueIds = list(np.unique(clipsAttributesDataframe['ClipId'].tolist()))
        self.allClipsUniqueIds = allClipsUniqueIds

        for clipId in allClipsUniqueIds:
            clipInstance = Clip(self.attributesFlattened)
            clipInstance.id = clipId

            thisClipMetaDataframe = clipsMetaDataframe[clipsMetaDataframe['ClipId'] == clipId]

            clipInstance.severity = (thisClipMetaDataframe['Severity']).tolist()[0]

            # Load clip attributes
            attrDataForClip = clipsAttributesDataframe[clipsAttributesDataframe['ClipId'] == clipId]
            attributesNonZero = attrDataForClip['AttrKey']
            attributesValues = attrDataForClip['AttrValue']

            # Load the dependencies
            clipDataDependencyType = thisClipMetaDataframe['DependencyType'].to_list()
            assert len(clipDataDependencyType) == 1, f"You have to specify a dependency TYPE for each clip, for example {clip.id} doesn't have one or has too many "
            clipDataDependencyType = clipDataDependencyType[0]
            if clipDataDependencyType == ClipDependencyType.DEP_PREVIOUS.value:
                clipDataDependencyType = ClipDependencyType.DEP_PREVIOUS
            elif clipDataDependencyType == ClipDependencyType.DEP_EXCLUDING_PREVIOUS.value:
                clipDataDependencyType = ClipDependencyType.DEP_EXCLUDING_PREVIOUS
            else:
                assert False, f"Unknown dependency type for clip {clipInstance.id} and dependency in data {clipDataDependencyType}"

            clipInstance.dependencyType = clipDataDependencyType
            clipInstance.setOfClipIdsDependencies = set()
            clipDataDependencies_str = str(thisClipMetaDataframe["Dependencies"].tolist()[0]).split(",")
            clipDataDependencies = [int(float(p)) for p in clipDataDependencies_str if p != "nan"]

            # Special case for -1 (no dependency)
            if len(clipDataDependencies) == 1 and clipDataDependencies[0] == NO_DEPENDENCY_CONST:
                clipInstance.clipDataDependencies = set()
            else:
                for clipIdDependency in clipDataDependencies:
                    if clipIdDependency != "nan":
                        clipInstance.setOfClipIdsDependencies.add(clipIdDependency)

            # Generates some attributes for this clip
            clipInstance.attributes = AttributesArray(self.attributesFlattened)

            clipInstance.attributes.scores = {}
            for index, key in enumerate(self.attributesFlattened):
                clipInstance.attributes.scores[key] = 0

            for attrKey, attrValue in zip(attributesNonZero, attributesValues):
                clipInstance.attributes.scores[attrKey] = attrValue
                if attrValue > 0:
                    clipInstance.attributes_nonzeroKeys.append(attrKey)

            self.clips[clipId] = clipInstance

    def LoadQuestions(self, questionsDataframe=None):
        allQuestionUniqueIds = list(np.unique(questionsDataframe['questionId'].tolist()))
        self.allQuestionUniqueIds = allQuestionUniqueIds

        self.questions_byId = {}
        self.questions_byClip = {}

        # The acceptable questions id inside a clip following one one previous question
        # acceptableQuestionsIdAfter[clipId][prev questionid ] = List of question ids available to put after prev question id
        self.acceptableQuestionsIdAfter: Dict[int, Dict[int, List[int]]] = {}

        for questionId in allQuestionUniqueIds:
            dataForQuestion = questionsDataframe[questionsDataframe['questionId'] == questionId]

            for name in ['ambiguity', 'severity', 'baseline']:
                dataForQuestion.loc[:, name] = dataForQuestion[name].astype(float)

            parentClipId = dataForQuestion['parentClipId'].tolist()[0]
            ambiguity = dataForQuestion['ambiguity'].tolist()[0]
            severity = dataForQuestion['severity'].tolist()[0]
            baseline = dataForQuestion['baseline'].tolist()[0]
            category = dataForQuestion['category'].tolist()[0]

            # update the acceptable followup questions
            if parentClipId not in self.acceptableQuestionsIdAfter:
                self.acceptableQuestionsIdAfter[parentClipId] = {}

            parents = str(dataForQuestion['parents'].tolist()[0]).split(",")
            parents = [int(p) for p in parents]
            for parentQuestionId in parents:
                if parentQuestionId not in self.acceptableQuestionsIdAfter[parentClipId]:
                    self.acceptableQuestionsIdAfter[parentClipId][parentQuestionId] = []
                self.acceptableQuestionsIdAfter[parentClipId][parentQuestionId].append(questionId)
            # --

            clipQuestion = QuestionForClip(self.categoriesList)
            clipQuestion.id = questionId
            clipQuestion.parentClipId = parentClipId

            parentClipData : Clip = self.clips[parentClipId]
            clipQuestion.parentClipAtttributes_Array        = parentClipData.attributes
            clipQuestion.parentClipAttributes_nonZeroKeys   = parentClipData.attributes_nonzeroKeys

            clipQuestion.ambiguity = ambiguity
            clipQuestion.severity = severity
            clipQuestion.baseline = baseline
            clipQuestion.setCategories([category])

            if self.questions_byClip.get(parentClipId) is None:
                self.questions_byClip[parentClipId] = []

            self.questions_byId[questionId] = clipQuestion
            self.questions_byClip[parentClipId].append(clipQuestion)

        return list(self.questions_byId.values())

    def LoadSurvey(self, surveyTemplateDataframe):
        def stringArrayToFloatsInPandaRow(rowData):
            numericValues = ast.literal_eval(rowData)
            assert isinstance(numericValues, List)
            numericValues = np.array(numericValues, dtype=float)
            return numericValues

        allThemes = list(np.unique(surveyTemplateDataframe['themeId'].tolist()))

        quest = Survey()
        quest.themes = [None] * len(allThemes)

        for themeId in allThemes:
            theme = Theme(self.attributesFlattened)
            theme.id = themeId

            themeData = surveyTemplateDataframe[surveyTemplateDataframe['themeId'] == themeId]
            clipIdUsed = themeData['clipId'].tolist()[0]
            questionsPerClipUsed = themeData['clipQuestionsIds'].tolist()[0]
            questionsPerClipUsed = ast.literal_eval(questionsPerClipUsed)

            themeData.loc[:,'ambiguitySeverityBaselines'] = themeData['ambiguitySeverityBaselines'].apply(stringArrayToFloatsInPandaRow)

            ambiguitySeverityBaselines = themeData['ambiguitySeverityBaselines'].tolist()[0]
            assert isinstance(ambiguitySeverityBaselines[0][0], np.float) or isinstance(ambiguitySeverityBaselines[0][0], float)

            theme.clip = self.clips[clipIdUsed]
            theme.questions = [None] * len(questionsPerClipUsed)

            for questionIter, questionId in enumerate(questionsPerClipUsed):
                theme.questions[questionIter] = self.questions_byId[questionId]

                # Add the overriding values per question
                theme.questions[questionIter].ambiguity = ambiguitySeverityBaselines[questionIter][0]
                theme.questions[questionIter].severity = ambiguitySeverityBaselines[questionIter][1]
                theme.questions[questionIter].baseline = ambiguitySeverityBaselines[questionIter][2]

            quest.themes[themeId] = theme

        return quest

    # Load responses from database
    def LoadResponsesDataset(self, inputSurvey: Survey, surveyResponsesDataframe=None, globalStatistics=None):
        allUserIds = list(np.unique(surveyResponsesDataframe['userId'].tolist()))

        responsesDataset = SurveyResponses(self.categoriesList, globalStatistics)
        responsesDataset.surveyTemplate = inputSurvey

        for userId in allUserIds:
            dataForUser = surveyResponsesDataframe[surveyResponsesDataframe['userId'] == userId]

            teamId = dataForUser['teamId'].tolist()[0]
            responses = dataForUser['responses'].tolist()[0]
            responses = ast.literal_eval(responses)

            userResponses = UserResponses()
            userResponses.teamId = teamId
            userResponses.userId = userId
            userResponses.responses = responses

            # Add this user responses to the map indexed by team id
            if responsesDataset.allUsersResponses.get(teamId) is None:
                responsesDataset.allUsersResponses[teamId] = []
            responsesDataset.allUsersResponses[teamId].append(userResponses)

            responsesDataset.userIdToTeamId[userId] = teamId
            responsesDataset.invidualUserResponses[userId] = userResponses

            """
            responsesDataset.userIdResponses[userId].append(responses)
            """
        return responsesDataset

    #@profile
    def LoadData(self,
            attrDataframe,
            catDataframe,
            clipsAttributesDataframe,
            clipsMetaDataframe,
            questionsDataframe,
            surveyTemplateDataframe,
            surveyResponsesDataframe,
            globalStatistics: GlobalStatistics,
            dataStoreSettings: DataStoreSettings
    ) -> Tuple[Survey, List[QuestionForClip], SurveyResponses]:
        # Step 1:
        # --------------------------------------------------------------------
        # LOAD all the questions categories and the behavioural attributes
        dataStore = self # DataStore()
        dataStore.LoadAttributesAndCategories(
            attrDataframe=attrDataframe,
            catDataframe=catDataframe,
        )

        # Step 2:
        # --------------------------------------------------------------------
        # LOAD vortexplore library first
        dataStore.LoadClips(
            clipsAttributesDataframe=clipsAttributesDataframe,
            clipsMetaDataframe=clipsMetaDataframe,
        )
        surveyQuestions: List[QuestionForClip] = dataStore.LoadQuestions(questionsDataframe=questionsDataframe)

        # Step 3:
        # --------------------------------------------------------------------
        # LOAD survey template and user responses for a client
        surveyTemplate: Survey = dataStore.LoadSurvey(surveyTemplateDataframe=surveyTemplateDataframe)

        # Load the survey responses
        surveyResponses: SurveyResponses = dataStore.LoadResponsesDataset(
            inputSurvey=surveyTemplate,
            surveyResponsesDataframe=surveyResponsesDataframe,
            globalStatistics=globalStatistics
        )

        # Step 4:
        # --------------------------------------------------------------------
        # Preprocess the client responses in a form that is ready for visualization
        # TODO: this preprocessing can be cached on disk to have fast queries over time
        surveyResponses.preprocessData()

        return surveyTemplate, surveyQuestions, surveyResponses

    # For debugging only !
    # give it the surveyId, the dictionary questionId -> avg deviation for it (raw !)
    # databaseFilePath = path to the csv file
    @staticmethod
    def debug_addDeviationsInDatabase(databaseFilePath, dataToSave):
        import csv
        csv_columns = ['QuestionId', 'AvgDeviation']
        try:
            with open(databaseFilePath, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()

                for questionId, questionDev in dataToSave.items():
                    csvfile.write("{},{}\n".format(questionId, questionDev))
        except IOError:
            print("I/O error")

    # ------------------------------------------------

# Loads a local disk datastore for experimentation purposes
def loadDataStore(optionalPath = None):
    dataStore = DataStore()
    baseFolderPath = os.path.join(os.path.abspath(''), optionalPath) if optionalPath is not None else os.path.abspath('')
    attrDataframe = pd.read_csv(os.path.join(baseFolderPath, "datastore/attributes.csv"))
    catDataframe = pd.read_csv(os.path.join(baseFolderPath, "datastore/categories.csv"))
    dataStore.LoadAttributesAndCategories(attrDataframe=attrDataframe, catDataframe=catDataframe)
    clipsAttributesDataframe = \
        pd.read_csv(os.path.join(baseFolderPath, "datastore/clips_attributes.csv"))
    clipsMetaDataframe = \
        pd.read_csv(os.path.join(baseFolderPath, "datastore/clips_meta.csv"))
    dataStore.LoadClips(clipsAttributesDataframe=clipsAttributesDataframe, clipsMetaDataframe=clipsMetaDataframe)
    questionsDataframe = \
        pd.read_csv(os.path.join(baseFolderPath, "datastore/questions.csv"))
    dataStore.LoadQuestions(questionsDataframe=questionsDataframe)
    return dataStore


# Builds a demo for manual clustering specification
def get_ManualClusteringSpecDemo():
    # Setup a cloud of behavior type / clusters.
    # Note: A ManualSingleClusterSpec can contain many behavior types, but all must share the same feature
    # FIRST class of cluster def
    type1_Spec = ManualSingleClusterSpec(features=[ManualClusterFeature(catName="Sensitivity",
                                                                        listOfAttributes=["Leadership", "Sexual Harassment"],
                                                                        mean=1, dev=1),
                                                    ManualClusterFeature(catName="Awareness",
                                                                        listOfAttributes=["Leadership", "Sexual Harassment"],
                                                                        mean=1, dev=1)],
                                         name="Cluster0-AnormalHuman")

    type2_Spec = ManualSingleClusterSpec(features=[ManualClusterFeature(catName="Sensitivity",
                                                                        listOfAttributes=["Mental Health", "Team Interaction"],
                                                                        mean=3, dev=1),
                                                   ManualClusterFeature(catName="Sanction",
                                                                        listOfAttributes=["Mental Health", "Team Interaction"],
                                                                        mean=2, dev=1)],
                                         name="Cluster1-TheBigLeader")


    type3_Spec = ManualSingleClusterSpec(features=[ManualClusterFeature(catName="Sensitivity",
                                                                        listOfAttributes=["Mental Health", "Team Interaction"],
                                                                        mean=3, dev=1),
                                                    ManualClusterFeature(catName="Awareness",
                                                                        listOfAttributes=["Mental Health", "Team Interaction"],
                                                                        mean=2, dev=1),
                                                    ManualClusterFeature(catName="Sanction",
                                                                        listOfAttributes=["Mental Health", "Team Interaction"],
                                                                        mean=0, dev=1)],
                                         name="Cluster2-TheOnesThatNeedPsyho")


    type4_Spec = ManualSingleClusterSpec(features=[ManualClusterFeature(catName="Sensitivity",
                                                                        listOfAttributes=["Personal Boundaries", "Sexual Harassment"],
                                                                        mean=0, dev=1)],
                                         name="Cluster3-Aliens")

    manualClustersSpec = ManualClustersSpec([type1_Spec, type2_Spec, type3_Spec, type4_Spec], sharedFeaturesTest=False)
    return manualClustersSpec