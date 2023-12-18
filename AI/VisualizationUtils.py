import operator as op
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, HTML

from DataDefinitions import * #ReduceOpType, PredictionGaussianMixtureMeta, CustomGaussianMixtureMeta, ManualClustersSpec, ManualClusterFeature, ManualClusterScores, Survey
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple
from mpl_toolkits import mplot3d
from VisualizationUtils_2 import *

import csv

# Doc:
"""
level - can be at global or per team
category - can be filtering by a category or not
showBiasedDeviations - if ambiguity and severity should bias the deviation results or not
asViolins - if true, violins instead of boxplots will be shown. They show the distribution around a value
showPoints - if true, individual repsonses will be marked on the graph
useAttributes - if true, attributes will be shown for overviews, not deviations
"""


class ThemeAndQuestionLevelVisualizationDeviationsOptions:
    def __init__(
            self,
            name: str = None,
            categoryName: str = None,
            teamId: int = None,
            showBiasedDeviations: bool = False,
            showViolins: bool = False,
            showPoints: bool = True,
            saveFigurePath: str = None,
    ):
        self.categoryName = categoryName
        self.teamId = teamId
        self.showBiasedDeviations = showBiasedDeviations
        self.showViolins = showViolins
        self.name = name
        self.showPoints = showPoints
        self.saveFigurePath = saveFigurePath


def showDeviations_themeAndQuestionLevel(
        options: ThemeAndQuestionLevelVisualizationDeviationsOptions,
        surveyTemplate: Survey,
        surveyResponses: SurveyResponses
):
    # Select the data to show on the graph according to the given option
    if options.teamId is None:
        if options.showBiasedDeviations:
            aggregatedDataChosen = surveyResponses.aggregatedDeviations_Global_biased
        else:
            aggregatedDataChosen = surveyResponses.aggregatedDeviations_Global_raw
    else:
        if options.showBiasedDeviations:
            aggregatedDataChosen = surveyResponses.aggregatedDeviations_PerTeam_biased[options.teamId]
        else:
            aggregatedDataChosen = surveyResponses.aggregatedDeviations_PerTeam_raw[options.teamId]

    # Show a graph of question results for each individual theme
    for themeIndex in range(len(surveyTemplate.themes)):
        sortedDict = sorted(aggregatedDataChosen[themeIndex].items(), key=op.itemgetter(0))

        # If requested, filter out questions that are not in the requested category
        if options.categoryName is not None:
            templateQuestionsForTheme = surveyTemplate.themes[themeIndex]
            questionsIdsToDelete = []
            for questionId, _ in enumerate(sortedDict):
                categoryOfQuestion = templateQuestionsForTheme.questions[questionId].category
                if (categoryOfQuestion not in options.categoryName) \
                        and (options.categoryName not in categoryOfQuestion):
                    questionsIdsToDelete.append(questionId)
            for questionId in sorted(questionsIdsToDelete, reverse=True):
                del sortedDict[questionId]

        if len(sortedDict) == 0:
            continue

        sorted_keys, sorted_values = zip(*sortedDict)
        # almost verbatim from question

        sns.set(context='notebook', style='whitegrid')
        if options.showViolins:
            ax = sns.violinplot(data=sorted_values, size=6)
        else:
            ax = sns.boxplot(data=sorted_values, width=.18)
        ax.set(xlabel="Question Index", ylabel="Deviations", title=f'{options.name} theme: {themeIndex}')

        if options.showPoints:
            sns.swarmplot(data=sorted_values, size=4, edgecolor="black", linewidth=.9)

        # category labels
        plt.xticks(plt.xticks()[0], sorted_keys)

        if options.saveFigurePath is not None:
            plt.savefig(f"{options.saveFigurePath}_{themeIndex}.png")

        plt.show()

    plt.clf()


class QuestionsCategoriesLevelVisualizationDeviationsOptions:
    def __init__(
            self,
            name: str = None,
            teamId: int = None,
            categoryName: str = None,
            useAttributes: bool = False,
            showBiasedDeviations: bool = False,
            saveFigurePath: str = None,
    ):
        self.name = name
        self.teamId = teamId
        self.categoryName = categoryName
        self.showBiasedDeviations = showBiasedDeviations
        self.useAttributes = useAttributes
        self.saveFigurePath = saveFigurePath


# This function shows the deviations on categories as an overview plot if attributes option is not used
# If attributes option is used, then an overview of how the attribute values are per team and global level
def showDeviations_questionsCategoriesLevelOverview(
        options: QuestionsCategoriesLevelVisualizationDeviationsOptions,
        surveyTemplate : Survey,
        surveyQuestions : List[QuestionForClip],
        surveyResponses: SurveyResponses,
):
    def getAggregatedDataChosen():
        # First, from which data do we want to plot ?
        if options.useAttributes:  # Attributes stats to show
            if options.teamId:
                if options.showBiasedDeviations:
                    result = surveyResponses.overviewAttributes_PerTeam_biased[options.teamId]
                else:
                    result = surveyResponses.overviewAttributes_PerTeam_raw[options.teamId]
            else:
                if options.showBiasedDeviations:
                    result = surveyResponses.overviewAttributes_Global_biased
                else:
                    result = surveyResponses.overviewAttributes_Global_raw
        else:
            if options.teamId:
                if options.showBiasedDeviations:
                    result = surveyResponses.overviewDeviationsByCategory_PerTeam_biased[options.teamId]
                else:
                    result = surveyResponses.overviewDeviationsByCategory_PerTeam_raw[options.teamId]
            else:
                if options.showBiasedDeviations:
                    result = surveyResponses.overviewDeviationsByCategory_Global_biased
                else:
                    result = surveyResponses.overviewDeviationsByCategory_Global_raw

        return result

    def showDeviations():
        def showDeviationForCurrentElement(elementId, xlabel, title):
            data = aggregatedDataChosen[elementId]

            sns.set(context='notebook', style='whitegrid')
            ax = sns.distplot(data, kde_kws={'bw': 1.5})
            ax.set(xlabel=xlabel, ylabel='Percent', title=title)

            if options.saveFigurePath is not None:
                plt.savefig(f"{options.saveFigurePath}_{elementId}.png")

            plt.show()

        if options.useAttributes:
            for attrId in surveyResponses.allAttributesUsedInDataset:
                showDeviationForCurrentElement(
                    elementId=attrId,
                    xlabel=f'Attribute {attrId} deviation scores',
                    title=f'{options.name}',
                )
        else:
            if options.categoryName:
                if options.categoryName in questionCategoriesUsedInCurrentSurvey:
                    showDeviationForCurrentElement(
                        elementId=options.categoryName,
                        xlabel='Deviation values',
                        title=f'{options.name} category {options.categoryName}',
                    )
            else:
                for questionCategory in questionCategoriesUsedInCurrentSurvey:
                    showDeviationForCurrentElement(
                        elementId=questionCategory,
                        xlabel='Deviation values',
                        title=f'{options.name} category {questionCategory}',
                    )

    questionCategoriesUsedInCurrentSurvey = {x.category for x in surveyQuestions}
    aggregatedDataChosen = getAggregatedDataChosen()
    showDeviations()
    plt.clf()


# Params description:
# itemsTo show is a list of items to correlate between that the client should choose. See the documentation for full
# range of features
# teamId can be None (correlation at global level) or the ID of one of the teams that one or more of the survey
# respondents belong to (correlation at that team's level)
# tresholdValue = The minimum correlation value to show correlations between
class VisualizationCorrelationOptions:
    def __init__(self, itemsToShow, teamId, saveFigurePath=None, thresholdValue = None):
        self.itemsToShow = itemsToShow
        self.teamId = teamId
        self.saveFigurePath = saveFigurePath
        self.thresholdValue = thresholdValue



class StatsOptions:
    def __init__(self, teamId):
        self.teamId = teamId


# Prints all the correlation statistics
def printStats(options: StatsOptions, surveyResponses: SurveyResponses):
    dataSelected = surveyResponses.statsTable \
        if options.teamId is None \
        else surveyResponses.statsTable_perTeam[options.teamId]
    display(HTML(dataSelected.to_html()))

# Params description:
# itemsTo show is a list of items to correlate between that the client should choose. See the documentation for full range of features
# teamId can be NONE - which means we want correlation at global level or not NONE - at a given team level

# Shows the correlation matrix between the given features
def showCorrelationMatrix(options : CorrelationOptions,
                          questionnaireResponses : SurveyResponses):

    corrListSorted, corrMatrix = questionnaireResponses.getCorrelationsBetweenStats(options)
    print (corrListSorted)


    mask = np.triu(np.ones_like(corrMatrix, dtype=np.bool)) # Show only upper triangular matrix
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corrMatrix, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink":.5})

    if options.saveFigurePath is not None:
        plt.savefig(f"{options.saveFigurePath}.png")

    plt.show()

# If userID is provided, it means we are only evaluating at user level.
# If userId is none, but teamID is not = > we are evaluating at the given teamID level
# If both userId and userID are none, we are evaluating at the global / organization level
# You must give the clusters specification, the datastore and where to write output - could be none
def showDetailedClustering(userId=None, teamId=None,
                           questionnaireResponses : SurveyResponses = None,
                           useBiased = False, statType = ReduceOpType.MEAN,
                           manualClustersSpec : ManualClustersSpec = None,
                           localDataStore : DataStore = None,
                           outFile_clustersScore = None,
                           outFile_clustersContributions = None):


    userDeviations_byCategory = {}
    userDeviations_byAttribute = {}

    displayedUserId = str(userId) if userId is not None else (str(teamId) if teamId is not None else "Global")

    # Get the stats table to use
    statsTableToUse = None
    if userId is not None: # User level !
        statsTableToUse = questionnaireResponses.statsTable_perUser_raw[userId] if useBiased is False \
                            else questionnaireResponses.statsTable_perUser_biased[userId]
    elif teamId is not None:  # Team level
        statsTableToUse = questionnaireResponses.statsTable_perTeam_raw[teamId] if useBiased is False \
                            else questionnaireResponses.statsTable_perTeam_biased[teamId]
    else: # Global level
        statsTableToUse = questionnaireResponses.statsTable_raw if useBiased is False \
                            else questionnaireResponses.statsTable_biased[teamId]

    for catName in questionnaireResponses.categoriesList:
        catDev = statsTableToUse[f'{catName}_{statType.name}'].mean()
        #assert len(catDev) == 1, "Why having multiple indices there ?"
        userDeviations_byCategory[catName] = catDev

    for attrId in questionnaireResponses.allAttributesUsedInDataset:
        attrDev = statsTableToUse[attrId].mean()
        #assert len(attrDev) == 1, "Why having multiple indices there ?"
        userDeviations_byAttribute[attrId] = attrDev

    # Compute survey results...

    # Get first the list of all responses that we need to analyze according to the input parameters
    #teamOfUser = questionnaireResponses.userIdToTeamId[userId]
    allUserResponsesToAnalyze = []
    if userId is not None: # Individual user given ?
        allUserResponsesToAnalyze.append(questionnaireResponses.invidualUserResponses[userId])
    elif teamId != None: # Team level ?
        allUserResponsesToAnalyze.extend(questionnaireResponses.allUsersResponses[teamId])
    else: # Global level ?
        for teamId, allResponsesPerTeam in questionnaireResponses.allUsersResponses.items():
            allUserResponsesToAnalyze.extend(allResponsesPerTeam)


    # Take and combine all users analyzed into a global aggregated results
    res: UserScoresSurveyResults = None
    for userResponseToAnalyze in allUserResponsesToAnalyze:
        userResponseToAnalyze.statType = statType
        userResponseToAnalyze.useRawData = not useBiased
        localRes : UserScoresSurveyResults =  SurveyResponses.scoreUserSurveyToClustersSpec(displayedUserId,
                                             userDeviationsByCategory=None, #userDeviations_byCategory,
                                             userAttributesDeviations=None,#userDeviations_byAttribute,
                                             userAttributesPerCategoryDeviations=None,#
                                             userResponse=userResponseToAnalyze,
                                             clustersSpec=manualClustersSpec,
                                            finalizeInEnd=False)

        if res is None:
            res = localRes
        else:
            res.merge(localRes)

    res.finalize()


    clusterPlortterHelper = ClusterPlotterHelp(clustersSpec=manualClustersSpec,
                                               dataStore=localDataStore,
                                               results=res,
                                               userDeviationsByCategory=userDeviations_byCategory,
                                               userAttributesDeviations=userDeviations_byAttribute,
                                               outFile_clusterScoreCurves=outFile_clustersScore,
                                               outFile_contributions=outFile_clustersContributions)


def showDetailedClustering_demo(questionnaireResponses : SurveyResponses,
                                clusterSpec : ManualClustersSpec,
                                dataStore : DataStore):
    # Demo 1: show at user level
    showDetailedClustering(userId=127, teamId=None, questionnaireResponses=questionnaireResponses,
                           useBiased=False,
                           statType=ReduceOpType.MEAN,
                           manualClustersSpec=clusterSpec,
                           localDataStore=dataStore,
                           outFile_clustersScore="dataout/User127_detailedClustering",
                           outFile_clustersContributions="dataout/User127_detailedClusterContributions")


    # Demo 2: show at team level
    showDetailedClustering(userId=None, teamId=15, questionnaireResponses=questionnaireResponses,
                           useBiased=False,
                           statType=ReduceOpType.MEAN,
                           manualClustersSpec=clusterSpec,
                           localDataStore=dataStore,
                           outFile_clustersScore="dataout/team15_detailedClustering",
                           outFile_clustersContributions="dataout/team15_detailedClusterContributions")

    # Demo 3: show at global level
    showDetailedClustering(userId=None, teamId=None, questionnaireResponses=questionnaireResponses,
                           useBiased=False,
                           statType=ReduceOpType.MEAN,
                           manualClustersSpec=clusterSpec,
                           localDataStore=dataStore,
                           outFile_clustersScore="dataout/global_detailedClustering",
                           outFile_clustersContributions="dataout/global_detailedClusterContributions")


def plotBehaviorsClusters(centroids : np.array,  # Each centroid on a row
                          featureData, # The data we want to plot on top of the centroids
                          predictions : np.array, # For each item in feature data, what is the predicted label for each ?
                          bounds_min,  # The min and max bounds of the values to be show on each axis
                          bounds_max,
                          axesLabels, # Name for each axis
                          savePath : str, # Where to save the visualization
                          title : str): # What is the title of the visualization
    numClasses = centroids.shape[0]
    dimensionOfOutput = featureData.shape[1]

    assert len(predictions) == len(featureData)
    assert len(np.where(np.logical_or(predictions < 0, predictions >= numClasses))), "Incorrect predictions input ! Check it please"

    if dimensionOfOutput < 2 or dimensionOfOutput > 3:
        print(f"ERROR: can plot only 2D or 3D dimensions ! {dimensionOfOutput} dimensions were requested")
        return

    # Establish the colors for prediction classes
    from itertools import cycle
    cycol = cycle('bgrycmko')
    colormap = np.array([next(cycol) for i in range(numClasses)])
    colorsToUse = colormap[predictions]

    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(title, fontsize=20)

    if dimensionOfOutput == 2:
        ax = plt.axes()

        ax.set_xlabel(axesLabels[0], fontsize=20)
        ax.set_ylabel(axesLabels[1], fontsize=20)

        # Plot the data
        scatterRes = ax.scatter(featureData[:, 0], featureData[:, 1], s=30, c=colorsToUse, label=predictions, alpha=0.5)

        # Plot the centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], s=300, c=colormap, marker='X')

        ax.set_xlim(xmin=bounds_min[0], xmax=bounds_max[0])
        ax.set_ylim(ymin=bounds_min[1], ymax=bounds_max[1])

    elif dimensionOfOutput == 3:
        ax = plt.axes(projection='3d')

        ax.set_xlabel(axesLabels[0], fontsize=20)
        ax.set_ylabel(axesLabels[1], fontsize=20)
        ax.set_zlabel(axesLabels[2], fontsize=20)

        # Plot the data
        scatterRes = ax.scatter3D(featureData[:, 0], featureData[:, 1], featureData[:, 2],
                                  s=30, c=colorsToUse, label=predictions, cmap='Greens')

        # Plot the centroids
        ax.scatter3D(centroids[:, 0], centroids[:, 1], s=300, c=colormap, marker='X')

        ax.set_xlim(left=bounds_min[0], right=bounds_max[0])
        ax.set_ylim(bottom=bounds_min[1], top=bounds_max[1])
        ax.set_zlim(bottom=bounds_min[2], top=bounds_max[2])

        # Draw the legend
        classes = ['Type ' + str(i) for i in range(1, numClasses + 1)]
        class_colours = colormap
        recs = []
        for i in range(0, len(class_colours)):
            recs.append(mpatches.Rectangle((0, 0), 5, 5, fc=class_colours[i]))
        plt.legend(recs, classes, loc=4)

    else:
        raise NotImplementedError

    if savePath is not None:
        plt.savefig(f"{savePath}.png")

    plt.clf()


def showClusteringVisualizations_autoClustering(questionnaireResponses):
    # The following example performs clusterization at a global level then compares the results global vs one team using plots and log data.
    # Similar things can be done to compare between: organizations, teams, groups of individuals !
    # You can comment or add more input categories below to obtain different graphs !

    # TODO Iancu: all these below inputs MUST exist on the GUI interface
    listOfCategoryIds = ["Sensitivity", "Awareness"]
    listOfAttributesIds = ["Leadership", "Sexual Harassment"]
    inputNumClusters = 4 # TODO Iancu : THIS CAN BE SET AS -1 , in this case, the number of clusters will be automatically adjust to the value that fits best the data presented for trianing !! See also another comment related to this below
    useRawData = True # SHould we use raw or biased data ?
    teamId = None       # The team id. None for global
    saveFeaturesAsCsv = 'featuresUsed.csv' # Where to save the features used for debugging purposes
    statType = ReduceOpType.MEAN    # Let this as default...we could also use MEDIAN as well
    # So initially the user is requesting N= len(listOfCategoryIds) + len(listOfAttributesIds) components.
    # This parameter means how much to reduce from original data to the plotted data !
    # Valid values are 2 and 3 only !
    numMaxComponentsDesired = 3 # TODO Iancu: have a GUI interface to modify this too !

    # TODO Iancu: have a GUI interface to get the statistics and global level then compare with individual teams !

    # Step 1: gather the features according to the input
    globalfeaturesDataFrame = questionnaireResponses.getResponsesFeaturesData(listOfCategoryIds, listOfAttributesIds,
                                                                        teamId=teamId, statType=statType,
                                                                        useRawData=useRawData)
    # The prepare function is able to reduce the dimensionality  of the input as requested by user using PCA
    features_global, isDimensionReduced = questionnaireResponses.BehaviorsAutoClusterization_prepareData(globalfeaturesDataFrame, numMaxComponentsDesired)

    if saveFeaturesAsCsv is not None:
        print("The csv file containing all features at global level has been saved to ", saveFeaturesAsCsv)
        globalfeaturesDataFrame.to_csv(saveFeaturesAsCsv)

    # Step 2: fit a model that explain the data seen
    # Note that the values sent as predict can be modified and sent back to you in a post processed / modified form

    # TODO Iancu: have an option for both things !
    # 2.A An example with predefined number of clusters
    #gmmDistribution, gmmMeta = questionnaireResponses.BehaviorsAutoClusterization_fit(numClusters=inputNumClusters, featuresData=features_global) # global or team id level..
    # 2.B An example with automatically finding the number of clusters
    gmmDistribution, gmmMeta = questionnaireResponses.BehaviorsAutoClusterization_fit(numClusters=-1, featuresData=features_global, minClusters=2, maxClusters=5) # global or team id level..


    # Step 3: let's use the model to predict and draw statistics !
    # First, let's use the same data sample at a global level
    predictionResult_global = questionnaireResponses.BehaviorsAutoClusterization_predict(gmmDistribution, gmmMeta, features_global)

    # Step 4: now, for multiple or even ALL teams, predict and compare the results !
    # 4.1 first get the specific team data sample. Note that we are using THE SAME PARAMETERS AS ABOVE, EXCEPTING THE TEAMID !!
    customTeamId = 15
    dataSampleOfCustomTeam = questionnaireResponses.getResponsesFeaturesData(listOfCategoryIds, listOfAttributesIds,
                                                    teamId=customTeamId, statType=statType,
                                                    useRawData=useRawData)

    # As with the global dataset, the team level set needs the same preparation
    features_team15, isDimensionReduced = questionnaireResponses.BehaviorsAutoClusterization_prepareData(dataSampleOfCustomTeam, numMaxComponentsDesired)

    predictionResult_team15 = questionnaireResponses.BehaviorsAutoClusterization_predict(gmmDistribution, gmmMeta, features_team15) # Note that here we use the team extracted sample


    # Step 5: Now let's plot / show results:
    # 5.1 LOGS folders : these needs to be somehow on the user interface !!!
    # TODO Iancu: have a GUI interface  to show these, discuss with hakon what is important from here.
    # Note that when the dimensionality was reduced, the centroids don't have too much meaning in this context
    with open('dataout/predictions_global.txt', 'w') as f:
        f.write("GlobalLevel Predictions:")
        f.write('\n'.join(["%s = %s" % (k, v) for k, v in predictionResult_global.__dict__.items()]))

    with open('dataout/predictions_team15.txt', 'w') as f:
        f.write(f"Team {customTeamId} Predictions")
        f.write('\n'.join(["%s = %s" % (k, v) for k, v in predictionResult_team15.__dict__.items()]))

    # 5.2 plot graphics only if 2D or 3D level. If not, we can handle this through PCA to reduce the dimensionality to 3D or 2D space first

    # The name of the labels are by default the list of attributes and categories.
    # HOWEVER, if PCA was used, then we'll get names such as Principal component 1, 2 ...
    labels = (listOfCategoryIds + listOfAttributesIds) if isDimensionReduced is None \
                else [f"Principal Component {i}" for i in range(features_global.shape[1])]

    # Compute the minimum and maximum values for each feature in the global data.
    # These will be used to compare against every team to have a proper comparison
    global_features_data_mins = np.min(features_global, axis=0)
    global_features_data_max  = np.max(features_global, axis=0)

    # TODO Iancu: observe the difference between the above (global) and this one (per team level).
    # We use the same centroids and bounds for showing the graphs, same feature names. Only the feature data used and predictions are different
    plotBehaviorsClusters(centroids=gmmMeta.centroids,
                            featureData=features_global,
                            predictions=predictionResult_global.predictions,
                            bounds_min=global_features_data_mins,
                            bounds_max=global_features_data_max,
                            axesLabels=labels,
                            savePath='dataout/predictions_global.jpg',
                            title="Clusterization by deviations features")

    plotBehaviorsClusters(centroids=gmmMeta.centroids,
                            featureData=features_team15,
                            predictions=predictionResult_team15.predictions,
                            bounds_min=global_features_data_mins,
                            bounds_max=global_features_data_max,
                            axesLabels=labels,
                            savePath='dataout/predictions_team15.jpg',
                            title="Clusterization by deviations features")

def showClusteringVisualizations_manualClustering(questionnaireResponses):
    # TODO Iancu: all this functionality must exist in the GUI as well...description below
    # The following example performs clusterization at a global level then compares the results global vs one team using plots and log data.
    # Similar things can be done to compare between: organizations, teams, groups of individuals !
    # You can comment or add more input categories below and values to obtain different graphs !

    # Step 0: API for Manual specifications of clusters method
    # TODO Iancu: this is something that you need to put on GUI: basically, we specify:
    #  - the list of features used, i.e. either category or attribute):
    #  - for each feature, what should be the ideal mean of that cluster, and the standard deviation  (e.g. mean could be 5, standard deviation +-2)

    # This is a specification of 3 behavior types inside an organization using the same type of attributes and categories
    manualClusterInstance = get_ManualClusteringSpecDemo()


    # TODO Iancu: all these below inputs MUST exist on the GUI interface
    useRawData = True # SHould we use raw or biased data ?
    teamId = None       # The team id. None for global
    statType = ReduceOpType.MEAN    # Let this as default...we could also use MEDIAN as well

    # TODO Iancu: have a GUI interface to get the statistics and global level then compare with individual teams !

    # Step 1: Get the scores at global level first...
    users_features_global, scores_global = questionnaireResponses.getManualClusteringScoresDistribution(clustersSpec=manualClusterInstance,
                                                                                                teamId=teamId, statType=statType,
                                                                                                useRawData=useRawData)
    users_features_global = users_features_global[:, 1:] # Cut the user id from users features, not needed after


    # Step 2: Not at one of the teams...
    customTeamId = 15
    users_features_customTeam, scores_customTeam = questionnaireResponses.getManualClusteringScoresDistribution(clustersSpec=manualClusterInstance,
                                                                                                teamId=customTeamId, statType=statType,
                                                                                                useRawData=useRawData)
    users_features_customTeam = users_features_customTeam[:, 1:] # Cut the user id from users features, not needed after

    # Compute the minimum and maximum values for each feature in the global data.
    # These will be used to compare against every team to have a proper comparison
    global_features_data_mins = np.min(users_features_global, axis=0)
    global_features_data_max  = np.max(users_features_global, axis=0)

    # Step 3: Now let's plot / show results:
    # 3.1 Logs containing the probability of each person being a type of behavior
    scores_global.writeStats("dataout/manualClustering_globalStats.csv")
    scores_customTeam.writeStats(f"dataout/manualClustering_{customTeamId}.csv")

    # 3.2 plot visuals if 2D or 3D features are used
    plotBehaviorsClusters(centroids=manualClusterInstance.centroids, featureData= users_features_global, predictions=scores_global.predictions,
                          bounds_min=global_features_data_mins, bounds_max=global_features_data_max,
                          axesLabels=manualClusterInstance.featureNames,
                          savePath='dataout/predictions_manual_global',
                          title="Clusterization by deviations features - global")

    # TODO Iancu: observe the difference between the above (global) and this one (per team level).
    # We use the same centroids and bounds for showing the graphs, same feature names. Only the feature data used and predictions are different
    plotBehaviorsClusters(centroids=manualClusterInstance.centroids, featureData= users_features_customTeam, predictions=scores_customTeam.predictions,
                          bounds_min=global_features_data_mins, bounds_max=global_features_data_max,
                          axesLabels=manualClusterInstance.featureNames,
                          savePath=f'dataout/predictions_manual_{customTeamId}',
                          title=f"Clusterization by deviations features - team {customTeamId}")


def RunDemoVisualizations(questionnaireTemplate : Survey,
                          questionsList : List[QuestionForClip],
                          questionnaireResponses : SurveyResponses,
                          dataStore : DataStore):

    # Get a demo of clusters for manual clustering specification
    demoOfClustersSpecs = get_ManualClusteringSpecDemo()

    # Show detailed clustering - per user working only for now
    showDetailedClustering_demo(questionnaireResponses = questionnaireResponses,
                                clusterSpec=demoOfClustersSpecs,
                                dataStore=dataStore)


    #---------
    return

    # --------
    # AUtomatic clustering demo
    showClusteringVisualizations_autoClustering(questionnaireResponses)

    # Manual clustering demo
    showClusteringVisualizations_manualClustering(questionnaireResponses)

    # Get per survey instance the avg deviations per each question used
    avgDeviationsPerQuestions_raw = questionnaireResponses.getAvgQuestionDeviationsPerSurvey(biasedResults=False)
    avgDeviationsPerQuestions_biased = questionnaireResponses.getAvgQuestionDeviationsPerSurvey(biasedResults=True)
    print(f"Raw deviations: \n {avgDeviationsPerQuestions_raw}")
    print(f"Biased deviations: \n {avgDeviationsPerQuestions_biased}")

    # Output some visualization examples
    # Category 1: Visualization of responses for each theme and question in the suvery
    # The viewer can be customized through parameters for individual teams, organization level,
    # violins vs box plots that show medians, percentile, outliers identifications.
    # Can also customize per category of questions. All above on different permutation: e.g. per team and category + violin
    # Eg. 1 Team level visualizations - as box plot and with individual points
    TEAM_TO_SHOW = 15


    options = ThemeAndQuestionLevelVisualizationDeviationsOptions(name="Per team deviations",
                                       teamId=TEAM_TO_SHOW,
                                       showBiasedDeviations=False,
                                       showViolins=False,
                                       showPoints=True,
                                       saveFigurePath="dataout/figure1.png")


    showDeviations_themeAndQuestionLevel(options, questionnaireTemplate, questionnaireResponses)

    # Eg. 2 Global level visualizations
    # Recommended to see as violin because there are too many, and hide individual points
    options = ThemeAndQuestionLevelVisualizationDeviationsOptions(name="Global deviations",
                                       teamId=None,
                                       showBiasedDeviations=False,
                                       showViolins=True,
                                       showPoints=False)
    showDeviations_themeAndQuestionLevel(options, questionnaireTemplate, questionnaireResponses)

    # Eg. 3 TEam level visualizations on a given category
    options = ThemeAndQuestionLevelVisualizationDeviationsOptions(name="Per team deviations - Sensitivity",
                                       teamId=TEAM_TO_SHOW,
                                       categoryName="Sensitivity",
                                       showBiasedDeviations=False,
                                       showViolins=False,
                                       showPoints=True)
    showDeviations_themeAndQuestionLevel(options, questionnaireTemplate, questionnaireResponses)

    ###############
    # E.g. 4,5 Global and team level view of deviations for each category
    # Plots will show the histogram and Kernel density estimate (KDE) of the data gathered
    options = QuestionsCategoriesLevelVisualizationDeviationsOptions(name="Overview global deviations - ",
                                                                    teamId=None,
                                                                    categoryName=None,
                                                                    useAttributes=False,
                                                                    showBiasedDeviations=False,
                                                                    saveFigurePath="dataout/overviews")


    showDeviations_questionsCategoriesLevelOverview(options, questionnaireTemplate, questionsList, questionnaireResponses)

    options = QuestionsCategoriesLevelVisualizationDeviationsOptions(name=f"Team {TEAM_TO_SHOW} deviations - ",
                                                                    teamId = TEAM_TO_SHOW,
                                                                    categoryName = None,
                                                                    useAttributes = False,
                                                                    showBiasedDeviations = False,
                                                                    saveFigurePath = "dataout/overviews")

    showDeviations_questionsCategoriesLevelOverview(options, questionnaireTemplate, questionsList, questionnaireResponses)
    """

    """
    # E.g. 5,6 Global and team level view of deviations for each hidden attribute
    # Plots will show the histogram and Kernel density estimate (KDE) of the data gathered
    options = QuestionsCategoriesLevelVisualizationDeviationsOptions(name="Overview global attributes scores - ",
                                                                        teamId=None,
                                                                        categoryName=None,
                                                                        showBiasedDeviations=False,
                                                                        useAttributes=True)

    showDeviations_questionsCategoriesLevelOverview(options, questionnaireTemplate, questionsList, questionnaireResponses)

    options = QuestionsCategoriesLevelVisualizationDeviationsOptions(name=f"Team {TEAM_TO_SHOW} attributes scores - ",
                                                                   teamId=TEAM_TO_SHOW,
                                                                   categoryName=None,
                                                                   showBiasedDeviations=False,
                                                                   useAttributes=True)
    showDeviations_questionsCategoriesLevelOverview(options, questionnaireTemplate, questionsList, questionnaireResponses)


    # Show correlations - Options:
    # teamId - GLOBAL LEVEL if teamId = None or a team level if you specify one integer value not None
    # itemsToShow - Default should be None. What you can do with this: filter various columns by selecting them, or by default leave them as NULL and we'll handle correlations between everything !
    # threshold - what is the minimum value to return the list of correlated values (between 0 min, 1 max).
    # Internally they select and returns a sorted list of tuples [(stat A, statB, correlations])
    L1 = [f"Sensitivity_{ReduceOpType.MEAN.name}",
                         f"Rumours",
                         f"Leadership",
                         f"Mental Health"]

    #options = CorrelationOptions(itemsToShow=L1, teamId=None, threshold=0.5, saveFigurePath="dataout/correlations_global", )
    #showCorrelationMatrix(options, questionnaireResponses)

    # Show correlations - TEAM LEVEL
    print("Showing raw, global correlations !")
    options = CorrelationOptions(itemsToShow=None, teamId=5, threshold=0.6, saveFigurePath="dataout/correlations_team0", useBiasedValues=False)
    showCorrelationMatrix(options, questionnaireResponses)


    options = CorrelationOptions(itemsToShow=None, teamId=None, threshold=0.05, saveFigurePath="dataout/correlations_global", questionFilteredId=None)
    showCorrelationMatrix(options, questionnaireResponses)
    #print(questionnaireResponses)
