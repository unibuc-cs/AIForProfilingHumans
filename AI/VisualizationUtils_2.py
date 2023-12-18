from DataDefinitions import *
from colorhash import ColorHash
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import sys
import seaborn as sns

SHOW_LOCAL_PLOTS = False # If try will try to show in the running client the plots, not only saved on disk

# We ar giving the datastore, the cluster specification, the results obtained after scoring each cluster
#, the deviations by categories and attributes, and where to write results if any
class ClusterPlotterHelp:
    def __init__(self, dataStore : DataStore,
                 clustersSpec : ManualClustersSpec,
                 results : UserScoresSurveyResults,
                userDeviationsByCategory : Dict[any, float],
                userAttributesDeviations : Dict[any, float],
                 outFile_clusterScoreCurves = None,
                 outFile_contributions=None,
                 ):
        self.dataStore = dataStore
        self.clustersSpec = clustersSpec
        self.outFile_clusterScoreCurves = outFile_clusterScoreCurves
        self.outFile_contributions = outFile_contributions

        self.results = results
        self.userDeviationsByCategory = userDeviationsByCategory
        self.userDeviationsByAttributes = userAttributesDeviations

        print(self.results)

        # Plot the detailed curves
        if self.outFile_clusterScoreCurves is not None:
            self.plotClusterScoreCurves(self.outFile_clusterScoreCurves)

        # Plot the detailed contributions
        if self.outFile_contributions is not None:
            self.plotContributions(self.outFile_contributions)

    def plotContributions(self, outPrefixFile):
        fig = plt.figure(figsize=(15, 4 * self.clustersSpec.numClusters), dpi=80)
        gs = fig.add_gridspec(self.clustersSpec.numClusters,
                              1, hspace=1.0, wspace =1.0)
        axs = gs.subplots(sharey='row', sharex='row')

        # Get the contributions on attributes and categories
        for i in range(self.clustersSpec.numClusters):
            contributions_dict = self.results.getContributionOfAttrAndCats(i)
            contributions_byAttrs = contributions_dict["attr"]
            contributions_byCat = contributions_dict["cat"]

            totalFeaturesDict = {**contributions_byAttrs, **contributions_byCat}
            totalFeaturesDict_keys = list(totalFeaturesDict.keys())
            totalFeaturesDict_values = [totalFeaturesDict[k] for k in totalFeaturesDict_keys]
            totalFeaturesDict_colors = [self.dataStore.colorByFeatureName[k] for k in totalFeaturesDict_keys]
            #print(totalFeaturesDict_keys)
            #print(totalFeaturesDict_values)
            # Draw the bar
            axs[i].barh(totalFeaturesDict_keys, totalFeaturesDict_values,
                       color=totalFeaturesDict_colors,  height=0.5)

            # Draw the vline
            axs[i].vlines(x=1.0, ymin=-0.5, ymax=len(totalFeaturesDict_keys), colors='black', ls='dotted', lw=2)
            axs[i].set_title(f"Cluster: {self.clustersSpec.clustersNames[i]} : {self.results.outProbabilityPerCluster_normalized[i]}",
                             color= 'orange')

        # Save and show the picture
        plt.savefig(outPrefixFile + ".png")

        if SHOW_LOCAL_PLOTS:
            fig.show()

    def plotClusterScoreCurves(self, outPrefixFile):
        # Add a figure and a grid with each cluster (rows) x features (columns)

        # Configure the big plot with titles per row
        fig = plt.figure(figsize=(8 * self.clustersSpec.maxNumFeatures, 8 * self.clustersSpec.numClusters), dpi=80)
        fig, big_axes = plt.subplots( figsize=(24, 6 * self.clustersSpec.numClusters) ,
                                      nrows=4, ncols=1, sharey=True, sharex=True)
        for row, big_ax in enumerate(big_axes, start=1):
            big_ax.set_title(f"Cluster: {self.clustersSpec.clustersNames[row-1]} : "
                             f"{self.results.outProbabilityPerCluster_normalized[row-1]} \n",
                             color= 'orange', fontsize=16)

            # Turn off axis lines and ticks of the big subplot
            # obs alpha is 0 in RGBA string!
            big_ax.axis('off')
             # removes the white frame
            big_ax._frameon = False

        # Configure the small grid plots with features per columns
        x_labels = list(range(int(-2*MAX_QUESTION_RESPONSE), int(2*MAX_QUESTION_RESPONSE)))
        y_labels = list(np.linspace(start=0.0, stop =1.0, num = 25))

        gridRows = self.clustersSpec.numClusters
        gridCols = self.clustersSpec.maxNumFeatures
        for i in range(gridRows):
            axs_forThisRow = []

            # Configure each feature and prepare
            for j in range(gridCols):
                ax = fig.add_subplot(gridRows,gridCols,i*gridCols + j + 1)
                #ax.set_title('Plot title ' + str(i))
                plt.sca(ax)
                plt.xticks(x_labels, color='black')
                plt.yticks(y_labels, color='black')
                axs_forThisRow.append(ax)

            # Now fill in plot data for row
            self.plotSingleCluster(axs_forThisRow, self.clustersSpec.clusters[i])

        fig.set_facecolor('w')
        plt.tight_layout()
        """"
        fig = plt.figure(figsize=(24, 6 * self.clustersSpec.numClusters), dpi=80)
        gs = fig.add_gridspec(self.clustersSpec.numClusters, self.clustersSpec.maxNumFeatures, hspace=0.2, wspace=0.2)
        axs = gs.subplots() #(sharey='row', sharex='row')

        # Put the labels on x and y for all axes
        x_labels = list(range(int(-2*MAX_QUESTION_RESPONSE), int(2*MAX_QUESTION_RESPONSE)))
        y_labels = list(np.linspace(start=0.0, stop =1.0, num = 25))
        for axRowIndex, axRowData in enumerate(axs):
            for ax in axRowData:
                plt.sca(ax)
                plt.xticks(x_labels, color='black')
                plt.yticks(y_labels, color='black')

            axRowData[0].set_title(f"Cluster: {self.clustersSpec.clustersNames[axRowIndex]} : {self.results.outProbabilityPerCluster_normalized[axRowIndex]}",
                             color= 'orange')

        # Iterate over each cluster plot on the target axes
        for clusterIndex, cluster in enumerate(self.clustersSpec.clusters):
            self.plotSingleCluster(axs[clusterIndex], cluster)
        """

        # Save and show the picture
        plt.savefig(outPrefixFile + ".png")

        if SHOW_LOCAL_PLOTS:
            fig.show()

    def plotSingleCluster(self, axisRowsToDrawOn, singleClusterSpec : ManualSingleClusterSpec):
        clusterName = singleClusterSpec.name
        numTotalFeatures = singleClusterSpec.numFeatures
        numCategories = len(singleClusterSpec.features)

        # Draw each feature data in order, in the received row of axes, first categories then attributes
        for featureIndex in range(numTotalFeatures):
            isCategoryFeature = featureIndex < numCategories
            featureSpec : ManualClusterFeature = singleClusterSpec.features[featureIndex] #\
                #if isCategoryFeature \
                #                                    else singleClusterSpec.attributes_features[featureIndex - numCategories]

            self.plotClusterFeature(axisRowsToDrawOn[featureIndex], f"{featureSpec.name}",
                                    featureMean=featureSpec.mean,
                                    featureDev=featureSpec.dev,
                                    isCategory=isCategoryFeature)

            # Plot a vertical line with the survey deviation for this user's feature
            userSurveyDeviationValue = self.userDeviationsByCategory[featureSpec.name] #if isCategoryFeature \
                                        #else self.userDeviationsByAttributes[featureSpec.catName]

            self.plotUserDeviationOnGraph(axisRowsToDrawOn[featureIndex], devValue=userSurveyDeviationValue)

        # For the unused features, just hide the axis
        for clusterIndexToHide in range(numTotalFeatures, len(axisRowsToDrawOn)):
            axisRowsToDrawOn[clusterIndexToHide].axis('off')

        # set the labels
        for ax in axisRowsToDrawOn:
            ax.set(xlabel='Deviation', ylabel='Probability')


    def plotClusterFeature(self, axisToDrawOn, featureName, featureMean, featureDev, isCategory):
        font1 = {'family':'serif','color': 'blue' if isCategory is True else 'Green','size':14}
        font2 = {'family':'serif','color':'black','size':10}

        curve_color = 'blue'

        axisToDrawOn.set_title(featureName, color= curve_color if isCategory else 'green')
        axisToDrawOn.set(xlabel="Dev", ylabel="Prob")
        self.featureName = featureName
        self.featureMean = featureMean

        self.sigma = math.sqrt(featureDev)
        x = np.linspace(featureMean - 3*self.sigma, self.featureMean + 3*self.sigma, 100)
        axisToDrawOn.plot(x, stats.norm.pdf(x, self.featureMean, self.sigma))
        self.gaussianDist = stats.norm(self.featureMean, self.sigma)


        xdrawmean = self.featureMean
        ydrawmean = self.gaussianDist.pdf(xdrawmean)
        axisToDrawOn.vlines(x=xdrawmean, ymin=0, ymax=ydrawmean, colors='blue', ls='solid', lw=2)

    def plotUserDeviationOnGraph(self, axisToDrawOn, devValue):
        yProbValue = self.gaussianDist.pdf(devValue)
        user_dev_color = 'red'
        # single vline with specific ymin and ymax
        axisToDrawOn.vlines(x=devValue, ymin=0, ymax=yProbValue, colors=user_dev_color, ls=':', lw=2) # label=f'Score for {self.featureName}={y}')
        #plt.legend()

