from DataDefinitions import *
from memory_profiler import profile
from memory_profiler import memory_usage
from memory_profiler import LogFile

import numpy as np
#from StatisticsUtils import *
import VisualizationUtils as Vis

if __name__ == "__main__":
    attrDataframe = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datastore/attributes.csv"))
    catDataframe = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datastore/categories.csv"))
    clipsAttributesDataframe = \
        pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datastore/clips_attributes.csv"))
    clipsMetaDataframe = \
        pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datastore/clips_meta.csv"))
    questionsDataframe = \
        pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datastore/questions.csv"))
    surveyTemplateDataframe = \
        pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datastore/questionnaire.csv"))
    surveyResponsesDataframe = \
        pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datastore/questionnaire_responses.csv"))

    questionsStatsDatabasePath = databaseFilePath=os.path.join(os.path.dirname(os.path.abspath(__file__)), "datastore/avgDeviationsByQuestion.csv")

    def loadGlobalStatistics() -> GlobalStatistics:
        import csv
        globalStatistics = GlobalStatistics()
        globalStatistics.avgDeviationByQuestion = {}
        with open(questionsStatsDatabasePath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                globalStatistics.avgDeviationByQuestion[int(row['QuestionId'])] = float(row['AvgDeviation'])

        return globalStatistics


    globalStatistics = loadGlobalStatistics()
    dataStoreSettings = DataStoreSettings()
    dataStoreSettings.useConstantBiasRemoval = True # False is default let it enabled as True here to see differences.

    dataStore = DataStore()
    questionnaireTemplate, questionsList, questionnaireResponses = dataStore.LoadData(
        attrDataframe=attrDataframe,
        catDataframe=catDataframe,
        clipsAttributesDataframe=clipsAttributesDataframe,
        clipsMetaDataframe=clipsMetaDataframe,
        questionsDataframe=questionsDataframe,
        surveyTemplateDataframe=surveyTemplateDataframe,
        surveyResponsesDataframe=surveyResponsesDataframe,
        globalStatistics=globalStatistics,
        dataStoreSettings=dataStoreSettings,
    )


    if False: # Just for debugging purposes to get some deviations
        avgDeviationsPerQuestions_raw = questionnaireResponses.getAvgQuestionDeviationsPerSurvey(biasedResults=False)
        DataStore.debug_addDeviationsInDatabase(questionStatsDatabasePath,
                                                dataToSave=avgDeviationsPerQuestions_raw)


    Vis.RunDemoVisualizations(questionnaireTemplate, questionsList, questionnaireResponses, dataStore)
