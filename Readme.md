# Adaptive questionnaire design using various types of AI agents for people profiling

This is a tool for automatic profiling and clustering of persons by conducting interviews.

The novelty of the work is that instead of using HR professionals, we use AI for this task!

More than that, the AI asking questions will adapt to the human being evaluated, asking questions dynamically to adapt and classify as correctly as possible.


## AI Methods 

We implement several AI agents implemented in Tensorflow 2.0. These implementations can be found in the AI subfolder.

* A reinforcement learning-based method using a policy-based algorithm, A2C method (best one at the moment for our use-case): AI/AgentRL_A2C.py
* The DQN method, using a double queue and replay buffer prioritization: AI/AGENTDDQN.py
* The PathfindingAI agent for evaluation purposes (https://www.scitepress.org/PublicationsDetail.aspx?ID=VwhqRAApZHo=&t=1): AI/AgentPathfinding.py.
 
## Datasets and visualization tools demonstration
* An anonymized dataset is available in the CSV files inside the datastore folder.
* Different experiments visualization in "AI/dataStore", while logs in CSV format in "logs". These consider the explainability of the models, correlations, viewing the results from different axes, dimensionality reduction using PCA, etc.
* The tools are implemented in AI/VisualizationUtils.py

## RL environment
* A synthetic environment for the RL methods and a collection of synthetic data generation tools are available in gymEnvs/envs/EnvQuestionnairRL.py.
* We use the OpenAI Gym interfaces, so you can swap with your own interfaces and libraries at any moment.

## Other failed or work-in-progress experiments
* Various other attempts are provided in the prototyping subfolder.
