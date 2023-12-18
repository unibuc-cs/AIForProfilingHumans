
import os

from gym.envs.registration import register

register(
    id='EnvQuestionnaireRL-v0',
    entry_point='gymEnvs.envs:EnvQuestionnaireRL',
)
