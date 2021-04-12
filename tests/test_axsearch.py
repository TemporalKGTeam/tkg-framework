from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
import ax
from ax.service.ax_client import AxClient
import numpy as np
from ax.utils.measurement.synthetic_functions import hartmann6

gs = GenerationStrategy(name="sobol+gpei", steps=[
    GenerationStep(model=ax.Models.SOBOL, num_trials=20, min_trials_observed=10, enforce_num_trials=True),
    GenerationStep(model=ax.Models.GPEI, num_trials=-1)])

parameters = [
    {
        "name": "x1",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    {
        "name": "x2",
        "type": "choice",
        "values": [1.0, 2.0],
    },
    {
        "name": "x3",
        "type": "range",
        "bounds": [0.0, 1.0],
    },
    {
        "name": "x4",
        "type": "range",
        "bounds": [0.0, 1.0],
    },
    {
        "name": "x5",
        "type": "range",
        "bounds": [0.0, 1.0],
    },
    {
        "name": "x6",
        "type": "range",
        "bounds": [0.0, 1.0],
    },
]


def dummy_eval(parameters):
    x = np.array([parameters.get(f"x{i + 1}") for i in range(6)])

    import random
    if random.random() < 0.4:
        raise ValueError
    return {'res': hartmann6(x)}


client = AxClient(generation_strategy=gs)
client.create_experiment(name='test', parameters=parameters, objective_name="res", minimize=True)

for i in range(40):
    parameters, trial_index = client.get_next_trial()
    try:
        data = dummy_eval(parameters)
    except ValueError:
        client.log_trial_failure(trial_index=trial_index)
    else:
        client.complete_trial(trial_index=trial_index, raw_data=data)

client.generation_strategy.trials_as_df.to_csv('~/Downloads/test.csv')
