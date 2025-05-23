import pandas as pd
import numpy as np
import datetime as dt

class SpikeAnomaly():

    def inject_anomaly(self, data, rng, magnitude):
        random_factors = rng.uniform(1, magnitude)
        return data * random_factors