import pandas as pd
import numpy as np
import datetime as dt


class CustomAnomaly():

    def inject_anomaly(self, data, magnitude):
        return data * magnitude