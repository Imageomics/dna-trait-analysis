import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as csv

from gtp.dataloading.tools import butterfly_states_to_ml_ready
from gtp.tools import profile_exe_time


class DataPreprocessor(ABC):
    def __init__(self, input_dir, output_dir) -> None:
        super().__init__()

        self.input_dir = input_dir
        self.output_dir = output_dir

        self.process_ran = False

    def process(self, *args, **kwargs):
        self.process_ran = True
        self._process(*args, **kwargs)

    def save_result(self, output_suffix):
        assert (
            self.process_ran
        ), "DataPreprocess did not run process, so save_result() cannot be run."
        self._save_result(os.path.join(self.output_dir, output_suffix))

    @abstractmethod
    def _save_result(self, path):
        pass

    @abstractmethod
    def _process(self):
        pass


class ButterflyPatternizePreprocessor(DataPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phenotype_data = None

    def _process(self, pca_csv_path_suffix):
        df = pd.read_csv(os.path.join(self.input_dir, pca_csv_path_suffix))
        df = df.rename(columns={"Unnamed: 0": "camid"})
        df.camid = df.camid.apply(lambda x: x.split("_")[0].strip())
        self.phenotype_data = df

    def _save_result(self, path) -> None:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "data.csv")
        self.phenotype_data.to_csv(file_path, index=False)


class ButterflyGenePreprocessor(DataPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.genotype_data = None

    @profile_exe_time
    def _process(self, pca_csv_path_suffix, verbose=False):
        """
        # of rows = # of positions
        We know that columns 1-4 give global information while columns 5 - END are the states related
        to that information for each specimen
        Column 1: Scaffold Location
        Column 2: Position in the scaffold
        Column 3: Reference Allele
        Column 4: Alternative Allele
        Columns 5-(# of specimens): [A-Z0-9]*=[01][\/|][01] ex: CAM016525=0|0
        where the alpha and digits preceeding the = is the specimen id, and
        the structure 0|0 to the right represent the states of two alleles. Each
        of which can be either 1 or 0

        Our goal is to create a dataframe where each row is associated with one specimen
        and the columns will contain the global information previously stated in columns 1-4
        and the states we extract after the '=' symbol.
        """

        @profile_exe_time
        def read_df():
            return pd.read_csv(
                os.path.join(self.input_dir, pca_csv_path_suffix),
                sep="\t",
                header=None,
                engine="pyarrow",
            )

        df = read_df()

        def extract_states(x):
            allele_states = [x.split("=")[1].replace("/", "|") for x in x.tolist()]
            return pd.Series(allele_states)

        def extract_camids(x):
            #! We are assuming all the camids are the same, we are just extracting from the first row
            camid = x.iloc[0].split("=")[0]
            return camid

        @profile_exe_time
        def df_extract_states(df):
            return df.apply(extract_states)
            # return df.map(extract_states_alt)

        df = df.rename(
            {
                0: "Scaffold",
                1: "Position",
                2: "Reference Allele",
                3: "Alternative Allele",
            },
            axis="columns",
        )

        # Remove duplicate positions
        df = df.drop_duplicates(subset=["Position"], keep=False)

        camids = df.iloc[:, 4:].apply(extract_camids)

        df.iloc[:, 4:] = df_extract_states(df.iloc[:, 4:])

        states = df.iloc[:, 4:].T.copy(deep=True)
        states.set_index(camids)
        positions = df["Position"].values.tolist()
        column_dict = {i + 4: camids.values[i] for i in range(len(camids))}
        df = df.rename(columns=column_dict)
        states.columns = positions

        @profile_exe_time
        def create_ml_ready(states):
            ml_ready = butterfly_states_to_ml_ready(states)
            ml_ready = ml_ready.astype(np.bool_)  # Saves significant memory
            return ml_ready

        ml_ready = create_ml_ready(states)

        self.genotype_data = {
            "all_info": df,
            "states": states,
            "camids": np.array(camids.values.tolist()),
            "ml_ready": ml_ready,
        }

    @profile_exe_time
    def _save_result(self, path) -> None:
        os.makedirs(path, exist_ok=True)

        csv.write_csv(
            pa.Table.from_pandas(self.genotype_data["all_info"]),
            os.path.join(path, "all_info.csv"),
        )
        csv.write_csv(
            pa.Table.from_pandas(self.genotype_data["states"], preserve_index=True),
            os.path.join(path, "states.csv"),
        )
        # self.genotype_data["all_info"].to_csv(os.path.join(path, "all_info.csv"), index=False)
        # self.genotype_data["states"].to_csv(os.path.join(path, "states.csv"), index=True)
        np.save(os.path.join(path, "camids.npy"), self.genotype_data["camids"])
        np.save(os.path.join(path, "ml_ready.npy"), self.genotype_data["ml_ready"])
