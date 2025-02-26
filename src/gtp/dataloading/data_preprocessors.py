import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.csv as csv

from gtp.dataloading.tools import butterfly_states_to_ml_ready, get_ml_state_map
from gtp.tools.timing import profile_exe_time


class DataPreprocessor(ABC):
    def __init__(self, input_dir, output_dir, verbose=False) -> None:
        super().__init__()

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.verbose = verbose

        self.process_ran = False

    def process(self, *args, **kwargs):
        self.process_ran = True
        self._process(*args, **kwargs)

    def save_result(self, output_suffix):
        assert self.process_ran, (
            "DataPreprocess did not run process, so save_result() cannot be run."
        )
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

    def _process_with_pandas(self, pca_csv_path_suffix, verbose=False, process_max_rows=None):
        @profile_exe_time(verbose=False)
        def read_df():
            return pd.read_csv(
                os.path.join(self.input_dir, pca_csv_path_suffix),
                sep="\t",
                header=None,
                engine="pyarrow",
            )

        df = read_df()
        if process_max_rows:
            df = df.iloc[:process_max_rows]

        def extract_states(x):
            allele_states = [x.split("=")[1].replace("/", "|") for x in x.tolist()]
            return pd.Series(allele_states)

        def extract_camids(x):
            #! We are assuming all the camids are the same, we are just extracting from the first row
            camid = x.iloc[0].split("=")[0]
            return camid

        @profile_exe_time(verbose=False)
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

        @profile_exe_time(verbose=False)
        def create_ml_ready(states):
            ml_ready = butterfly_states_to_ml_ready(states)
            ml_ready = ml_ready.astype(np.bool_)  # Saves significant memory
            return ml_ready

        ml_ready = create_ml_ready(states)

        self.genotype_data = {
            "all_info": df,
            "states": states,
            "camids": np.array(camids.values.tolist()),
            "positions": np.array(positions),
            "ml_ready": ml_ready,
        }

    def _process_with_polars(self, pca_csv_path_suffix, verbose=False, process_max_rows=None):
        @profile_exe_time(verbose=False)
        def read_df():
            return pl.read_csv(
                os.path.join(self.input_dir, pca_csv_path_suffix),
                separator="\t",
                has_header=False,
                quote_char=None,
            )

        # Read in DF
        df = read_df()
        if process_max_rows:
            df = df[:process_max_rows]

        df = df.rename(
            {
                "column_1": "Scaffold",
                "column_2": "Position",
                "column_3": "Reference Allele",
                "column_4": "Alternative Allele",
            }
        )

        # Remove duplicate positions
        df = df.unique(subset=["Position"], keep="none")

        # Extract camids
        data_cols = df.columns[4:]
        camids = [x.split("=")[0] for x in df[0, data_cols].rows()[0]]

        # Extract States
        df = df.with_columns(
            pl.col(old_col)
            .str.split_exact("=", 1)
            .struct[1]
            .str.replace("/", "|")
            .alias(old_col)
            for old_col in data_cols
        )

        # Compute Intermediates
        states = df.select(data_cols).transpose()
        state_columns = states.columns
        str_pos = df[:, "Position"].cast(pl.String).to_list()
        states = (
            states.with_columns(
                pl.Series(camids).alias("camids"),
            )
            .rename(dict(zip(state_columns, str_pos)))
            .select(["camids"] + str_pos)
        )

        # Create ML Ready matrix
        values = states.select(
            pl.col(str_pos).str.split("|").cast(pl.List(pl.Int32)).list.sum()
        ).rows()
        np_values = np.array(values)
        one_hot_size = np_values.max() + 1
        ml_ready = np.zeros(np_values.shape + (one_hot_size,))
        ml_ready.reshape(-1, one_hot_size)[
            np.arange(np_values.size), np_values.reshape(-1)
        ] = 1
        ml_ready = ml_ready.astype(np.bool_)

        self.genotype_data = {
            "all_info": df,
            "states": states,
            "camids": np.array(camids),
            "positions": np.array(str_pos),
            "ml_ready": ml_ready,
        }

    @profile_exe_time(verbose=False)
    def _process(self, pca_csv_path_suffix, verbose=False, processor="polars", process_max_rows=None):
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

        match processor:
            case "pandas":
                self._processor = "pandas"
                self._process_with_pandas(pca_csv_path_suffix, verbose=False, process_max_rows=process_max_rows)
            case "polars":
                self._processor = "polars"
                self._process_with_polars(pca_csv_path_suffix, verbose=False, process_max_rows=process_max_rows)
            case _:
                raise NotImplementedError(
                    f"{processor} has not been implemented as a processor for {self.__class__.__name__}"
                )

    @profile_exe_time(verbose=False)
    def _save_result(self, path) -> None:
        os.makedirs(path, exist_ok=True)

        all_info_path = os.path.join(path, "all_info.csv")
        states_path = os.path.join(path, "states.csv")
        match self._processor:
            case "pandas":
                csv.write_csv(
                    pa.Table.from_pandas(self.genotype_data["all_info"]),
                    all_info_path,
                )
                csv.write_csv(
                    pa.Table.from_pandas(self.genotype_data["states"], preserve_index=True),
                    states_path,
                )
            case "polars":
                self.genotype_data["all_info"].write_csv(all_info_path, separator=",")
                self.genotype_data["states"].write_csv(states_path, separator=",")
            case _:
                raise NotImplementedError(f"{self._processor} has not been implemented as a processor in {self.__class__.__name__}")
        # self.genotype_data["all_info"].to_csv(os.path.join(path, "all_info.csv"), index=False)
        # self.genotype_data["states"].to_csv(os.path.join(path, "states.csv"), index=True)
        np.save(os.path.join(path, "camids.npy"), self.genotype_data["camids"])
        np.save(os.path.join(path, "ml_ready.npy"), self.genotype_data["ml_ready"])
        np.save(os.path.join(path, "positions.npy"), self.genotype_data["positions"])
