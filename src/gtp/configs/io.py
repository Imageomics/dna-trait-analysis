from dataclasses import dataclass


@dataclass
class RawDataInputConfigs:
    root: str
    genotype: str
    phenotype: str


@dataclass
class DataPostProcessingConfigs:
    root: str
    genotype: str
    phenotype: str


@dataclass
class ResultsIOConfigs:
    root: str
    training_metadata: str


@dataclass
class IOConfigs:
    default_root: str  # Default root. Used if any 'root' option is null
    raw_data_input: RawDataInputConfigs
    data_post_process: DataPostProcessingConfigs
    results: ResultsIOConfigs

    def __post_init__(self):
        self.raw_data_input = RawDataInputConfigs(**self.raw_data_input)
        self.data_post_process = DataPostProcessingConfigs(**self.data_post_process)
        self.results = ResultsIOConfigs(**self.results)
