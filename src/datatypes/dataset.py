from pydantic import BaseModel


class DatasetSpec(BaseModel):
    """
    Specification for a CSV dataset: prefix match, filename, grouping ID column, and volley-text column.
    """
    name: str
    filename: str
    id_col: str
    volley_text: str
    speaker_col: str