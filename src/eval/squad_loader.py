import pandas as pd
from dataclasses import dataclass
from typing import Iterable, List, Optional

@dataclass(frozen=True)
class SQuADTestCase:
    query: str
    gold_answer: str
    context: str
    label: str # "Answerable" or "Unanswerable"

def iter_squad_cases(csv_path: str) -> Iterable[SQuADTestCase]:
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        yield SQuADTestCase(
            query=row['question'],
            gold_answer=str(row.get('gold_answer', 'N/A')), # SQuAD 2.0 might have empty answers
            context=row['context'],
            label=row['label']
        )