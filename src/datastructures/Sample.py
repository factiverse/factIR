from typing import Optional

from datastructures.Verdict import Verdict
from datastructures.Evidence import Evidence
from datastructures.Claim import Claim
from dataclasses import dataclass


@dataclass
class Sample:
    """A base class to hold one datapoint/sample with a question and its corresponding verdict.

    Attributes:
        claim (Claim): Claim to be fact-checked.
        verdict (Verdict): verdict of the given claim.
        evidence (Evidence): Optional context/evidence for the given question.
        _idx (int): The ID of the answer.
    """

    idx: int
    claim: Claim
    evidences: Optional[Evidence]
