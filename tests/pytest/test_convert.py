""" This testing class is an integration test suite. 
What it does is consider one of the demo scenarios and test whether we can obtain the same results with the refactored code
"""

import sys
import pytest

sys.path.append("./tests/")
sys.path.append("./tests/refactor-tests/")
import context

import palimpzest as pz
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.elements.records import DataRecord
from palimpzest.operators import LLMConvertConventional, LLMConvertBonded
from palimpzest.operators.datasource import MarshalAndScanDataOp
from palimpzest.constants import PromptStrategy

@pytest.mark.parametrize("convert_op", [LLMConvertBonded, LLMConvertConventional])
def test_convert(convert_op, email_schema):
    """Test whether convert operators"""
    model = pz.Model.GPT_4
    scanOp = MarshalAndScanDataOp(outputSchema=pz.TextFile, dataset_id="enron-eval-tiny")
    convertOp = convert_op(
        inputSchema=pz.File,
        outputSchema=email_schema,
        model=model,
        prompt_strategy=PromptStrategy.DSPY_COT_QA,
    )
 
    datasource = DataDirectory().getRegisteredDataset("enron-eval-tiny")
    candidate = DataRecord(schema=pz.File, parent_id=None, scan_idx=0)
    candidate.idx = 0
    candidate.get_item_fn = datasource.getItem
    candidate.cardinality = datasource.cardinality
    # run DataSourcePhysicalOp on record

    outputs = []
    records, _ = scanOp(candidate)
    for record in records:
        output, _ = convertOp(record)
        outputs.extend(output)

    for record in outputs:
        print(record.sender, record.subject)