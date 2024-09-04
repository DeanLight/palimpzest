# Note: include in the tests folder a .env file that contains the API keys for the services used in the tests
import os
if not os.environ.get('OPENAI_API_KEY'):
    import context
    
from palimpzest.constants import PZ_DIR
import palimpzest as pz

import pandas as pd
import time

pz.DataDirectory().clearCache(keep_registry=True)

class CaseData(pz.Schema):
    """An individual row extracted from a table containing medical study data."""
    case_submitter_id = pz.Field(desc="The ID of the case", required=True)
    age_at_diagnosis = pz.Field(desc="The age of the patient at the time of diagnosis", required=False)
    race = pz.Field(desc="An arbitrary classification of a taxonomic group that is a division of a species.", required=False)
    ethnicity = pz.Field(desc="Whether an individual describes themselves as Hispanic or Latino or not.", required=False)
    gender = pz.Field(desc="Text designations that identify gender.", required=False)
    vital_status = pz.Field(desc="The vital status of the patient", required=False)
    primary_diagnosis = pz.Field(desc="Text term used to describe the patient's histologic diagnosis, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O).", required=False)

# Make sure to run
# pz reg --name biofabric-tiny --path testdata/biofabric-tiny
pz.DataDirectory().clearCache(keep_registry=True)

patient_tables = pz.Dataset('biofabric-mimic', schema=pz.Table)
patient_tables = patient_tables.filter("The table contains biometric information about the patient")
case_data = patient_tables.convert(CaseData, desc="The patient data in the table",cardinality=pz.Cardinality.ONE_TO_MANY)

output = case_data

policy = pz.MaxQuality()
engine = pz.StreamingSequentialExecution(
    allow_bonded_query=True,
    allow_code_synth=False,
    allow_token_reduction=False,
)

plan = engine.generate_plan(dataset=output, policy=policy)

start_time = time.time()
input_records = engine.get_input_records()
for idx, record in enumerate(input_records):
    output_records = engine.execute_opstream(plan, record)
    if idx == len(input_records) - 1:
        total_time = time.time() - start_time
        engine.plan_stats.finalize(total_time)
        finished = True
    stats = engine.plan_stats
    
    for table in output_records:
        header = table.header
        subset_rows = table.rows[:3]

        print("Table name:", table.name)
        # breakpoint()
        print(table.case_submitter_id, end=", ", flush=True)
        print(table.age_at_diagnosis, end=", ", flush=True)
        print(table.race, end=", ", flush=True)
        print(table.ethnicity, end=", ", flush=True)
        print(table.gender, end=", ", flush=True)
        print(table.vital_status, end=", ", flush=True)
        print(table.primary_diagnosis, flush=True)
        # input("Press Enter to continue...")

print(stats)