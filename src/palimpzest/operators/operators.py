from __future__ import annotations

from palimpzest.constants import Model, QueryStrategy
from palimpzest.datamanager import DataDirectory
from palimpzest.elements import *
from palimpzest.operators import (
    ApplyCountAggregateOp,
    ApplyAverageAggregateOp,
    ApplyUserFunctionOp,
    CacheScanDataOp,
    FilterCandidateOp,
    InduceFromCandidateOp,
    LimitScanOp,
    MarshalAndScanDataOp,
    ParallelFilterCandidateOp,
    ParallelInduceFromCandidateOp,
    PhysicalOp,
    ApplyGroupByOp
)
from palimpzest.profiler import StatsProcessor

from copy import deepcopy
from itertools import permutations
from typing import List, Tuple

import pandas as pd

import os
import random

# DEFINITIONS
PhysicalPlan = Tuple[float, float, float, PhysicalOp]


class LogicalOperator:
    """
    A logical operator is an operator that operates on Sets. Right now it can be one of:
    - BaseScan (scans data from DataSource)
    - CacheScan (scans cached Set)
    - FilteredScan (scans input Set and applies filter)
    - ConvertScan (scans input Set and converts it to new Schema)
    - LimitScan (scans up to N records from a Set)
    - ApplyAggregateFunction (applies an aggregation on the Set)
    """
    def __init__(self, outputSchema: Schema, inputSchema: Schema, inputOp: LogicalOperator=None):
        self.outputSchema = outputSchema
        self.inputSchema = inputSchema
        self.inputOp = inputOp

    def dumpLogicalTree(self) -> Tuple[LogicalOperator, LogicalOperator]:
        raise NotImplementedError("Abstract method")

    def _getPhysicalTree(self, strategy: str=None, source: PhysicalOp=None, shouldProfile: bool=False) -> PhysicalOp:
        raise NotImplementedError("Abstract method")

    def _getModels(self, include_vision: bool=False):
        models = []
        if os.getenv('OPENAI_API_KEY') is not None:
            models.extend([Model.GPT_3_5, Model.GPT_4])
            # models.extend([Model.GPT_4])
            # models.extend([Model.GPT_3_5])

        if os.getenv('TOGETHER_API_KEY') is not None:
            models.extend([Model.MIXTRAL])

        if os.getenv('GOOGLE_API_KEY') is not None:
            models.extend([Model.GEMINI_1])

        if include_vision:
            models.append(Model.GPT_4V)

        return models

    @staticmethod
    def _compute_legal_permutations(filterAndConvertOps: List[LogicalOperator]) -> List[List[LogicalOperator]]:
        # There are a few rules surrounding which permutation(s) of logical operators are legal:
        # 1. if a filter depends on a field in a convert's outputSchema, it must be executed after the convert
        # 2. if a convert depends on another operation's outputSchema, it must be executed after that operation
        # 3. if depends_on is not specified for a convert operator, it cannot be swapped with another convert
        # 4. if depends_on is not specified for a filter, it can not be swapped with a convert (but it can be swapped w/adjacent filters)

        # compute implicit depends_on relationships, keep in mind that operations closer to the end of the list are executed first;
        # if depends_on is not specified for a convert or filter, it implicitly depends_on all preceding converts
        for idx, op in enumerate(filterAndConvertOps):
            if op.depends_on is None:
                all_prior_generated_fields = []
                for upstreamOp in filterAndConvertOps[idx+1:]:
                    if isinstance(upstreamOp, ConvertScan):
                        all_prior_generated_fields.extend(upstreamOp.generated_fields)
                op.depends_on = all_prior_generated_fields

        # compute all permutations of operators
        opPermutations = permutations(filterAndConvertOps)

        # iterate over permutations and determine if they are legal;
        # keep in mind that operations closer to the end of the list are executed first
        legalOpPermutations = []
        for opPermutation in opPermutations:
            is_valid = True
            for idx, op in enumerate(opPermutation):
                # if this op is a filter, we can skip because no upstream ops will conflict with this
                if isinstance(op, FilteredScan):
                    continue

                # invalid if upstream op depends on field generated by this op
                for upstreamOp in opPermutation[idx+1:]:
                    for col in upstreamOp.depends_on:
                        if col in op.generated_fields:
                            is_valid = False
                            break
                    if is_valid is False:
                        break
                if is_valid is False:
                    break
            
            # if permutation order is valid, then:
            # 1. make unique copy of each logical op
            # 2. update inputOp's
            # 3. update inputSchema's
            if is_valid:
                opCopyPermutation = [deepcopy(op) for op in opPermutation]
                for idx, op in enumerate(opCopyPermutation):
                    op.inputOp = opCopyPermutation[idx + 1] if idx + 1 < len(opCopyPermutation) else None

                # set schemas in reverse order
                for idx in range(len(opCopyPermutation) -1, 0, -1):
                    op = opCopyPermutation[idx-1]
                    op.inputSchema = opCopyPermutation[idx].outputSchema

                    if isinstance(op, FilteredScan):
                        op.outputSchema = op.inputSchema

                legalOpPermutations.append(opCopyPermutation)

        return legalOpPermutations

    # TODO: debug if deepcopy is not making valid copies to resolve duplicate profiler issue
    @staticmethod
    def _createLogicalPlans(rootOp: LogicalOperator) -> List[LogicalOperator]:
        """
        Given the logicalOperator, compute all possible equivalent plans with filter
        and convert operations re-ordered.
        """
        # base case, if this operator is a BaseScan or CacheScan, return operator
        if isinstance(rootOp, BaseScan) or isinstance(rootOp, CacheScan):
            return [rootOp]

        # if this operator is not a FilteredScan: compute the re-orderings for its inputOp,
        # point rootOp to each of the re-orderings, and return
        if not isinstance(rootOp, FilteredScan) and not isinstance(rootOp, ConvertScan):
            subTrees = LogicalOperator._createLogicalPlans(rootOp.inputOp)

            all_plans = []
            for tree in subTrees:
                rootOpCopy = deepcopy(rootOp)
                rootOpCopy.inputOp = tree
                all_plans.append(rootOpCopy)

            return all_plans

        # otherwise, if this operator is a FilteredScan or ConvertScan, make one plan per (legal)
        # permutation of consecutive converts and filters and recurse
        else:
            # get list of consecutive converts and filters
            filterAndConvertOps = []
            nextOp = rootOp
            while isinstance(nextOp, FilteredScan) or isinstance(nextOp, ConvertScan):
                filterAndConvertOps.append(nextOp)
                nextOp = nextOp.inputOp

            # compute set of legal permutations
            opPermutations = LogicalOperator._compute_legal_permutations(filterAndConvertOps)

            # compute filter reorderings for rest of tree
            subTrees = LogicalOperator._createLogicalPlans(nextOp)

            # compute cross-product of opPermutations and subTrees by linking final op w/first op in subTree
            for ops in opPermutations:
                for tree in subTrees:
                    ops[-1].inputOp = tree
                    ops[-1].inputSchema = tree.outputSchema

            # return roots of opPermutations
            return list(map(lambda ops: ops[0], opPermutations))


    def _createPhysicalPlans(self, allow_model_selection: bool=False, allow_codegen: bool=False, allow_token_reduction: bool=False, shouldProfile: bool=False) -> List[PhysicalOp]:
        """
        Given the logical plan implied by this LogicalOperator, enumerate up to `max`
        possible physical plans and return them as a list.
        """
        # TODO: for each FilteredScan & ConvertScan try:
        # 1. swapping different models
        #    a. different model hyperparams?
        # 2. different prompt strategies
        #    a. Zero-Shot vs. Few-Shot vs. COT vs. DSPy
        # 3. input sub-selection
        #    a. vector DB, LLM attention, ask-the-LLM

        # choose set of acceptable models based on possible llmservices
        models = self._getModels()
        assert len(models) > 0, "No models available to create physical plans! You must set at least one of the following environment variables: [OPENAI_API_KEY, TOGETHER_API_KEY, GOOGLE_API_KEY]"

        # determine which query strategies may be used
        query_strategies = [QueryStrategy.BONDED_WITH_FALLBACK]
        if allow_codegen:
            query_strategies.append(QueryStrategy.CODE_GEN_WITH_FALLBACK)

        token_budgets = [1.0]
        if allow_token_reduction:
            token_budgets.extend([0.1, 0.5, 0.9])

        # base case: this is a root op
        if self.inputOp is None:
            # NOTE: right now, the root op must be a CacheScan or BaseScan which does not require an LLM;
            #       if this ever changes we may need to return a list of physical ops here
            return [self._getPhysicalTree(strategy=PhysicalOp.LOCAL_PLAN, shouldProfile=shouldProfile)]

        # recursive case: get list of possible input physical plans
        subTreePhysicalPlans = self.inputOp._createPhysicalPlans(
            allow_model_selection=allow_model_selection,
            allow_codegen=allow_codegen,
            allow_token_reduction=allow_token_reduction,
            shouldProfile=shouldProfile,
        )

        def get_models_from_subtree(phys_op):
            phys_op_models = []
            while phys_op is not None:
                phys_op_models.append(getattr(phys_op, 'model', None))
                phys_op = phys_op.source

            return phys_op_models

        # compute (list of) physical plans for this op
        physicalPlans = []
        if isinstance(self, ConvertScan):
            for subTreePhysicalPlan in subTreePhysicalPlans:
                for qs in query_strategies:
                    for token_budget in token_budgets:
                        for model in models:
                            # if model selection is disallowed; skip any plans which would use a different model in this operator
                            subtree_models = [m for m in get_models_from_subtree(subTreePhysicalPlan) if m is not None and m != Model.GPT_4V]
                            if not allow_model_selection and len(subtree_models) > 0 and subtree_models[0] != model:
                                continue

                            # NOTE: failing to make a copy will lead to duplicate profile information being captured
                            # create a copy of subTreePhysicalPlan and use it as source for this physicalPlan
                            subTreePhysicalPlan = subTreePhysicalPlan.copy()
                            physicalPlan = self._getPhysicalTree(strategy=PhysicalOp.LOCAL_PLAN, source=subTreePhysicalPlan, model=model, query_strategy=qs, token_budget=token_budget, shouldProfile=shouldProfile)
                            physicalPlans.append(physicalPlan)
                            # GV Checking if there is an hardcoded function exposes that we need to refactor the solver/physical function generation
                            td = physicalPlan._makeTaskDescriptor()
                            if td.model == None:
                                break

        elif isinstance(self, FilteredScan):
            for subTreePhysicalPlan in subTreePhysicalPlans:
                for model in models:
                    # if model selection is disallowed; skip any plans which would use a different model in this operator
                    subtree_models = [m for m in get_models_from_subtree(subTreePhysicalPlan) if m is not None and m != Model.GPT_4V]
                    if not allow_model_selection and len(subtree_models) > 0 and subtree_models[0] != model:
                        continue

                    # NOTE: failing to make a copy will lead to duplicate profile information being captured
                    # create a copy of subTreePhysicalPlan and use it as source for this physicalPlan
                    subTreePhysicalPlan = subTreePhysicalPlan.copy()
                    physicalPlan = self._getPhysicalTree(strategy=PhysicalOp.LOCAL_PLAN, source=subTreePhysicalPlan, model=model, shouldProfile=shouldProfile)
                    physicalPlans.append(physicalPlan)
                    # GV Checking if there is an hardcoded function exposes that we need to refactor the solver/physical function generation
                    td = physicalPlan._makeTaskDescriptor()
                    if td.model == None:
                        break

        else:
            for subTreePhysicalPlan in subTreePhysicalPlans:
                # NOTE: failing to make a copy will lead to duplicate profile information being captured
                # create a copy of subTreePhysicalPlan and use it as source for this physicalPlan
                subTreePhysicalPlan = subTreePhysicalPlan.copy()
                physicalPlan = self._getPhysicalTree(strategy=PhysicalOp.LOCAL_PLAN, source=subTreePhysicalPlan, shouldProfile=shouldProfile)
                physicalPlans.append(physicalPlan)

        return physicalPlans

    def createPhysicalPlanCandidates(self, max: int=None, cost_estimate_sample_data: List[Dict[str, Any]]=None, allow_model_selection: bool=False, allow_codegen: bool=False, allow_token_reduction: bool=False, pareto_optimal: bool=True, shouldProfile: bool=False) -> List[PhysicalPlan]:
        """Return a set of physical trees of operators."""
        # create set of logical plans (e.g. consider different filter/join orderings)
        logicalPlans = LogicalOperator._createLogicalPlans(self)
        print(f"LOGICAL PLANS: {len(logicalPlans)}")

        # iterate through logical plans and evaluate multiple physical plans
        physicalPlans = [
            physicalPlan
            for logicalPlan in logicalPlans
            for physicalPlan in logicalPlan._createPhysicalPlans(
                allow_model_selection=allow_model_selection,
                allow_codegen=allow_codegen,
                allow_token_reduction=allow_token_reduction,
                shouldProfile=shouldProfile,
            )
        ]
        print(f"INITIAL PLANS: {len(physicalPlans)}")

        # compute estimates for every operator
        op_filters_to_estimates = {}
        if cost_estimate_sample_data is not None:
            # construct full dataset of samples
            df = pd.DataFrame(cost_estimate_sample_data)

            # get unique set of operator filters:
            # - for base/cache scans this is very simple
            # - for filters, this is based on the unique filter string or function (per-model)
            # - for induce, this is based on the generated field(s) (per-model)
            op_filters_to_estimates = {}
            logical_op = logicalPlans[0]
            while logical_op is not None:
                op_filter, estimates = None, None
                if isinstance(logical_op, BaseScan):
                    op_filter = "op_name == 'base_scan'"
                    op_df = df.query(op_filter)
                    if not op_df.empty:
                        estimates = {
                            "time_per_record": StatsProcessor._est_time_per_record(op_df)
                        }
                    op_filters_to_estimates[op_filter] = estimates

                elif isinstance(logical_op, CacheScan):
                    op_filter = "op_name == 'cache_scan'"
                    op_df = df.query(op_filter)
                    if not op_df.empty:
                        estimates = {
                            "time_per_record": StatsProcessor._est_time_per_record(op_df)
                        }
                    op_filters_to_estimates[op_filter] = estimates

                elif isinstance(logical_op, ConvertScan):
                    generated_fields_str = "-".join(sorted(logical_op.generated_fields))
                    op_filter = f"(generated_fields == '{generated_fields_str}') & (op_name == 'induce' | op_name == 'p_induce')"
                    op_df = df.query(op_filter)
                    if not op_df.empty:
                        models = self._getModels(include_vision=True)
                        estimates = {model: None for model in models}
                        for model in models:
                            est_tokens = StatsProcessor._est_num_input_output_tokens(op_df, model_name=model.value)
                            model_estimates = {
                                "time_per_record": StatsProcessor._est_time_per_record(op_df, model_name=model.value),
                                "cost_per_record": StatsProcessor._est_usd_per_record(op_df, model_name=model.value),
                                "est_num_input_tokens": est_tokens[0],
                                "est_num_output_tokens": est_tokens[1],
                                "selectivity": StatsProcessor._est_selectivity(df, op_df, model_name=model.value),
                                "quality": StatsProcessor._est_quality(op_df, model_name=model.value),
                            }
                            estimates[model.value] = model_estimates
                    op_filters_to_estimates[op_filter] = estimates

                elif isinstance(logical_op, FilteredScan):
                    import pdb
                    pdb.set_trace()
                    filter_str = self.filter.filterCondition if self.filter.filterCondition is not None else str(self.filter.filterFn)
                    op_filter = f"(filter == '{str(filter_str)}') & (op_name == 'filter' | op_name == 'p_filter')"
                    op_df = df.query(op_filter)
                    if not op_df.empty:
                        models = (
                            self._getModels()
                            if self.filter.filterCondition is not None
                            else [None]
                        )
                        estimates = {model: None for model in models}
                        for model in models:
                            model_name = model.value if model is not None else None
                            est_tokens = StatsProcessor._est_num_input_output_tokens(op_df, model_name=model_name)
                            model_estimates = {
                                "time_per_record": StatsProcessor._est_time_per_record(op_df, model_name=model_name),
                                "cost_per_record": StatsProcessor._est_usd_per_record(op_df, model_name=model_name),
                                "est_num_input_tokens": est_tokens[0],
                                "est_num_output_tokens": est_tokens[1],
                                "selectivity": StatsProcessor._est_selectivity(df, op_df, model_name=model_name),
                                "quality": StatsProcessor._est_quality(op_df, model_name=model_name),
                            }
                            estimates[model_name] = model_estimates
                    op_filters_to_estimates[op_filter] = estimates
                    pdb.set_trace()

                elif isinstance(logical_op, LimitScan):
                    op_filter = "(op_name == 'limit')"
                    op_df = df.query(op_filter)
                    if not op_df.empty:
                        estimates = {
                            "time_per_record": StatsProcessor._est_time_per_record(op_df)
                        }
                    op_filters_to_estimates[op_filter] = estimates

                elif isinstance(logical_op, ApplyAggregateFunction) and logical_op.aggregationFunction.funcDesc == "COUNT":
                    op_filter = "(op_name == 'count')"
                    op_df = df.query(op_filter)
                    if not op_df.empty:
                        estimates = {
                            "time_per_record": StatsProcessor._est_time_per_record(op_df)
                        }
                    op_filters_to_estimates[op_filter] = estimates

                elif isinstance(logical_op, ApplyAggregateFunction) and logical_op.aggregationFunction.funcDesc == "AVERAGE":
                    op_filter = "(op_name == 'average')"
                    op_df = df.query(op_filter)
                    if not op_df.empty:
                        estimates = {
                            "time_per_record": StatsProcessor._est_time_per_record(op_df)
                        }
                    op_filters_to_estimates[op_filter] = estimates

                logical_op = logical_op.inputOp

        # estimate the cost (in terms of USD, latency, throughput, etc.) for each plan
        plans = []
        cost_est_data = None if op_filters_to_estimates == {} else op_filters_to_estimates
        for physicalPlan in physicalPlans:
            planCost, fullPlanCostEst = physicalPlan.estimateCost(cost_est_data=cost_est_data)

            totalTime = planCost["totalTime"]
            totalCost = planCost["totalUSD"]  # for now, cost == USD
            quality = planCost["quality"]

            plans.append((totalTime, totalCost, quality, physicalPlan, fullPlanCostEst))

        # drop duplicate plans in terms of time, cost, and quality, as these can cause
        # plans on the pareto frontier to be dropped if they are "dominated" by a duplicate
        dedup_plans, dedup_desc_set = [], set()
        for plan in plans:
            planDesc = (plan[0], plan[1], plan[2])
            if planDesc not in dedup_desc_set:
                dedup_desc_set.add(planDesc)
                dedup_plans.append(plan)
        
        print(f"DEDUP PLANS: {len(dedup_plans)}")

        # return de-duplicated set of plans if we don't want to compute the pareto frontier
        if not pareto_optimal:
            if max is not None:
                dedup_plans = dedup_plans[:max]
                print(f"LIMIT DEDUP PLANS: {len(dedup_plans)}")

            return dedup_plans

        # compute the pareto frontier of candidate physical plans and return the list of such plans
        # - brute force: O(d*n^2);
        #   - for every tuple, check if it is dominated by any other tuple;
        #   - if it is, throw it out; otherwise, add it to pareto frontier
        #
        # more efficient algo.'s exist, but they are non-trivial to implement, so for now I'm using
        # brute force; it may ultimately be best to compute a cheap approx. of the pareto front:
        # - e.g.: https://link.springer.com/chapter/10.1007/978-3-642-12002-2_6
        paretoFrontierPlans = []
        for i, (totalTime_i, totalCost_i, quality_i, plan, fullPlanCostEst) in enumerate(dedup_plans):
            paretoFrontier = True

            # check if any other plan dominates plan i
            for j, (totalTime_j, totalCost_j, quality_j, _, _) in enumerate(dedup_plans):
                if i == j:
                    continue

                # if plan i is dominated by plan j, set paretoFrontier = False and break
                if totalTime_j <= totalTime_i and totalCost_j <= totalCost_i and quality_j >= quality_i:
                    paretoFrontier = False
                    break

            # add plan i to pareto frontier if it's not dominated
            if paretoFrontier:
                paretoFrontierPlans.append((totalTime_i, totalCost_i, quality_i, plan, fullPlanCostEst))

        print(f"PARETO PLANS: {len(paretoFrontierPlans)}")
        if max is not None:
            paretoFrontierPlans = paretoFrontierPlans[:max]
            print(f"LIMIT PARETO PLANS: {len(paretoFrontierPlans)}")

        return paretoFrontierPlans


class ConvertScan(LogicalOperator):
    """A ConvertScan is a logical operator that represents a scan of a particular data source, with conversion applied."""
    def __init__(self, outputSchema: Schema, inputOp: LogicalOperator, cardinality: str=None, image_conversion: bool=False, depends_on: List[str]=None, desc: str=None, targetCacheId: str=None):
        super().__init__(outputSchema, inputOp.outputSchema)
        self.inputOp = inputOp
        self.cardinality = cardinality
        self.image_conversion = image_conversion
        self.depends_on = depends_on
        self.desc = desc
        self.targetCacheId = targetCacheId

        # compute generated fields as set of fields in outputSchema that are not in inputSchema
        self.generated_fields = [
            field
            for field in self.outputSchema.fieldNames()
            if field not in self.inputSchema.fieldNames()
        ]

    def __str__(self):
        return "ConvertScan(" + str(self.inputSchema) + ", " + str(self.outputSchema) + ", " + str(self.desc) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, self.inputOp.dumpLogicalTree())

    def _getPhysicalTree(self, strategy: str=None, source: PhysicalOp=None, model: Model=None, query_strategy: QueryStrategy=None, token_budget: float=None, shouldProfile: bool=False):
        # TODO: dont set input op here
        # If the input is in core, and the output is NOT in core but its superclass is, then we should do a
        # 2-stage conversion. This will maximize chances that there is a pre-existing conversion to the superclass
        # in the known set of functions
        intermediateSchema = self.outputSchema
        while not intermediateSchema == Schema and not PhysicalOp.solver.easyConversionAvailable(intermediateSchema, self.inputSchema):
            intermediateSchema = intermediateSchema.__bases__[0]

        if intermediateSchema == Schema or intermediateSchema == self.outputSchema:
            if DataDirectory().current_config.get("parallel") == True:
                return ParallelInduceFromCandidateOp(self.outputSchema,
                                                     source,
                                                     model,
                                                     self.cardinality,
                                                     self.image_conversion,
                                                     query_strategy=query_strategy,
                                                     token_budget=token_budget,
                                                     desc=self.desc,
                                                     targetCacheId=self.targetCacheId,
                                                     shouldProfile=shouldProfile)
            else:
                return InduceFromCandidateOp(self.outputSchema,
                                             source,
                                             model,
                                             self.cardinality,
                                             self.image_conversion,
                                             query_strategy=query_strategy,
                                             token_budget=token_budget,
                                             desc=self.desc,
                                             targetCacheId=self.targetCacheId,
                                             shouldProfile=shouldProfile)
        else:
            if DataDirectory().current_config.get("parallel") == True:
                return ParallelInduceFromCandidateOp(self.outputSchema,
                                                     ParallelInduceFromCandidateOp(
                                                         intermediateSchema,
                                                         source,
                                                         model,
                                                         self.cardinality,
                                                         self.image_conversion,          # TODO: only one of these should have image_conversion
                                                         query_strategy=query_strategy,
                                                         token_budget=token_budget,
                                                         shouldProfile=shouldProfile),
                                                     model,
                                                     "oneToOne",
                                                     image_conversion=self.image_conversion,  # TODO: only one of these should have image_conversion
                                                     query_strategy=query_strategy,
                                                     token_budget=token_budget,
                                                     desc=self.desc,
                                                     targetCacheId=self.targetCacheId,
                                                     shouldProfile=shouldProfile)
            else:
                return InduceFromCandidateOp(self.outputSchema,
                                             InduceFromCandidateOp(
                                                 intermediateSchema,
                                                 source,
                                                 model,
                                                 self.cardinality,
                                                 self.image_conversion,          # TODO: only one of these should have image_conversion
                                                 query_strategy=query_strategy,
                                                 token_budget=token_budget,
                                                 shouldProfile=shouldProfile),
                                             model,
                                             "oneToOne",
                                             image_conversion=self.image_conversion,  # TODO: only one of these should have image_conversion
                                             query_strategy=query_strategy,
                                             token_budget=token_budget,
                                             desc=self.desc,
                                             targetCacheId=self.targetCacheId,
                                             shouldProfile=shouldProfile)


class CacheScan(LogicalOperator):
    """A CacheScan is a logical operator that represents a scan of a cached Set."""
    def __init__(self, outputSchema: Schema, cachedDataIdentifier: str, num_samples: int=None, scan_start_idx: int=0):
        super().__init__(outputSchema, None)
        self.cachedDataIdentifier = cachedDataIdentifier
        self.num_samples = num_samples
        self.scan_start_idx = scan_start_idx

    def __str__(self):
        return "CacheScan(" + str(self.outputSchema) + ", " + str(self.cachedDataIdentifier) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, None)

    def _getPhysicalTree(self, strategy: str=None, source: PhysicalOp=None, shouldProfile: bool=False):
        return CacheScanDataOp(self.outputSchema, self.cachedDataIdentifier, num_samples=self.num_samples, scan_start_idx=self.scan_start_idx, shouldProfile=shouldProfile)


class BaseScan(LogicalOperator):
    """A BaseScan is a logical operator that represents a scan of a particular data source."""
    def __init__(self, outputSchema: Schema, datasetIdentifier: str, num_samples: int=None, scan_start_idx: int=0):
        super().__init__(outputSchema, None)
        self.datasetIdentifier = datasetIdentifier
        self.num_samples = num_samples
        self.scan_start_idx = scan_start_idx

    def __str__(self):
        return "BaseScan(" + str(self.outputSchema) + ", " + self.datasetIdentifier + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, None)

    def _getPhysicalTree(self, strategy: str=None, source: PhysicalOp=None, shouldProfile: bool=False):
        return MarshalAndScanDataOp(self.outputSchema, self.datasetIdentifier, num_samples=self.num_samples, scan_start_idx=self.scan_start_idx, shouldProfile=shouldProfile)


class LimitScan(LogicalOperator):
    def __init__(self, outputSchema: Schema, inputOp: LogicalOperator, limit: int, targetCacheId: str=None):
        super().__init__(outputSchema, inputOp.outputSchema)
        self.inputOp = inputOp
        self.targetCacheId = targetCacheId
        self.limit = limit

    def __str__(self):
        return "LimitScan(" + str(self.inputSchema) + ", " + str(self.outputSchema) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, self.inputOp.dumpLogicalTree())

    def _getPhysicalTree(self, strategy: str=None, source: PhysicalOp=None, shouldProfile: bool=False):
        return LimitScanOp(self.outputSchema, source, self.limit, targetCacheId=self.targetCacheId, shouldProfile=shouldProfile)


class FilteredScan(LogicalOperator):
    """A FilteredScan is a logical operator that represents a scan of a particular data source, with filters applied."""
    def __init__(self, outputSchema: Schema, inputOp: LogicalOperator, filter: Filter, depends_on: List[str]=None, targetCacheId: str=None):
        super().__init__(outputSchema, inputOp.outputSchema)
        self.inputOp = inputOp
        self.filter = filter
        self.depends_on = depends_on
        self.targetCacheId = targetCacheId

    def __str__(self):
        return "FilteredScan(" + str(self.outputSchema) + ", " + "Filters: " + str(self.filter) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, self.inputOp.dumpLogicalTree())

    def _getPhysicalTree(self, strategy: str=None, source: PhysicalOp=None, model: Model=None, shouldProfile: bool=False):
        if DataDirectory().current_config.get("parallel") == True:
            return ParallelFilterCandidateOp(self.outputSchema, source, self.filter, model=model, targetCacheId=self.targetCacheId, shouldProfile=shouldProfile)
        else:
            return FilterCandidateOp(self.outputSchema, source, self.filter, model=model, targetCacheId=self.targetCacheId, shouldProfile=shouldProfile)


class GroupByAggregate(LogicalOperator):
    def __init__(self, outputSchema: Schema, inputOp: LogicalOperator, gbySig: elements.GroupBySig, targetCacheId: str=None):
        super().__init__(outputSchema, inputOp.outputSchema)
        (valid, error) = gbySig.validateSchema(inputOp.outputSchema)
        if (not valid):
            raise TypeError(error)
        self.inputOp = inputOp 
        self.gbySig = gbySig
        self.targetCacheId = targetCacheId
    def __str__(self):
        descStr = "Grouping Fields:" 
        return (f"GroupBy({elements.GroupBySig.serialize(self.gbySig)})")

    def dumpLogicalTree(self):
        """Return the logical subtree rooted at this operator"""
        return (self, self.inputOp.dumpLogicalTree())

    def _getPhysicalTree(self, strategy: str=None, source: PhysicalOp=None, model: Model=None, shouldProfile: bool=False):
        return ApplyGroupByOp(source, self.gbySig, targetCacheId=self.targetCacheId, shouldProfile=shouldProfile)


class ApplyAggregateFunction(LogicalOperator):
    """ApplyAggregateFunction is a logical operator that applies a function to the input set and yields a single result."""
    def __init__(self, outputSchema: Schema, inputOp: LogicalOperator, aggregationFunction: AggregateFunction, targetCacheId: str=None):
        super().__init__(outputSchema, inputOp.outputSchema)
        self.inputOp = inputOp
        self.aggregationFunction = aggregationFunction
        self.targetCacheId=targetCacheId

    def __str__(self):
        return "ApplyAggregateFunction(function: " + str(self.aggregationFunction) + ")"

    def dumpLogicalTree(self):
        """Return the logical subtree rooted at this operator"""
        return (self, self.inputOp.dumpLogicalTree())

    def _getPhysicalTree(self, strategy: str=None, source: PhysicalOp=None, model: Model=None, shouldProfile: bool=False):
        if self.aggregationFunction.funcDesc == "COUNT":
            return ApplyCountAggregateOp(source, self.aggregationFunction, targetCacheId=self.targetCacheId, shouldProfile=shouldProfile)
        elif self.aggregationFunction.funcDesc == "AVERAGE":
            return ApplyAverageAggregateOp(source, self.aggregationFunction, targetCacheId=self.targetCacheId, shouldProfile=shouldProfile)
        else:
            raise Exception(f"Cannot find implementation for {self.aggregationFunction}")


class ApplyUserFunction(LogicalOperator):
    """ApplyUserFunction is a logical operator that applies a user-provided function to the input set and yields a result."""
    def __init__(self, outputSchema: Schema, inputOp: LogicalOperator, fnid:str, targetCacheId: str=None):
        super().__init__(outputSchema, inputOp.outputSchema)
        self.inputOp = inputOp
        self.fnid = fnid
        self.fn = DataDirectory().getUserFunction(fnid)
        self.targetCacheId=targetCacheId

    def __str__(self):
        return "ApplyUserFunction(function: " + str(self.fnid) + ")"

    def dumpLogicalTree(self):
        """Return the logical subtree rooted at this operator"""
        return (self, self.inputOp.dumpLogicalTree())

    def _getPhysicalTree(self, strategy: str=None, source: PhysicalOp=None, model: Model=None, shouldProfile: bool=False):
        return ApplyUserFunctionOp(source, self.fn, targetCacheId=self.targetCacheId, shouldProfile=shouldProfile)
