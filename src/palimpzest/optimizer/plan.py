from __future__ import annotations

from palimpzest.constants import MAX_ID_CHARS
from palimpzest.dataclasses import PlanCost
from palimpzest.operators import PhysicalOperator

# backwards compatability for users who are still on Python 3.9
try:
    from itertools import pairwise
except:
    from more_itertools import pairwise # type: ignore

from typing import List, Optional

import hashlib


class Plan:
    """A generic Plan is a graph of nodes (#TODO a list for now).
    The main subclasses are a LogicalPlan, which is composed of logical Operators, and a PhysicalPlan, which is composed of physical Operators.
    Plans are typically generated by objects of class Planner, and consumed by several objects, e.g., Execution, CostModel, Optimizer, etc. etc.
    """

    operators = []

    def __init__(self):
        raise NotImplementedError

    def __iter__(self):
        return iter(self.operators)

    def __next__(self):
        return next(iter(self.operators))

    def __len__(self):
        return len(self.operators)

    def __getitem__(self, idx: int):
        return self.operators[idx]

    def __str__(self):
        if self.operators:
            return f"{self.__class__.__name__}:\n" + "\n".join(
                map(str, [f"{idx}. {str(op)}" for idx, op in enumerate(self.operators)])
            )
        else:
            return f"{self.__class__.__name__}: No operator tree."


class PhysicalPlan(Plan):

    def __init__(self, operators: List[PhysicalOperator], plan_cost: Optional[PlanCost] = None):
        self.operators = operators
        self.plan_cost = plan_cost if plan_cost is not None else PlanCost(cost=0.0, time=0.0, quality=1.0)
        self.plan_id = self.compute_plan_id()

    def compute_plan_id(self) -> str:
        """
        NOTE: This is NOT a universal ID.
        
        Two different PhysicalPlan instances with the identical lists of operators will have equivalent plan_ids.
        """
        hash_str = str(tuple(op.get_op_id() for op in self.operators))
        return hashlib.sha256(hash_str.encode("utf-8")).hexdigest()[:MAX_ID_CHARS]

    def __eq__(self, other: PhysicalPlan):
        return self.operators == other.operators

    def __hash__(self):
        return int(self.plan_id, 16)

    @staticmethod
    def fromOpsAndSubPlan(ops: List[PhysicalOperator], ops_plan_cost: PlanCost, subPlan: PhysicalPlan) -> PhysicalPlan:
        # create copies of all logical operators
        copySubPlan = [op.copy() for op in subPlan.operators]
        copyOps = [op.copy() for op in ops]

        # construct full set of operators
        copySubPlan.extend(copyOps)

        # aggregate cost of ops and subplan
        full_plan_cost = subPlan.plan_cost + ops_plan_cost
        full_plan_cost.op_estimates = ops_plan_cost.op_estimates

        # return the PhysicalPlan
        return PhysicalPlan(operators=copySubPlan, plan_cost=full_plan_cost)

    def __repr__(self) -> str:
        """Computes a string representation for this plan."""
        label = "-".join([str(op) for op in self.operators])
        return f"PZ-{label}"

    def getPlanModelNames(self) -> List[str]:
        model_names = []
        for op in self.operators:
            model = getattr(op, "model", None)
            if model is not None:
                model_names.append(model.value)

        return model_names

    def printPlan(self) -> None:
        """Print the physical plan."""
        print_ops = self.operators
        start = print_ops[0]
        print(f" 0. {type(start).__name__} -> {start.outputSchema.__name__} \n")

        for idx, (left, right) in enumerate(pairwise(print_ops)):
            in_schema = left.outputSchema
            out_schema = right.outputSchema
            print(
                f" {idx+1}. {in_schema.__name__} -> {type(right).__name__} -> {out_schema.__name__} ",
                end="",
            )
            # if right.desc is not None:
            #     print(f" ({right.desc})", end="")
            # check if right has a model attribute
            if hasattr(right, "model"):
                print(f"\n    Using {right.model}", end="")
                if hasattr(right, "filter"):
                    filter_str = (
                        right.filter.filterCondition
                        if right.filter.filterCondition is not None
                        else str(right.filter.filterFn)
                    )
                    print(f'\n    Filter: "{filter_str}"', end="")
                if hasattr(right, "token_budget"):
                    print(f"\n    Token budget: {right.token_budget}", end="")
            print()
            print(
                f"    ({','.join(in_schema.fieldNames())[:15]}...) -> ({','.join(out_schema.fieldNames())[:15]}...)"
            )
            print()
