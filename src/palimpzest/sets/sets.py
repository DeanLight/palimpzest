from __future__ import annotations

from palimpzest.datamanager import DataDirectory
from palimpzest.corelib import Number, Schema
from palimpzest.elements import (
    AggregateFunction,
    Filter,
    UserFunction,
    GroupBySig,
)
from palimpzest.datasources import *

from typing import List, Optional, Union

import hashlib
import json


#####################################################
#
#####################################################
class Set:
    """
    A Set is the logical abstraction for a set of DataRecords matching some Schema. It is
    also a node in the computation graph of a Dataset.

    Each Dataset consists of one or more Sets. The "initial" Set in a Dataset can be thought
    of as the Set that results from reading each DataRecord unaltered from the source. For each
    filter or transformation that is applied to the Dataset, a new Set is created which defines
    the set of DataRecords that result from applying that filter or transformation. In brief,
    the Sets define a Dataset's computation graph. Sets can also be cached to maximize the reuse
    of past computation.

    Sets are initialized with a dataset_id, a schema, and a source. The source is either an
    existing Set or a raw data source (such as a directory or S3 prefix). Sets may be initialized
    with a Filter (which defines the filtering performed on the source to obtain *this* Set),
    and a description of what this Set is meant to represent.
    """

    SET_VERSION = 0.1

    def __init__(
        self,
        schema: Schema,
        source: Union[Set, DataSource],
        desc: str = None,
        filter: Filter = None,
        aggFunc: AggregateFunction = None,
        groupBy: GroupBySig = None,
        limit: int = None,
        fnid: str = None,
        cardinality: str = None,
        image_conversion: bool = None,
        depends_on: List[str] = None,
        nocache: bool = False,
    ):
        self.schema = schema
        self._source = source
        self._desc = desc
        self._filter = filter
        self._aggFunc = aggFunc
        self._groupBy = groupBy
        self._limit = limit
        self._fnid = fnid
        self._cardinality = cardinality
        self._image_conversion = image_conversion
        self._depends_on = depends_on
        self._nocache = nocache

    def __str__(self):
        return f"{self.__class__.__name__}(schema={self.schema}, desc={self._desc}, filter={str(self._filter)}, aggFunc={str(self._aggFunc)}, limit={str(self._limit)}, uid={self.universalIdentifier()})"

    def serialize(self):
        d = {
            "version": Set.SET_VERSION,
            "schema": self.schema.jsonSchema(),
            "source": self._source.serialize(),
            "desc": repr(self._desc),
            "filter": None if self._filter is None else self._filter.serialize(),
            "aggFunc": None if self._aggFunc is None else self._aggFunc.serialize(),
            "fnid": self._fnid,
            "cardinality": self._cardinality,
            "image_conversion": self._image_conversion,
            "depends_on": self._depends_on,
            "limit": self._limit,
            "groupBy": (
                None if self._groupBy is None else GroupBySig.serialize(self._groupBy)
            ),
        }

        return d

    def deserialize(inputObj):
        raise Exception("Cannot deserialize Set because it is the wrong version")

        # if inputObj["version"] != Set.SET_VERSION:
        #     raise Exception("Cannot deserialize Set because it is the wrong version")
        # # TODO: I don't believe this would actually work for the DataSources,
        # #       as inputObj["schema"] is a dict. not a Schema; also need to add
        # #       dataset_id to constructor here somehow
        # # deserialize source depending on whether it's a Set or DataSource
        # source = None
        # if "version" in inputObj["source"]:
        #     source = Set.deserialize(inputObj["source"])

        # elif inputObj["source_type"] == "directory":
        #     source = DirectoryTextFilesSource(inputObj["schema"])

        # elif inputObj["source_type"] == "file":
        #     source = FileSource(inputObj["schema"])

        # elif inputObj["source_type"] == "jsonstream":
        #     raise Exception("This can't possibly work, can it?")
        #     # source = JSONStreamSource(inputObj["schema"])

        # # deserialize agg. function
        # aggFuncStr = inputObj.get("aggFunc", None)
        # aggFunc = (
        #     None if aggFuncStr is None else AggregateFunction.deserialize(aggFuncStr)
        # )

        # # deserialize agg. function
        # groupByStr = inputObj.get("groupBy", None)
        # groupBy = None if groupByStr is None else GroupBySig.deserialize(groupByStr)

        # # deserialize limit
        # limitStr = inputObj.get("limit", None)
        # limit = None if limitStr is None else int(limitStr)

        # fnid = inputObj.get("fnid", None)
        # cardinality = inputObj.get("cardinality", None)
        # image_conversion = inputObj.get("image_conversion", None)
        # depends_on = inputObj.get("depends_on", None)

        # return Set(
        #     schema=inputObj["schema"].jsonSchema(),
        #     source=source,
        #     desc=inputObj["desc"],
        #     filter=Filter.deserialize(inputObj["filter"]),
        #     aggFunc=aggFunc,
        #     fnid=fnid,
        #     cardinality=cardinality,
        #     image_conversion=image_conversion,
        #     depends_on=depends_on,
        #     limit=limit,
        #     groupBy=groupBy,
        # )

    def universalIdentifier(self):
        """Return a unique identifier for this Set."""
        d = self.serialize()
        ordered = json.dumps(d, sort_keys=True)
        result = hashlib.sha256(ordered.encode()).hexdigest()
        return result

    def dataSourceId(self) -> Optional[str]:
        """
        Return the dataset_id of the DataSource if this Set's source is a DataSource.
        Otherwise return None.
        """
        return self._source.dataset_id if isinstance(self._source, DataSource) else None

    def jsonSchema(self):
        """Return the JSON schema for this Set."""
        return self.schema.jsonSchema()


class Dataset(Set):
    """
    A Dataset is the intended abstraction for programmers to interact with when manipulating Sets.

    Users instantiate a Dataset by specifying a `source` that either points to a
    DataSource or an existing cached Set. Users can then perform computations on
    the Dataset in an imperative fashion by leveraging functions such as `filter`,
    `convert`, `aggregate`, etc. Underneath the hood, each of these operations creates
    a new Set which is cached by the DataManager. As a result, the Sets define the
    lineage of computation on a Dataset, and this enables programmers to re-use
    previously cached computation by providing it as a `source` to some future Dataset.
    """

    def __init__(self, source: Union[str, DataSource], *args, **kwargs):
        # convert source (str) -> source (DataSource) if need be
        source = (
            DataDirectory().getRegisteredDataset(source)
            if isinstance(source, str)
            else source
        )

        # intialize class
        super().__init__(source=source, *args, **kwargs)

        if type(self._depends_on) == str:
            self._depends_on = [self._depends_on]

    def deserialize(inputObj):
        raise Exception("Cannot deserialize Set because it is the wrong version")
        # # TODO: this deserialize operation will not work; I need to finish the deserialize impl. for Schema
        # if inputObj["version"] != Set.SET_VERSION:
        #     raise Exception("Cannot deserialize Set because it is the wrong version")

        # # deserialize source depending on whether it's a Set or DataSource
        # source = None
        # if "version" in inputObj["source"]:
        #     source = Set.deserialize(inputObj["source"])

        # elif inputObj["source_type"] == "directory":
        #     source = DirectoryTextFilesSource(inputObj["schema"])

        # elif inputObj["source_type"] == "file":
        #     source = FileSource(inputObj["schema"])

        # return Dataset(source, inputObj["schema"], desc=inputObj["desc"])

    def filter(
        self,
        _filter: Union[str, callable],
        depends_on: Union[str, List[str]] = None,
        desc: str = "Apply filter(s)",
    ) -> Dataset:
        """Add a filter to the Set. This filter will possibly restrict the items that are returned later."""
        f = None
        if type(_filter) == str:
            f = Filter(_filter)
        elif callable(_filter):
            f = Filter(filterFn=_filter)
        else:
            raise Exception("Filter type not supported.", type(_filter))

        return Dataset(
            source=self,
            schema=self.schema,
            desc=desc,
            filter=f,
            depends_on=depends_on,
            nocache=self._nocache,
        )

    # TODO: here we should update the schema to be the incrementally-grown schema
    #       NOTE we would need to change this if/when we do logical re-ordering
    def convert(
        self,
        newSchema: Schema,
        cardinality: str = None,
        image_conversion: bool = False,
        depends_on: Union[str, List[str]] = None,
        desc: str = "Convert to new schema",
    ) -> Dataset:
        """Convert the Set to a new schema."""
        return Dataset(
            source=self,
            schema=newSchema,
            cardinality=cardinality,
            image_conversion=image_conversion,
            depends_on=depends_on,
            desc=desc,
            nocache=self._nocache,
        )

    def map(self, fn: UserFunction) -> Dataset:
        """Convert the Set to a new schema."""
        if not fn.inputSchema == self.schema:
            raise Exception(
                "Input schema of function ("
                + str(fn.inputSchema.getDesc())
                + ") does not match schema of input Set ("
                + str(self.schema.getDesc())
                + ")"
            )
        return Dataset(
            source=self, schema=fn.outputSchema, fnid=fn.udfid, nocache=self._nocache
        )

    def aggregate(self, aggFuncDesc: str) -> Dataset:
        """Apply an aggregate function to this set"""
        return Dataset(
            source=self,
            schema=Number,
            desc="Aggregate results",
            aggFunc=AggregateFunction(aggFuncDesc),
            nocache=self._nocache,
        )

    def groupby(self, groupBy: GroupBySig) -> Dataset:
        return Dataset(
            source=self,
            schema=groupBy.outputSchema(),
            desc="Group By",
            groupBy=groupBy,
            nocache=self._nocache,
        )

    def limit(self, n: int) -> Dataset:
        """Limit the set size to no more than n rows"""
        return Dataset(
            source=self,
            schema=self.schema,
            desc="LIMIT " + str(n),
            limit=n,
            nocache=self._nocache,
        )
