"""Type aliases for OpenFOAM data structures."""

from ._files import typing as _typing

#: A single OpenFOAM value, or multiple values as a tuple.
Data = _typing.Data
#: A single OpenFOAM value.
DataEntry = _typing.DataEntry
#: Any type that could be interpreted as a :type:`DataEntry`.
DataEntryLike = _typing.DataEntryLike
#: Any type that could be interpreted as a :type:`Data`.
DataLike = _typing.DataLike
#: An OpenFOAM dictionary.
Dict = _typing.Dict
#: Any mapping that could be interpreted as a :type:`Dict`.
DictLike = _typing.DictLike
#: Any type that could be interpreted as a :class:`foamlib.DimensionSet`.
DimensionSetLike = _typing.DimensionSetLike
#: An OpenFOAM field of scalars, vectors, symmetric tensors, or full tensors.
Field = _typing.Field
#: Any type that could be interpreted as a :type:`Field`.
FieldLike = _typing.FieldLike
#: An entire OpenFOAM file as a :class:`dict` or :class:`MultiDict`.
FileDict = _typing.FileDict
#: Any mapping that could be interpreted as a :type:`FileDict`.
FileDictLike = _typing.FileDictLike
#: An OpenFOAM keyword entry (i.e., a key-value pair).
KeywordEntry = _typing.KeywordEntry
#: Any 2-tuple that could be interpreted as a :type:`KeywordEntry`.
KeywordEntryLike = _typing.KeywordEntryLike
#: An OpenFOAM list.
List = _typing.List
#: Any sequence that could be interpreted as a :type:`List`.
ListLike = _typing.ListLike
#: One or more OpenFOAM values that can appear at the top level of a file.
StandaloneData = _typing.StandaloneData
#: A single OpenFOAM value that can appear at the top level of a file.
StandaloneDataEntry = _typing.StandaloneDataEntry
#: Any type that could be interpreted as a :type:`StandaloneDataEntry`.
StandaloneDataEntryLike = _typing.StandaloneDataEntryLike
#: Any type that could be interpreted as a :type:`StandaloneData`.
StandaloneDataLike = _typing.StandaloneDataLike
#: An OpenFOAM dictionary nested in a file.
SubDict = _typing.SubDict
#: Any mapping that could be interpreted as a :type:`SubDict`.
SubDictLike = _typing.SubDictLike
#: An OpenFOAM scalar, vector, symmetric tensor, or full tensor.
Tensor = _typing.Tensor
#: Any type that could be interpreted as a :type:`Tensor`.
TensorLike = _typing.TensorLike

__all__ = [
    "Data",
    "DataEntry",
    "DataEntryLike",
    "DataLike",
    "Dict",
    "DictLike",
    "DimensionSetLike",
    "Field",
    "FieldLike",
    "FileDict",
    "FileDictLike",
    "KeywordEntry",
    "KeywordEntryLike",
    "List",
    "ListLike",
    "StandaloneData",
    "StandaloneDataEntry",
    "StandaloneDataEntryLike",
    "StandaloneDataLike",
    "SubDict",
    "SubDictLike",
    "Tensor",
    "TensorLike",
]
