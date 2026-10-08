🏷️ Typing (:py:mod:`foamlib.typing`)
====================================

Standard types
--------------

The following are aliases of the primary types used throughout **foamlib** to represent the equivalent OpenFOAM data structures.

.. note:: For concrete classes in **foamlib** that represent files and some stored data types (like :class:`foamlib.Dimensioned`), see :ref:`additional-classes`.

.. autotype:: foamlib.typing.File
.. autotype:: foamlib.typing.SubDict
.. autotype:: foamlib.typing.Data
.. autotype:: foamlib.typing.StandaloneData
.. autotype:: foamlib.typing.DataEntry
.. autotype:: foamlib.typing.StandaloneDataEntry
.. autotype:: foamlib.typing.Dict
.. autotype:: foamlib.typing.KeywordEntry
.. autotype:: foamlib.typing.List
.. autotype:: foamlib.typing.Field
.. autotype:: foamlib.typing.Tensor


Other accepted types
--------------------

These "Like" type variants accept the standard type plus other formats that could potentially be converted to the standard type.

.. autotype:: foamlib.typing.FileLike
.. autotype:: foamlib.typing.SubDictLike
.. autotype:: foamlib.typing.DataLike
.. autotype:: foamlib.typing.StandaloneDataLike
.. autotype:: foamlib.typing.DataEntryLike
.. autotype:: foamlib.typing.StandaloneDataEntryLike
.. autotype:: foamlib.typing.DictLike
.. autotype:: foamlib.typing.KeywordEntryLike
.. autotype:: foamlib.typing.ListLike
.. autotype:: foamlib.typing.FieldLike
.. autotype:: foamlib.typing.TensorLike
.. autotype:: foamlib.typing.DimensionSetLike
