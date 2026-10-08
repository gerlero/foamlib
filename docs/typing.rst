🏷️ Typing (:py:mod:`foamlib.typing`)
====================================

Standard types
--------------

The following are aliases of the primary types used throughout **foamlib** to represent the equivalent OpenFOAM data structures.

.. note:: For concrete classes in **foamlib** that represent files and some stored data types (like :class:`foamlib.Dimensioned`), see :ref:`additional-classes`.

.. autodata:: foamlib.typing.Tensor
.. autodata:: foamlib.typing.Field
.. autodata:: foamlib.typing.Dict
.. autodata:: foamlib.typing.KeywordEntry
.. autodata:: foamlib.typing.List
.. autodata:: foamlib.typing.DataEntry
.. autodata:: foamlib.typing.Data
.. autodata:: foamlib.typing.StandaloneDataEntry
.. autodata:: foamlib.typing.StandaloneData
.. autodata:: foamlib.typing.SubDict
.. autodata:: foamlib.typing.FileDict


Other accepted types
--------------------

These "Like" type variants accept the standard type plus other formats that could potentially be converted to the standard type.

.. autodata:: foamlib.typing.TensorLike
.. autodata:: foamlib.typing.FieldLike
.. autodata:: foamlib.typing.DictLike
.. autodata:: foamlib.typing.KeywordEntryLike
.. autodata:: foamlib.typing.ListLike
.. autodata:: foamlib.typing.DimensionSetLike
.. autodata:: foamlib.typing.DataEntryLike
.. autodata:: foamlib.typing.DataLike
.. autodata:: foamlib.typing.StandaloneDataEntryLike
.. autodata:: foamlib.typing.StandaloneDataLike
.. autodata:: foamlib.typing.SubDictLike
.. autodata:: foamlib.typing.FileDictLike