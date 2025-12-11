🏷️ Typing (:py:mod:`foamlib.typing`)
====================================

Standard types
--------------

The following are aliases of the primary types used throughout **foamlib** to represent the equivalent OpenFOAM data structures.

.. note:: For concrete classes in **foamlib** that represent files and some stored data types (like :class:`foamlib.Dimensioned`), see the :class:`foamlib.FoamFile` section.

.. autodata:: foamlib.typing.File
.. autodata:: foamlib.typing.SubDict
.. autodata:: foamlib.typing.Data
.. autodata:: foamlib.typing.StandaloneData
.. autodata:: foamlib.typing.DataEntry
.. autodata:: foamlib.typing.StandaloneDataEntry
.. autodata:: foamlib.typing.Dict
.. autodata:: foamlib.typing.KeywordEntry
.. autodata:: foamlib.typing.Field
.. autodata:: foamlib.typing.Tensor


Other accepted types
--------------------

These "Like" type variants accept the standard type plus other formats that could potentially be converted to the standard type.

.. autodata:: foamlib.typing.FileLike
.. autodata:: foamlib.typing.SubDictLike
.. autodata:: foamlib.typing.DataLike
.. autodata:: foamlib.typing.StandaloneDataLike
.. autodata:: foamlib.typing.DataEntryLike
.. autodata:: foamlib.typing.StandaloneDataEntryLike
.. autodata:: foamlib.typing.DictLike
.. autodata:: foamlib.typing.KeywordEntryLike
.. autodata:: foamlib.typing.FieldLike
.. autodata:: foamlib.typing.TensorLike