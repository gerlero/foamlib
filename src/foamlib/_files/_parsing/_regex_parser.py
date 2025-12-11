"""Regex-based parser for OpenFOAM files working on bytes."""

import re
import sys
from typing import Any, Literal, TypeVar

import numpy as np
from numpy.typing import DTypeLike

if sys.version_info >= (3, 11):
    from typing import Never
else:
    from typing_extensions import Never

from .._common import dict_from_items
from .._typing import Data, Dict, File, StandaloneData, SubDict
from ..types import Dimensioned, DimensionSet

_T = TypeVar("_T")


class ParseError(ValueError):
    """Error raised when parsing fails."""

    def __init__(self, message: str, position: int = 0) -> None:
        super().__init__(message)
        self.position = position


def _skip_whitespace_and_comments(data: bytes, pos: int) -> int:
    """Skip whitespace and comments, return new position."""
    while pos < len(data):
        # Skip whitespace
        if data[pos:pos+1] in b' \t\n\r':
            pos += 1
            continue
        
        # Skip C++ style comments
        if data[pos:pos+2] == b'//':
            # Find end of line
            newline = data.find(b'\n', pos)
            if newline == -1:
                return len(data)
            pos = newline + 1
            continue
        
        # Skip C style comments
        if data[pos:pos+2] == b'/*':
            end = data.find(b'*/', pos + 2)
            if end == -1:
                raise ParseError("Unterminated comment", pos)
            pos = end + 2
            continue
        
        break
    
    return pos


def _match_pattern(pattern: bytes, data: bytes, pos: int, *, skip_ws: bool = True) -> tuple[int, bytes] | None:
    """Match a regex pattern at position, return (new_pos, matched_bytes) or None."""
    if skip_ws:
        pos = _skip_whitespace_and_comments(data, pos)
    
    match = re.match(pattern, data[pos:])
    if match:
        matched = match.group(0)
        return pos + len(matched), matched
    return None


def _parse_number(data: bytes, pos: int) -> tuple[int, int | float]:
    """Parse a number (int or float)."""
    pos = _skip_whitespace_and_comments(data, pos)
    
    # Try special float values first
    for pattern, value in [
        (rb'(?i:nan\b)', np.nan),
        (rb'(?i:inf(?:inity)?\b)', np.inf),
        (rb'(?i:-inf(?:inity)?\b)', -np.inf),
    ]:
        if result := _match_pattern(pattern, data, pos, skip_ws=False):
            return result[0], value
    
    # Regular number pattern
    float_pattern = rb'[+-]?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)'
    if result := _match_pattern(float_pattern, data, pos, skip_ws=False):
        matched = result[1]
        try:
            # Try int first
            if b'.' not in matched and b'e' not in matched.lower():
                return result[0], int(matched)
            return result[0], float(matched)
        except ValueError as e:
            raise ParseError(f"Invalid number: {matched}", pos) from e
    
    raise ParseError("Expected number", pos)


def _parse_bool(data: bytes, pos: int) -> tuple[int, bool]:
    """Parse a boolean value."""
    pos = _skip_whitespace_and_comments(data, pos)
    
    for pattern, value in [
        (rb'\byes\b', True),
        (rb'\btrue\b', True),
        (rb'\bon\b', True),
        (rb'\bno\b', False),
        (rb'\bfalse\b', False),
        (rb'\boff\b', False),
    ]:
        if result := _match_pattern(pattern, data, pos, skip_ws=False):
            return result[0], value
    
    raise ParseError("Expected boolean", pos)


def _parse_identifier(data: bytes, pos: int) -> tuple[int, str]:
    """Parse an identifier (keyword)."""
    pos = _skip_whitespace_and_comments(data, pos)
    
    # Identifier can include special chars but not ;(){}[]
    # Can also be a quoted string
    if data[pos:pos+1] == b'"':
        # Quoted string
        end = pos + 1
        while end < len(data) and data[end:end+1] != b'"':
            if data[end:end+1] == b'\\':
                end += 2
            else:
                end += 1
        if end >= len(data):
            raise ParseError("Unterminated string", pos)
        return end + 1, data[pos:end+1].decode('latin-1')
    
    # Regular identifier with optional balanced parentheses
    pattern = rb'[A-Za-z_$][^\s;(){}[\]]*'
    if result := _match_pattern(pattern, data, pos, skip_ws=False):
        new_pos, matched = result
        identifier = matched.decode('latin-1')
        
        # Check for balanced parentheses
        if new_pos < len(data) and data[new_pos:new_pos+1] == b'(':
            paren_start = new_pos
            new_pos += 1
            depth = 1
            while new_pos < len(data) and depth > 0:
                if data[new_pos:new_pos+1] == b'(':
                    depth += 1
                elif data[new_pos:new_pos+1] == b')':
                    depth -= 1
                new_pos += 1
            if depth != 0:
                raise ParseError("Unbalanced parentheses", paren_start)
            identifier = data[pos:new_pos].decode('latin-1')
        
        return new_pos, identifier
    
    raise ParseError("Expected identifier", pos)


def _parse_dimension_set(data: bytes, pos: int) -> tuple[int, DimensionSet]:
    """Parse a dimension set [kg m s K mol A cd]."""
    pos = _skip_whitespace_and_comments(data, pos)
    
    if data[pos:pos+1] != b'[':
        raise ParseError("Expected '['", pos)
    pos += 1
    
    dimensions = []
    for _ in range(7):  # 7 dimensions max
        pos = _skip_whitespace_and_comments(data, pos)
        if data[pos:pos+1] == b']':
            break
        try:
            pos, num = _parse_number(data, pos)
            dimensions.append(num)
        except ParseError:
            break
    
    pos = _skip_whitespace_and_comments(data, pos)
    if data[pos:pos+1] != b']':
        raise ParseError("Expected ']'", pos)
    pos += 1
    
    return pos, DimensionSet(*dimensions)


def _parse_tensor(data: bytes, pos: int) -> tuple[int, float | np.ndarray]:
    """Parse a tensor (scalar or vector/symmTensor/tensor)."""
    pos = _skip_whitespace_and_comments(data, pos)
    
    # Try scalar first
    try:
        return _parse_number(data, pos)
    except ParseError:
        pass
    
    # Parse parenthesized tensor
    if data[pos:pos+1] != b'(':
        raise ParseError("Expected number or '('", pos)
    pos += 1
    
    values = []
    while True:
        try:
            pos, num = _parse_number(data, pos)
            values.append(num)
        except ParseError:
            break
    
    pos = _skip_whitespace_and_comments(data, pos)
    if data[pos:pos+1] != b')':
        raise ParseError("Expected ')'", pos)
    pos += 1
    
    if len(values) not in (3, 6, 9):
        raise ParseError(f"Invalid tensor size: {len(values)}", pos)
    
    return pos, np.array(values, dtype=float)


def _parse_dimensioned(data: bytes, pos: int) -> tuple[int, Dimensioned]:
    """Parse a dimensioned value: [name] [dimensions] value."""
    start_pos = pos
    pos = _skip_whitespace_and_comments(data, pos)
    
    # Try to parse optional name
    name = None
    try:
        temp_pos, name = _parse_identifier(data, pos)
        # Check if next is a dimension set
        temp_pos2 = _skip_whitespace_and_comments(data, temp_pos)
        if data[temp_pos2:temp_pos2+1] == b'[':
            pos = temp_pos
    except ParseError:
        pass
    
    # Parse dimensions
    try:
        pos, dimensions = _parse_dimension_set(data, pos)
    except ParseError:
        raise ParseError("Expected dimension set in dimensioned value", start_pos) from None
    
    # Parse value
    pos, value = _parse_tensor(data, pos)
    
    return pos, Dimensioned(value, dimensions, name)


def _parse_ascii_list(data: bytes, pos: int, dtype: DTypeLike, elshape: tuple[int, ...] = (), empty_ok: bool = False) -> tuple[int, np.ndarray]:
    """Parse an ASCII numeric list."""
    pos = _skip_whitespace_and_comments(data, pos)
    
    requested_dtype = np.dtype(dtype)
    
    # Try counted format: N(...)
    count_match = re.match(rb'(\d+)', data[pos:])
    if count_match:
        count_str = count_match.group(1)
        temp_pos = pos + len(count_str)
        temp_pos = _skip_whitespace_and_comments(data, temp_pos)
        
        # Check for {...} format (repeated value)
        if data[temp_pos:temp_pos+1] == b'{':
            temp_pos += 1
            # Parse the repeated element
            if elshape:
                temp_pos = _skip_whitespace_and_comments(data, temp_pos)
                if data[temp_pos:temp_pos+1] != b'(':
                    raise ParseError("Expected '('", temp_pos)
                temp_pos += 1
                values = []
                for _ in range(elshape[0]):
                    temp_pos, num = _parse_number(data, temp_pos)
                    values.append(num)
                temp_pos = _skip_whitespace_and_comments(data, temp_pos)
                if data[temp_pos:temp_pos+1] != b')':
                    raise ParseError("Expected ')'", temp_pos)
                temp_pos += 1
            else:
                temp_pos, value = _parse_number(data, temp_pos)
                values = [value]
            
            temp_pos = _skip_whitespace_and_comments(data, temp_pos)
            if data[temp_pos:temp_pos+1] != b'}':
                raise ParseError("Expected '}'", temp_pos)
            temp_pos += 1
            
            count = int(count_str)
            # Auto-detect dtype if needed
            actual_dtype = requested_dtype
            if any(isinstance(v, float) for v in values):
                actual_dtype = np.float64
            arr = np.array(values, dtype=actual_dtype)
            if elshape:
                arr = np.tile(arr, (count, 1))
            else:
                arr = np.repeat(arr, count)
            
            return temp_pos, arr
    
    # Standard format: [N](...) 
    count = None
    if count_match:
        count = int(count_str)
        pos += len(count_str)
        pos = _skip_whitespace_and_comments(data, pos)
    
    if data[pos:pos+1] != b'(':
        raise ParseError("Expected '('", pos)
    pos += 1
    
    # Parse elements
    elements = []
    while True:
        pos = _skip_whitespace_and_comments(data, pos)
        if data[pos:pos+1] == b')':
            break
        
        if elshape:
            # Parse element tuple
            if data[pos:pos+1] != b'(':
                raise ParseError("Expected '('", pos)
            pos += 1
            element_values = []
            for _ in range(elshape[0]):
                pos, num = _parse_number(data, pos)
                element_values.append(num)
            pos = _skip_whitespace_and_comments(data, pos)
            if data[pos:pos+1] != b')':
                raise ParseError("Expected ')'", pos)
            pos += 1
            elements.append(element_values)
        else:
            # Parse scalar
            pos, num = _parse_number(data, pos)
            elements.append(num)
    
    pos += 1  # Skip closing ')'
    
    if not elements and not empty_ok:
        raise ParseError("Empty list not allowed", pos)
    
    if count is not None and len(elements) != count:
        raise ParseError(f"Expected {count} elements, got {len(elements)}", pos)
    
    # Auto-detect dtype if needed - if any element is float, use float
    actual_dtype = requested_dtype
    flat_elements = []
    for el in elements:
        if isinstance(el, list):
            flat_elements.extend(el)
        else:
            flat_elements.append(el)
    if any(isinstance(v, float) for v in flat_elements):
        actual_dtype = np.float64
    
    arr = np.array(elements, dtype=actual_dtype)
    if elshape and arr.ndim == 1:
        arr = arr.reshape(-1, *elshape)
    
    return pos, arr


def _parse_binary_list(data: bytes, pos: int, dtype: DTypeLike, elshape: tuple[int, ...] = (), empty_ok: bool = False) -> tuple[int, np.ndarray]:
    """Parse a binary numeric list."""
    pos = _skip_whitespace_and_comments(data, pos)
    
    dtype = np.dtype(dtype)
    
    # Parse count
    count_match = re.match(rb'(\d+)', data[pos:])
    if not count_match:
        raise ParseError("Expected count for binary list", pos)
    
    count = int(count_match.group(1))
    pos += len(count_match.group(1))
    pos = _skip_whitespace_and_comments(data, pos)
    
    if count == 0 and not empty_ok:
        raise ParseError("Empty binary list not allowed", pos)
    
    if data[pos:pos+1] != b'(':
        raise ParseError("Expected '('", pos)
    pos += 1
    
    # Calculate total bytes needed
    elsize = 1 if not elshape else elshape[0]
    total_bytes = dtype.itemsize * elsize * count
    
    # Read binary data
    binary_data = data[pos:pos+total_bytes]
    if len(binary_data) != total_bytes:
        raise ParseError("Insufficient binary data", pos)
    pos += total_bytes
    
    if data[pos:pos+1] != b')':
        raise ParseError("Expected ')' after binary data", pos)
    pos += 1
    
    # Convert to array
    arr = np.frombuffer(binary_data, dtype=dtype)
    if elshape:
        arr = arr.reshape(count, *elshape)
    
    return pos, arr


def _parse_field(data: bytes, pos: int) -> tuple[int, float | np.ndarray]:
    """Parse a field (uniform or nonuniform)."""
    pos = _skip_whitespace_and_comments(data, pos)
    
    # Try uniform
    if result := _match_pattern(rb'\buniform\b', data, pos, skip_ws=False):
        pos = result[0]
        return _parse_tensor(data, pos)
    
    # Try nonuniform
    if result := _match_pattern(rb'\bnonuniform\b', data, pos, skip_ws=False):
        pos = result[0]
        pos = _skip_whitespace_and_comments(data, pos)
        
        # Check for optional List<type> syntax
        if result := _match_pattern(rb'\bList\s*<\s*scalar\s*>\s*', data, pos, skip_ws=False):
            pos = result[0]
            # Try binary first, then ASCII
            try:
                return _parse_binary_list(data, pos, np.float64, empty_ok=True)
            except ParseError:
                try:
                    return _parse_binary_list(data, pos, np.float32, empty_ok=True)
                except ParseError:
                    return _parse_ascii_list(data, pos, float, empty_ok=True)
        
        if result := _match_pattern(rb'\bList\s*<\s*vector\s*>\s*', data, pos, skip_ws=False):
            pos = result[0]
            try:
                return _parse_binary_list(data, pos, np.float64, elshape=(3,), empty_ok=True)
            except ParseError:
                try:
                    return _parse_binary_list(data, pos, np.float32, elshape=(3,), empty_ok=True)
                except ParseError:
                    return _parse_ascii_list(data, pos, float, elshape=(3,), empty_ok=True)
        
        if result := _match_pattern(rb'\bList\s*<\s*symmTensor\s*>\s*', data, pos, skip_ws=False):
            pos = result[0]
            try:
                return _parse_binary_list(data, pos, np.float64, elshape=(6,), empty_ok=True)
            except ParseError:
                try:
                    return _parse_binary_list(data, pos, np.float32, elshape=(6,), empty_ok=True)
                except ParseError:
                    return _parse_ascii_list(data, pos, float, elshape=(6,), empty_ok=True)
        
        if result := _match_pattern(rb'\bList\s*<\s*tensor\s*>\s*', data, pos, skip_ws=False):
            pos = result[0]
            try:
                return _parse_binary_list(data, pos, np.float64, elshape=(9,), empty_ok=True)
            except ParseError:
                try:
                    return _parse_binary_list(data, pos, np.float32, elshape=(9,), empty_ok=True)
                except ParseError:
                    return _parse_ascii_list(data, pos, float, elshape=(9,), empty_ok=True)
        
        # Bare nonuniform - try to infer type
        # Try scalar
        try:
            return _parse_binary_list(data, pos, np.float64, empty_ok=True)
        except ParseError:
            pass
        try:
            return _parse_binary_list(data, pos, np.float32, empty_ok=True)
        except ParseError:
            pass
        try:
            return _parse_ascii_list(data, pos, float, empty_ok=True)
        except ParseError:
            pass
        
        # Try vector
        try:
            return _parse_binary_list(data, pos, np.float64, elshape=(3,), empty_ok=True)
        except ParseError:
            pass
        try:
            return _parse_binary_list(data, pos, np.float32, elshape=(3,), empty_ok=True)
        except ParseError:
            pass
        
        raise ParseError("Could not parse nonuniform field", pos)
    
    raise ParseError("Expected uniform or nonuniform field", pos)


def _parse_list(data: bytes, pos: int) -> tuple[int, list[Any]]:
    """Parse a list (...) or N{value}."""
    pos = _skip_whitespace_and_comments(data, pos)
    
    # Check for N{value} format
    count_match = re.match(rb'(\d+)', data[pos:])
    if count_match:
        temp_pos = pos + len(count_match.group(1))
        temp_pos = _skip_whitespace_and_comments(data, temp_pos)
        if data[temp_pos:temp_pos+1] == b'{':
            count = int(count_match.group(1))
            temp_pos += 1
            # Parse single value
            temp_pos, value = _parse_data_entry(data, temp_pos)
            temp_pos = _skip_whitespace_and_comments(data, temp_pos)
            if data[temp_pos:temp_pos+1] != b'}':
                raise ParseError("Expected '}'", temp_pos)
            temp_pos += 1
            return temp_pos, [value] * count
    
    # Check for N(...) format (counted array)
    if count_match:
        count = int(count_match.group(1))
        temp_pos = pos + len(count_match.group(1))
        temp_pos = _skip_whitespace_and_comments(data, temp_pos)
        if data[temp_pos:temp_pos+1] == b'(':
            temp_pos += 1
            elements = []
            for _ in range(count):
                temp_pos, value = _parse_data_entry(data, temp_pos)
                elements.append(value)
            temp_pos = _skip_whitespace_and_comments(data, temp_pos)
            if data[temp_pos:temp_pos+1] != b')':
                raise ParseError("Expected ')'", temp_pos)
            temp_pos += 1
            return temp_pos, elements
    
    # Standard (...) format
    if data[pos:pos+1] != b'(':
        raise ParseError("Expected '('", pos)
    pos += 1
    
    elements = []
    while True:
        pos = _skip_whitespace_and_comments(data, pos)
        if data[pos:pos+1] == b')':
            break
        
        # Try dict first
        if data[pos:pos+1] == b'{':
            try:
                pos, value = _parse_dict(data, pos)
                elements.append(value)
                continue
            except ParseError:
                pass
        
        # Try keyword entry pattern
        start_pos = pos
        keyword_value = None
        
        # Try identifier or list as keyword
        if data[pos:pos+1] == b'(':
            # List as keyword
            try:
                pos, keyword_value = _parse_list(data, pos)
            except ParseError:
                pass
        else:
            # Identifier as keyword
            try:
                pos, keyword_value = _parse_identifier(data, pos)
            except ParseError:
                pass
        
        if keyword_value is not None:
            temp_pos = _skip_whitespace_and_comments(data, pos)
            
            # Check for dict after keyword
            if data[temp_pos:temp_pos+1] == b'{':
                pos, value = _parse_dict(data, temp_pos)
                elements.append((keyword_value, value))
                continue
            
            # Check for data; pattern
            try:
                pos, value = _parse_data(data, temp_pos)
                temp_pos2 = _skip_whitespace_and_comments(data, pos)
                if data[temp_pos2:temp_pos2+1] == b';':
                    pos = temp_pos2 + 1
                    elements.append((keyword_value, value))
                    continue
            except ParseError:
                pass
        
        # Not a keyword entry, revert and parse as data entry
        pos = start_pos
        pos, value = _parse_data_entry(data, pos)
        elements.append(value)
    
    pos += 1  # Skip ')'
    
    return pos, elements


def _parse_dict(data: bytes, pos: int) -> tuple[int, Dict]:
    """Parse a dictionary {...}."""
    pos = _skip_whitespace_and_comments(data, pos)
    
    if data[pos:pos+1] != b'{':
        raise ParseError("Expected '{'", pos)
    pos += 1
    
    entries = []
    while True:
        pos = _skip_whitespace_and_comments(data, pos)
        if data[pos:pos+1] == b'}':
            break
        
        # Parse keyword
        pos, keyword = _parse_identifier(data, pos)
        pos = _skip_whitespace_and_comments(data, pos)
        
        # Parse value (dict or data;)
        if data[pos:pos+1] == b'{':
            pos, value = _parse_dict(data, pos)
            entries.append((keyword, value))
        else:
            pos, value = _parse_data(data, pos)
            pos = _skip_whitespace_and_comments(data, pos)
            if data[pos:pos+1] != b';':
                raise ParseError("Expected ';'", pos)
            pos += 1
            entries.append((keyword, value))
    
    pos += 1  # Skip '}'
    
    return pos, dict_from_items(entries, target=Dict)


def _parse_directive(data: bytes, pos: int) -> tuple[int, str]:
    """Parse a directive #include, #directive, etc."""
    pos = _skip_whitespace_and_comments(data, pos)
    
    if data[pos:pos+1] != b'#':
        raise ParseError("Expected '#'", pos)
    
    pattern = rb'#[^\s;(){}[\]]*'
    if result := _match_pattern(pattern, data, pos, skip_ws=False):
        return result[0], result[1].decode('latin-1')
    
    raise ParseError("Invalid directive", pos)


def _try_parse_faces_list(data: bytes, pos: int) -> tuple[int, list[np.ndarray]] | None:
    """Try to parse a faces-like list."""
    start_pos = pos
    pos = _skip_whitespace_and_comments(data, pos)
    
    # Optional count
    count_match = re.match(rb'(\d+)', data[pos:])
    if count_match:
        count = int(count_match.group(1))
        pos += len(count_match.group(1))
        pos = _skip_whitespace_and_comments(data, pos)
    else:
        count = None
    
    if data[pos:pos+1] != b'(':
        return None
    pos += 1
    
    faces = []
    try:
        while True:
            pos = _skip_whitespace_and_comments(data, pos)
            if data[pos:pos+1] == b')':
                break
            
            # Parse face size (3 or 4)
            face_size_match = re.match(rb'([34])', data[pos:])
            if not face_size_match:
                return None
            
            face_size = int(face_size_match.group(1))
            pos += 1
            pos = _skip_whitespace_and_comments(data, pos)
            
            if data[pos:pos+1] != b'(':
                return None
            pos += 1
            
            # Parse face indices
            indices = []
            for _ in range(face_size):
                pos, idx = _parse_number(data, pos)
                if not isinstance(idx, int):
                    return None
                indices.append(idx)
            
            pos = _skip_whitespace_and_comments(data, pos)
            if data[pos:pos+1] != b')':
                return None
            pos += 1
            
            faces.append(np.array(indices, dtype=np.int64))
    except ParseError:
        return None
    
    pos += 1  # Skip closing ')'
    
    if count is not None and len(faces) != count:
        return None
    
    return pos, faces


def _parse_data_entry(data: bytes, pos: int) -> tuple[int, Any]:
    """Parse a single data entry (not standalone)."""
    pos = _skip_whitespace_and_comments(data, pos)
    
    # Try field (uniform/nonuniform)
    if data[pos:pos+7] == b'uniform' or data[pos:pos+10] == b'nonuniform':
        try:
            return _parse_field(data, pos)
        except ParseError:
            pass
    
    # Try list (generic, not numeric list)
    if data[pos:pos+1] == b'(' or (pos < len(data) - 1 and data[pos:pos+1].isdigit()):
        # Try regular list
        try:
            return _parse_list(data, pos)
        except ParseError:
            pass
    
    # Try dimensioned value
    try:
        return _parse_dimensioned(data, pos)
    except ParseError:
        pass
    
    # Try dimension set alone
    if data[pos:pos+1] == b'[':
        try:
            return _parse_dimension_set(data, pos)
        except ParseError:
            pass
    
    # Try number
    try:
        return _parse_number(data, pos)
    except ParseError:
        pass
    
    # Try bool
    try:
        return _parse_bool(data, pos)
    except ParseError:
        pass
    
    # Try directive
    if data[pos:pos+1] == b'#':
        try:
            return _parse_directive(data, pos)
        except ParseError:
            pass
    
    # Try identifier/keyword
    try:
        return _parse_identifier(data, pos)
    except ParseError:
        pass
    
    # Try dict
    if data[pos:pos+1] == b'{':
        try:
            return _parse_dict(data, pos)
        except ParseError:
            pass
    
    raise ParseError(f"Could not parse data entry at position {pos}", pos)


def _parse_data(data: bytes, pos: int) -> tuple[int, Data]:
    """Parse data (one or more data entries)."""
    entries = []
    start_pos = pos
    
    while True:
        try:
            pos, entry = _parse_data_entry(data, pos)
            entries.append(entry)
            # Check if there's more (not followed by ; or })
            temp_pos = _skip_whitespace_and_comments(data, pos)
            if temp_pos >= len(data) or data[temp_pos:temp_pos+1] in b';})':
                break
        except ParseError:
            break
    
    if not entries:
        raise ParseError("Expected data", start_pos)
    
    if len(entries) == 1:
        return pos, entries[0]
    return pos, tuple(entries)


def _parse_keyword_entry(data: bytes, pos: int) -> tuple[int, tuple[str, Any]]:
    """Parse a keyword entry (keyword value; or keyword {...} or keyword;)."""
    pos = _skip_whitespace_and_comments(data, pos)
    
    # Check for directive
    if data[pos:pos+1] == b'#':
        pos, directive = _parse_directive(data, pos)
        # Parse optional data entry after directive
        try:
            pos, value = _parse_data_entry(data, pos)
        except ParseError:
            value = None
        # Directives end at line end
        while pos < len(data) and data[pos:pos+1] not in b'\n\r':
            pos += 1
        if pos < len(data):
            pos += 1
        return pos, (directive, value)
    
    # Parse keyword
    pos, keyword = _parse_identifier(data, pos)
    pos = _skip_whitespace_and_comments(data, pos)
    
    # Check for ; immediately (keyword with no value)
    if data[pos:pos+1] == b';':
        pos += 1
        return pos, (keyword, None)
    
    # Parse value (dict or data;)
    if data[pos:pos+1] == b'{':
        pos, value = _parse_dict(data, pos)
        return pos, (keyword, value)
    else:
        pos, value = _parse_data(data, pos)
        pos = _skip_whitespace_and_comments(data, pos)
        if data[pos:pos+1] != b';':
            raise ParseError("Expected ';'", pos)
        pos += 1
        return pos, (keyword, value)


def _parse_standalone_data(data: bytes, pos: int) -> tuple[int, StandaloneData]:
    """Parse standalone data (for files with just data, no keywords)."""
    pos = _skip_whitespace_and_comments(data, pos)
    
    # Try faces list first
    if result := _try_parse_faces_list(data, pos):
        return result
    
    # Try ASCII numeric lists
    try:
        return _parse_ascii_list(data, pos, int)
    except ParseError:
        pass
    try:
        return _parse_ascii_list(data, pos, float)
    except ParseError:
        pass
    try:
        return _parse_ascii_list(data, pos, float, elshape=(3,))
    except ParseError:
        pass
    
    # Try binary lists
    count_match = re.match(rb'(\d+)', data[pos:])
    if count_match:
        try:
            return _parse_binary_list(data, pos, np.int32)
        except ParseError:
            pass
        try:
            return _parse_binary_list(data, pos, np.float64)
        except ParseError:
            pass
        try:
            return _parse_binary_list(data, pos, np.float64, elshape=(3,))
        except ParseError:
            pass
        try:
            return _parse_binary_list(data, pos, np.float32, elshape=(3,))
        except ParseError:
            pass
    
    # Try generic data entry
    try:
        pos, value = _parse_data_entry(data, pos)
        # If it's a tuple, return as-is, otherwise return the single value
        return pos, value
    except ParseError:
        pass
    
    raise ParseError("Could not parse standalone data", pos)


def parse_file(data: bytes) -> File:
    """Parse a complete OpenFOAM file."""
    pos = 0
    entries = []
    standalone_data_entry = None
    
    while pos < len(data):
        pos = _skip_whitespace_and_comments(data, pos)
        if pos >= len(data):
            break
        
        # Try keyword entry
        try:
            pos, entry = _parse_keyword_entry(data, pos)
            entries.append(entry)
            continue
        except ParseError:
            pass
        
        # Try standalone data (only once, in the middle)
        if standalone_data_entry is None:
            try:
                pos, standalone_data = _parse_standalone_data(data, pos)
                standalone_data_entry = (None, standalone_data)
                entries.append(standalone_data_entry)
                continue
            except ParseError:
                pass
        
        # If we get here, we couldn't parse anything
        raise ParseError(f"Could not parse file at position {pos}", pos)
    
    return dict_from_items(entries, target=File)


def parse_token(data: bytes) -> str:
    """Parse a single token (identifier or directive)."""
    pos = _skip_whitespace_and_comments(data, 0)
    
    if data[pos:pos+1] == b'#':
        pos, directive = _parse_directive(data, pos)
        return directive
    
    pos, identifier = _parse_identifier(data, pos)
    pos = _skip_whitespace_and_comments(data, pos)
    
    if pos < len(data):
        raise ParseError(f"Unexpected data after token at position {pos}", pos)
    
    return identifier


def parse_data(data: bytes) -> Data:
    """Parse data value(s)."""
    pos, result = _parse_data(data, 0)
    pos = _skip_whitespace_and_comments(data, pos)
    
    if pos < len(data):
        raise ParseError(f"Unexpected data after value at position {pos}", pos)
    
    return result


def parse_standalone_data(data: bytes) -> StandaloneData:
    """Parse standalone data."""
    pos, result = _parse_standalone_data(data, 0)
    pos = _skip_whitespace_and_comments(data, pos)
    
    if pos < len(data):
        raise ParseError(f"Unexpected data after standalone data at position {pos}", pos)
    
    return result


def parse_file_with_locations(data: bytes) -> list[tuple[tuple[str | None, Any], int, int]]:
    """Parse a file and return entries with their byte positions."""
    pos = 0
    entries = []
    standalone_entries = []
    standalone_start = -1
    
    while pos < len(data):
        start = _skip_whitespace_and_comments(data, pos)
        if start >= len(data):
            break
        
        pos = start
        
        # Try keyword entry
        try:
            end_pos, entry = _parse_keyword_entry(data, pos)
            # If we have accumulated standalone entries, add them first
            if standalone_entries:
                combined = standalone_entries[0] if len(standalone_entries) == 1 else tuple(standalone_entries)
                entries.append(((None, combined), standalone_start, start))
                standalone_entries = []
                standalone_start = -1
            entries.append((entry, start, end_pos))
            pos = end_pos
            continue
        except ParseError:
            pass
        
        # Try standalone data entry (tries numeric lists first)
        try:
            # Use _parse_standalone_data_entry for first attempt
            end_pos = pos
            # Try faces list first
            if result := _try_parse_faces_list(data, end_pos):
                end_pos, standalone_data = result
            else:
                # Try ASCII numeric lists
                parsed = False
                for dtype, elshape in [
                    (int, ()),
                    (float, ()),
                    (float, (3,)),
                ]:
                    try:
                        end_pos, standalone_data = _parse_ascii_list(data, pos, dtype, elshape=elshape)
                        parsed = True
                        break
                    except ParseError:
                        pass
                
                if not parsed:
                    # Try binary lists
                    count_match = re.match(rb'(\d+)', data[pos:])
                    if count_match:
                        for dtype, elshape in [
                            (np.int32, ()),
                            (np.float64, ()),
                            (np.float64, (3,)),
                            (np.float32, (3,)),
                        ]:
                            try:
                                end_pos, standalone_data = _parse_binary_list(data, pos, dtype, elshape=elshape)
                                parsed = True
                                break
                            except ParseError:
                                pass
                
                if not parsed:
                    # Fall back to _parse_data_entry
                    end_pos, standalone_data = _parse_data_entry(data, pos)
            
            if not standalone_entries:
                standalone_start = start
            standalone_entries.append(standalone_data)
            pos = end_pos
            continue
        except ParseError:
            pass
        
        # If we get here, we couldn't parse anything
        raise ParseError(f"Could not parse file at position {pos}", pos)
    
    # Add any remaining standalone entries
    if standalone_entries:
        combined = standalone_entries[0] if len(standalone_entries) == 1 else tuple(standalone_entries)
        entries.append(((None, combined), standalone_start, pos))
    
    return entries
