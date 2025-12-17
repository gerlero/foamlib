#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdbool.h>
#include <ctype.h>
#include <string.h>
#include <math.h>

/* Skip whitespace and comments */
static const char* skip_whitespace_and_comments(const char* p, const char* end) {
    while (p < end) {
        /* Skip whitespace */
        if (isspace(*p)) {
            p++;
            continue;
        }
        
        /* Check for comments */
        if (p + 1 < end) {
            /* C++ style comment: // */
            if (p[0] == '/' && p[1] == '/') {
                p += 2;
                /* Skip until newline or end */
                while (p < end && *p != '\n') {
                    /* Handle line continuation */
                    if (*p == '\\' && p + 1 < end && p[1] == '\n') {
                        p += 2;
                        continue;
                    }
                    p++;
                }
                if (p < end && *p == '\n') {
                    p++;
                }
                continue;
            }
            
            /* C style comment: slash-star ... star-slash */
            if (p[0] == '/' && p[1] == '*') {
                p += 2;
                /* Skip until star-slash or end */
                while (p + 1 < end) {
                    if (p[0] == '*' && p[1] == '/') {
                        p += 2;
                        break;
                    }
                    p++;
                }
                continue;
            }
        }
        
        /* Not whitespace or comment, done */
        break;
    }
    return p;
}

/* Parse a single number (int or float) from a string */
static const char* parse_number(const char* str, const char* end, double* result, bool* is_int) {
    const char* p = str;
    bool negative = false;
    bool has_decimal = false;
    bool has_exp = false;
    
    /* Skip leading whitespace */
    while (p < end && isspace(*p)) {
        p++;
    }
    
    if (p >= end) {
        return NULL;
    }
    
    /* Handle sign */
    if (*p == '-') {
        negative = true;
        p++;
    } else if (*p == '+') {
        p++;
    }
    
    /* Check for special values: nan, inf, infinity */
    if (p < end) {
        /* Check for NaN */
        if (p + 3 <= end && (strncasecmp(p, "nan", 3) == 0)) {
            *result = NAN;
            *is_int = false;
            return p + 3;
        }
        
        /* Check for infinity */
        if (p + 3 <= end && (strncasecmp(p, "inf", 3) == 0)) {
            const char* next = p + 3;
            /* Check for full "infinity" */
            if (next + 5 <= end && strncasecmp(next, "inity", 5) == 0) {
                next += 5;
            }
            *result = negative ? -INFINITY : INFINITY;
            *is_int = false;
            return next;
        }
    }
    
    const char* start = p;
    
    /* Parse digits before decimal */
    while (p < end && isdigit(*p)) {
        p++;
    }
    
    /* Check for decimal point */
    if (p < end && *p == '.') {
        has_decimal = true;
        p++;
        
        /* Parse digits after decimal */
        while (p < end && isdigit(*p)) {
            p++;
        }
    }
    
    /* Check for exponent */
    if (p < end && (*p == 'e' || *p == 'E')) {
        has_exp = true;
        p++;
        
        /* Handle exponent sign */
        if (p < end && (*p == '+' || *p == '-')) {
            p++;
        }
        
        /* Parse exponent digits */
        while (p < end && isdigit(*p)) {
            p++;
        }
    }
    
    /* If we didn't parse anything, fail */
    if (p == start) {
        return NULL;
    }
    
    /* Convert to number */
    char buffer[128];
    /* Copy from original position (including any sign) to end of number */
    const char* copy_start = str;
    /* Skip any leading whitespace that was already skipped */
    while (copy_start < start && isspace(*copy_start)) {
        copy_start++;
    }
    
    size_t len = (size_t)(p - copy_start);
    if (len >= sizeof(buffer)) {
        return NULL;
    }
    memcpy(buffer, copy_start, len);
    buffer[len] = '\0';
    
    *is_int = !(has_decimal || has_exp);
    
    char* endptr;
    if (*is_int) {
        long long val = strtoll(buffer, &endptr, 10);
        /* Check for parsing errors */
        if (endptr == buffer || *endptr != '\0') {
            return NULL;
        }
        *result = (double)val;
    } else {
        *result = strtod(buffer, &endptr);
        /* Check for parsing errors */
        if (endptr == buffer || *endptr != '\0') {
            return NULL;
        }
    }
    
    return p;
}

/* Parse an ASCII numeric list from raw content
 * 
 * Args:
 *   contents: The full buffer containing the list
 *   start_pos: Position after the opening '(' 
 *   is_float: Whether to parse as float (1) or int (0)
 *   elshape: Element shape - 0 for scalars, 3 for vectors, 6 for symmTensors, 9 for tensors
 *
 * Returns:
 *   Tuple of (numpy_array, end_pos) where end_pos is position after the closing ')'
 */
static PyObject* parse_ascii_list(PyObject* self, PyObject* args) {
    Py_buffer buffer;
    Py_ssize_t start_pos;
    int is_float;
    int elshape;
    
    if (!PyArg_ParseTuple(args, "y*npi", &buffer, &start_pos, &is_float, &elshape)) {
        return NULL;
    }
    
    const char* contents = (const char*)buffer.buf;
    Py_ssize_t contents_len = buffer.len;
    const char* p = contents + start_pos;
    const char* end = contents + contents_len;
    
    /* Temporary storage for parsed values */
    size_t capacity = 1024;
    double* values = (double*)malloc(capacity * sizeof(double));
    if (values == NULL) {
        PyBuffer_Release(&buffer);
        return PyErr_NoMemory();
    }
    size_t count = 0;
    
    /* Track parenthesis depth for nested structures */
    int paren_depth = 0;
    
    /* Parse numbers until we hit the closing ')' at depth 0 */
    while (p < end) {
        /* Skip whitespace and comments */
        p = skip_whitespace_and_comments(p, end);
        
        if (p >= end) {
            free(values);
            PyBuffer_Release(&buffer);
            PyErr_SetString(PyExc_ValueError, "Unexpected end of input");
            return NULL;
        }
        
        /* Handle opening parenthesis */
        if (*p == '(') {
            /* For scalar lists (elshape==0), any opening paren means nesting */
            if (elshape == 0) {
                free(values);
                PyBuffer_Release(&buffer);
                PyErr_SetString(PyExc_ValueError, "Nested parentheses in scalar list");
                return NULL;
            }
            paren_depth++;
            p++;
            continue;
        }
        
        /* Handle closing parenthesis */
        if (*p == ')') {
            if (paren_depth == 0) {
                /* This is the final closing paren */
                p++;
                break;
            }
            paren_depth--;
            p++;
            continue;
        }
        
        /* Check for invalid characters that indicate this is not a numeric list */
        if (*p == '{' || *p == '}' || *p == '[' || *p == ']' || *p == ';') {
            free(values);
            PyBuffer_Release(&buffer);
            PyErr_SetString(PyExc_ValueError, "Invalid character for numeric list");
            return NULL;
        }
        
        /* Try to parse a number */
        double value;
        bool is_int_val;
        const char* next = parse_number(p, end, &value, &is_int_val);
        
        if (next == NULL) {
            /* Not a number - this is an error */
            free(values);
            PyBuffer_Release(&buffer);
            PyErr_Format(PyExc_ValueError, "Unexpected character '%c' in numeric list", *p);
            return NULL;
        }
        
        /* Check type compatibility */
        if (!is_float && !is_int_val) {
            free(values);
            PyBuffer_Release(&buffer);
            PyErr_SetString(PyExc_ValueError, "Found float value when parsing as integer");
            return NULL;
        }
        
        /* Expand storage if needed */
        if (count >= capacity) {
            capacity *= 2;
            double* new_values = (double*)realloc(values, capacity * sizeof(double));
            if (new_values == NULL) {
                free(values);
                PyBuffer_Release(&buffer);
                return PyErr_NoMemory();
            }
            values = new_values;
        }
        
        values[count++] = value;
        p = next;
    }
    
    /* Create numpy array */
    npy_intp dims[2];
    int ndim;
    
    if (elshape == 0) {
        /* Scalar list */
        dims[0] = count;
        ndim = 1;
    } else {
        /* Vector/tensor list */
        if (count % elshape != 0) {
            free(values);
            PyBuffer_Release(&buffer);
            PyErr_Format(PyExc_ValueError, 
                "Number of values (%zu) is not a multiple of element shape (%d)", 
                count, elshape);
            return NULL;
        }
        dims[0] = count / elshape;
        dims[1] = elshape;
        ndim = 2;
    }
    
    PyArrayObject* array;
    if (is_float) {
        array = (PyArrayObject*)PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    } else {
        array = (PyArrayObject*)PyArray_SimpleNew(ndim, dims, NPY_INT64);
    }
    
    if (array == NULL) {
        free(values);
        PyBuffer_Release(&buffer);
        return NULL;
    }
    
    /* Copy values into array */
    for (size_t i = 0; i < count; i++) {
        if (is_float) {
            if (ndim == 1) {
                *((double*)PyArray_GETPTR1(array, i)) = values[i];
            } else {
                *((double*)PyArray_GETPTR2(array, i / elshape, i % elshape)) = values[i];
            }
        } else {
            if (ndim == 1) {
                *((long long*)PyArray_GETPTR1(array, i)) = (long long)values[i];
            } else {
                *((long long*)PyArray_GETPTR2(array, i / elshape, i % elshape)) = (long long)values[i];
            }
        }
    }
    
    free(values);
    
    /* Return tuple of (array, end_position) */
    Py_ssize_t end_pos = p - contents;
    PyObject* result = Py_BuildValue("(Nn)", array, end_pos);
    
    PyBuffer_Release(&buffer);
    return result;
}

/* Parse an ASCII faces-like list from raw content
 * This handles lists where each element starts with a count: 3(a b c) or 4(a b c d)
 * 
 * Args:
 *   contents: The full buffer containing the list
 *   start_pos: Position after the opening '(' 
 *
 * Returns:
 *   Tuple of (values_array, end_pos) where values_array is a flat int64 array
 *   containing all values (including the counts), and end_pos is position after the closing ')'
 */
static PyObject* parse_ascii_faces_list(PyObject* self, PyObject* args) {
    Py_buffer buffer;
    Py_ssize_t start_pos;
    
    if (!PyArg_ParseTuple(args, "y*n", &buffer, &start_pos)) {
        return NULL;
    }
    
    const char* contents = (const char*)buffer.buf;
    Py_ssize_t contents_len = buffer.len;
    const char* p = contents + start_pos;
    const char* end = contents + contents_len;
    
    /* Temporary storage for parsed values */
    size_t capacity = 1024;
    long long* values = (long long*)malloc(capacity * sizeof(long long));
    if (values == NULL) {
        PyBuffer_Release(&buffer);
        return PyErr_NoMemory();
    }
    size_t count = 0;
    
    /* Track whether we're inside a face (between ( and )) */
    bool inside_face = false;
    
    /* Parse until we hit the closing ')' at top level */
    while (p < end) {
        /* Skip whitespace and comments */
        p = skip_whitespace_and_comments(p, end);
        
        if (p >= end) {
            free(values);
            PyBuffer_Release(&buffer);
            PyErr_SetString(PyExc_ValueError, "Unexpected end of input");
            return NULL;
        }
        
        /* Handle parentheses */
        if (*p == '(') {
            inside_face = true;
            p++;
            continue;
        }
        
        if (*p == ')') {
            if (inside_face) {
                /* Closing a face */
                inside_face = false;
                p++;
                continue;
            } else {
                /* Final closing paren */
                p++;
                break;
            }
        }
        
        /* Try to parse a number */
        double value;
        bool is_int_val;
        const char* next = parse_number(p, end, &value, &is_int_val);
        
        if (next == NULL) {
            /* Not a number, skip this character */
            p++;
            continue;
        }
        
        /* Must be an integer */
        if (!is_int_val) {
            free(values);
            PyBuffer_Release(&buffer);
            PyErr_SetString(PyExc_ValueError, "Expected integer in faces list");
            return NULL;
        }
        
        /* Expand storage if needed */
        if (count >= capacity) {
            capacity *= 2;
            long long* new_values = (long long*)realloc(values, capacity * sizeof(long long));
            if (new_values == NULL) {
                free(values);
                PyBuffer_Release(&buffer);
                return PyErr_NoMemory();
            }
            values = new_values;
        }
        
        values[count++] = (long long)value;
        p = next;
    }
    
    /* Create numpy array */
    npy_intp dims[1] = {count};
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT64);
    
    if (array == NULL) {
        free(values);
        PyBuffer_Release(&buffer);
        return NULL;
    }
    
    /* Copy values into array */
    for (size_t i = 0; i < count; i++) {
        *((long long*)PyArray_GETPTR1(array, i)) = values[i];
    }
    
    free(values);
    
    /* Return tuple of (array, end_position) */
    Py_ssize_t end_pos = p - contents;
    PyObject* result = Py_BuildValue("(Nn)", array, end_pos);
    
    PyBuffer_Release(&buffer);
    return result;
}

/* Module methods */
static PyMethodDef ParseAsciiMethods[] = {
    {"parse_ascii_list", parse_ascii_list, METH_VARARGS,
     "Parse an ASCII numeric list from raw content.\n\n"
     "Args:\n"
     "    contents (bytes): The full buffer containing the list\n"
     "    start_pos (int): Position after the opening '('\n"
     "    is_float (bool): Whether to parse as float (True) or int (False)\n"
     "    elshape (int): Element shape - 0 for scalars, 3 for vectors, 6 for symmTensors, 9 for tensors\n\n"
     "Returns:\n"
     "    tuple: (numpy_array, end_pos) where end_pos is position after the closing ')'\n"},
    {"parse_ascii_faces_list", parse_ascii_faces_list, METH_VARARGS,
     "Parse an ASCII faces-like list from raw content.\n\n"
     "Args:\n"
     "    contents (bytes): The full buffer containing the list\n"
     "    start_pos (int): Position after the opening '('\n\n"
     "Returns:\n"
     "    tuple: (values_array, end_pos) where values_array is a flat int64 array\n"},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef parse_ascii_module = {
    PyModuleDef_HEAD_INIT,
    "_parse_ascii",
    "C extension for parsing ASCII numeric lists",
    -1,
    ParseAsciiMethods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__parse_ascii(void) {
    import_array();
    return PyModule_Create(&parse_ascii_module);
}
