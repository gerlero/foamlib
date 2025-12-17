#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdbool.h>
#include <ctype.h>
#include <string.h>
#include <math.h>

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
    /* Calculate length from the beginning of sign/number to end of parsed content */
    const char* copy_start = str;
    while (copy_start < start && (*copy_start == '+' || *copy_start == '-')) {
        copy_start++;
    }
    if (copy_start > start) {
        copy_start = str;  /* Include the sign */
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

/* Parse a space-separated list of numbers from bytes */
static PyObject* parse_numeric_list(PyObject* self, PyObject* args) {
    Py_buffer buffer;
    int is_float;  /* 1 for float, 0 for int */
    
    if (!PyArg_ParseTuple(args, "y*p", &buffer, &is_float)) {
        return NULL;
    }
    
    const char* data = (const char*)buffer.buf;
    Py_ssize_t data_len = buffer.len;
    
    /* First pass: count numbers */
    size_t count = 0;
    const char* p = data;
    const char* end = data + data_len;
    
    while (p < end) {
        double value;
        bool is_int;
        const char* next = parse_number(p, end, &value, &is_int);
        
        if (next == NULL) {
            /* Skip this character and continue */
            p++;
            continue;
        }
        
        count++;
        p = next;
    }
    
    /* Create numpy array */
    npy_intp dims[1] = {count};
    PyArrayObject* array;
    
    if (is_float) {
        array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    } else {
        array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT64);
    }
    
    if (array == NULL) {
        PyBuffer_Release(&buffer);
        return NULL;
    }
    
    /* Second pass: fill array */
    p = data;
    size_t idx = 0;
    
    while (p < end && idx < count) {
        double value;
        bool is_int_val;
        const char* next = parse_number(p, end, &value, &is_int_val);
        
        if (next == NULL) {
            p++;
            continue;
        }
        
        /* If parsing as int but found a float, raise an error */
        if (!is_float && !is_int_val) {
            Py_DECREF(array);
            PyBuffer_Release(&buffer);
            PyErr_SetString(PyExc_ValueError, 
                "Found float value when parsing as integer");
            return NULL;
        }
        
        if (is_float) {
            *((double*)PyArray_GETPTR1(array, idx)) = value;
        } else {
            *((long long*)PyArray_GETPTR1(array, idx)) = (long long)value;
        }
        
        idx++;
        p = next;
    }
    
    PyBuffer_Release(&buffer);
    return (PyObject*)array;
}

/* Module methods */
static PyMethodDef ParseAsciiMethods[] = {
    {"parse_numeric_list", parse_numeric_list, METH_VARARGS,
     "Parse a space-separated list of numbers from bytes and return a numpy array."},
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
