#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <errno.h>

static PyObject *FoamFileDecodeError = NULL;
static PyObject *ParseError = NULL;

/* Python function: _skip(contents: bytes | bytearray, pos: int, *, newline_ok: bool = True) -> int */
static PyObject *
skip(PyObject *self, PyObject *args, PyObject *kwargs)
{
    Py_buffer buffer;
    Py_ssize_t pos;
    int newline_ok = 1;
    const unsigned char *contents;
    Py_ssize_t len;
    
    static char *kwlist[] = {"contents", "pos", "newline_ok", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y*n|$p", kwlist,
                                     &buffer, &pos, &newline_ok)) {
        return NULL;
    }
    
    contents = (const unsigned char *)buffer.buf;
    len = buffer.len;
    
    /* Main loop: skip whitespace and comments */
    while (1) {
        /* Skip whitespace */
        while (pos < len) {
            unsigned char c = contents[pos];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v' ||
                (c == '\n' && newline_ok)) {
                pos++;
            } else {
                break;
            }
        }
        
        /* Check for comments */
        if (pos + 1 >= len) {
            break;  /* Not enough characters for a comment */
        }
        
        unsigned char next1 = contents[pos];
        unsigned char next2 = contents[pos + 1];
        
        /* Handle single-line comments: // */
        if (next1 == '/' && next2 == '/') {
            pos += 2;
            while (pos < len) {
                unsigned char c = contents[pos];
                if (c == '\n') {
                    if (newline_ok) {
                        pos++;
                    }
                    break;
                }
                /* Handle line continuation with backslash */
                if (c == '\\' && pos + 1 < len && contents[pos + 1] == '\n') {
                    pos += 2;
                    continue;
                }
                pos++;
            }
            continue;  /* Continue outer loop to skip more whitespace */
        }
        
        /* Handle multi-line comments: slash-star ... star-slash */
        if (next1 == '/' && next2 == '*') {
            pos += 2;
            /* Find closing star-slash */
            int found = 0;
            while (pos + 1 < len) {
                if (contents[pos] == '*' && contents[pos + 1] == '/') {
                    pos += 2;
                    found = 1;
                    break;
                }
                pos++;
            }
            if (!found) {
                /* Unclosed comment - raise error at end of file */
                /* Get the original object that owns the buffer */
                PyObject *contents_obj = buffer.obj;
                Py_INCREF(contents_obj);
                PyBuffer_Release(&buffer);
                
                /* Call FoamFileDecodeError with position at end of file */
                PyObject *kwargs = Py_BuildValue("{s:s}", "expected", "*/");
                if (kwargs == NULL) {
                    Py_DECREF(contents_obj);
                    return NULL;
                }
                PyObject *args = Py_BuildValue("(On)", contents_obj, len);
                if (args == NULL) {
                    Py_DECREF(contents_obj);
                    Py_DECREF(kwargs);
                    return NULL;
                }
                PyObject *error = PyObject_Call(FoamFileDecodeError, args, kwargs);
                Py_DECREF(contents_obj);
                Py_DECREF(args);
                Py_DECREF(kwargs);
                if (error != NULL) {
                    PyErr_SetObject(FoamFileDecodeError, error);
                    Py_DECREF(error);
                }
                return NULL;
            }
            continue;  /* Continue outer loop to skip more whitespace */
        }
        
        /* No more whitespace or comments */
        break;
    }
    
    PyBuffer_Release(&buffer);
    return PyLong_FromSsize_t(pos);
}

/* Lookup tables for _parse_number */
static int _IS_POSSIBLE_FLOAT[256] = {0};
static int _IS_POSSIBLE_INTEGER[256] = {0};
static int _IS_TOKEN_CONTINUATION[256] = {0};

static void
init_lookup_tables(void)
{
    /* Initialize _IS_POSSIBLE_FLOAT */
    const char *float_chars = "0123456789.-+eEinfnatyINFNATY";
    for (int i = 0; float_chars[i]; i++) {
        _IS_POSSIBLE_FLOAT[(unsigned char)float_chars[i]] = 1;
    }
    
    /* Initialize _IS_POSSIBLE_INTEGER */
    const char *int_chars = "0123456789-+";
    for (int i = 0; int_chars[i]; i++) {
        _IS_POSSIBLE_INTEGER[(unsigned char)int_chars[i]] = 1;
    }
    
    /* Initialize _IS_TOKEN_CONTINUATION */
    const char *token_start = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_#$";
    const char *token_continuation = "0123456789._<>#$:+-*/|^%&=!";
    for (int i = 0; token_start[i]; i++) {
        _IS_TOKEN_CONTINUATION[(unsigned char)token_start[i]] = 1;
    }
    for (int i = 0; token_continuation[i]; i++) {
        _IS_TOKEN_CONTINUATION[(unsigned char)token_continuation[i]] = 1;
    }
}

/* Target type enum for _parse_number */
enum ParseNumberTarget {
    TARGET_INT = 1,
    TARGET_FLOAT = 2,
    TARGET_INT_OR_FLOAT = 3
};

/* Helper function to raise ParseError with the given parameters */
static void
raise_parse_error(PyObject *contents_obj, Py_ssize_t pos, const char *expected)
{
    /* Create ParseError instance with attributes */
    PyObject *exc_instance = PyObject_CallObject(ParseError, NULL);
    if (exc_instance == NULL) {
        return;
    }
    
    /* Set attributes on the exception instance */
    PyObject_SetAttrString(exc_instance, "_contents", contents_obj);
    PyObject_SetAttrString(exc_instance, "pos", PyLong_FromSsize_t(pos));
    PyObject_SetAttrString(exc_instance, "_expected", PyUnicode_FromString(expected));
    
    /* Set the exception */
    PyErr_SetObject(ParseError, exc_instance);
    Py_DECREF(exc_instance);
}

/* Python function: _parse_number(contents: bytes | bytearray, pos: int, *, target: type[int|float|int|float]) 
   -> tuple[int|float, int] */
static PyObject *
parse_number(PyObject *self, PyObject *args, PyObject *kwargs)
{
    Py_buffer buffer;
    Py_ssize_t pos;
    PyObject *target = NULL;
    const unsigned char *contents;
    Py_ssize_t len;
    enum ParseNumberTarget target_type = TARGET_INT_OR_FLOAT;
    
    static char *kwlist[] = {"contents", "pos", "target", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y*n|$O", kwlist,
                                     &buffer, &pos, &target)) {
        return NULL;
    }
    
    contents = (const unsigned char *)buffer.buf;
    len = buffer.len;
    
    /* Determine target type */
    if (target != NULL) {
        /* Check if target is int type */
        if (target == (PyObject *)&PyLong_Type) {
            target_type = TARGET_INT;
        }
        /* Check if target is float type */
        else if (target == (PyObject *)&PyFloat_Type) {
            target_type = TARGET_FLOAT;
        }
        /* Otherwise assume int | float */
    }
    
    /* Select the appropriate lookup table */
    int *is_numeric = (target_type == TARGET_INT) ? _IS_POSSIBLE_INTEGER : _IS_POSSIBLE_FLOAT;
    
    /* Find the end of the numeric string */
    Py_ssize_t end = pos;
    while (end < len && is_numeric[contents[end]]) {
        end++;
    }
    
    /* Check if followed by token continuation character */
    if (end < len && _IS_TOKEN_CONTINUATION[contents[end]]) {
        PyObject *contents_obj = buffer.obj;
        Py_INCREF(contents_obj);
        PyBuffer_Release(&buffer);
        raise_parse_error(contents_obj, pos, "number");
        Py_DECREF(contents_obj);
        return NULL;
    }
    
    /* Check if we found any numeric characters */
    if (pos == end) {
        PyObject *contents_obj = buffer.obj;
        Py_INCREF(contents_obj);
        PyBuffer_Release(&buffer);
        raise_parse_error(contents_obj, pos, "number");
        Py_DECREF(contents_obj);
        return NULL;
    }
    
    /* Extract the numeric string */
    PyObject *chars = PyBytes_FromStringAndSize((const char *)(contents + pos), end - pos);
    if (chars == NULL) {
        PyBuffer_Release(&buffer);
        return NULL;
    }
    
    PyObject *result_value = NULL;
    
    /* Try to parse as integer first if target allows */
    if (target_type != TARGET_FLOAT) {
        PyObject *int_result = PyLong_FromString(PyBytes_AS_STRING(chars), NULL, 10);
        if (int_result != NULL) {
            result_value = int_result;
        } else {
            PyErr_Clear();
            if (target_type == TARGET_INT) {
                /* Must be an integer, so this is an error */
                PyObject *contents_obj = buffer.obj;
                Py_INCREF(contents_obj);
                PyBuffer_Release(&buffer);
                Py_DECREF(chars);
                raise_parse_error(contents_obj, pos, "integer");
                Py_DECREF(contents_obj);
                return NULL;
            }
        }
    }
    
    /* Try to parse as float if we haven't got a result yet */
    if (result_value == NULL) {
        PyObject *float_result = PyFloat_FromString(chars);
        if (float_result != NULL) {
            result_value = float_result;
        } else {
            PyErr_Clear();
            PyObject *contents_obj = buffer.obj;
            Py_INCREF(contents_obj);
            PyBuffer_Release(&buffer);
            Py_DECREF(chars);
            const char *expected_msg = (target_type == TARGET_FLOAT) ? "float" : "number";
            raise_parse_error(contents_obj, pos, expected_msg);
            Py_DECREF(contents_obj);
            return NULL;
        }
    }
    
    Py_DECREF(chars);
    PyBuffer_Release(&buffer);
    
    /* Return tuple (value, end) */
    PyObject *result = PyTuple_Pack(2, result_value, PyLong_FromSsize_t(end));
    Py_DECREF(result_value);
    return result;
}

static PyMethodDef skip_methods[] = {
    {"_skip", (PyCFunction)skip, METH_VARARGS | METH_KEYWORDS,
     "Skip whitespace and comments in OpenFOAM file contents.\n\n"
     "Args:\n"
     "    contents: bytes or bytearray to parse\n"
     "    pos: current position in contents\n"
     "    newline_ok: if False, newlines are not skipped (default: True)\n\n"
     "Returns:\n"
     "    New position after skipping whitespace and comments\n"},
    {"_parse_number", (PyCFunction)parse_number, METH_VARARGS | METH_KEYWORDS,
     "Parse a number (integer or float) from OpenFOAM file contents.\n\n"
     "Args:\n"
     "    contents: bytes or bytearray to parse\n"
     "    pos: current position in contents\n"
     "    target: target type (int, float, or int|float)\n\n"
     "Returns:\n"
     "    Tuple of (number, new_position)\n"},
    {NULL, NULL, 0, NULL}
};

static int
skip_module_traverse(PyObject *m, visitproc visit, void *arg)
{
    Py_VISIT(FoamFileDecodeError);
    Py_VISIT(ParseError);
    return 0;
}

static int
skip_module_clear(PyObject *m)
{
    Py_CLEAR(FoamFileDecodeError);
    Py_CLEAR(ParseError);
    return 0;
}

static void
skip_module_free(void *m)
{
    skip_module_clear((PyObject *)m);
}

static struct PyModuleDef skip_module = {
    PyModuleDef_HEAD_INIT,
    "_skip_ext",
    "C extension module for fast whitespace and comment skipping in OpenFOAM parser",
    -1,
    skip_methods,
    NULL,
    skip_module_traverse,
    skip_module_clear,
    skip_module_free
};

PyMODINIT_FUNC
PyInit__skip_ext(void)
{
    PyObject *module;
    
    /* Initialize lookup tables */
    init_lookup_tables();
    
    module = PyModule_Create(&skip_module);
    if (module == NULL) {
        return NULL;
    }
    
    /* Import FoamFileDecodeError from the exceptions module */
    PyObject *exceptions_module = PyImport_ImportModule("foamlib._files._parsing.exceptions");
    if (exceptions_module == NULL) {
        Py_DECREF(module);
        return NULL;
    }
    
    FoamFileDecodeError = PyObject_GetAttrString(exceptions_module, "FoamFileDecodeError");
    Py_DECREF(exceptions_module);
    
    if (FoamFileDecodeError == NULL) {
        Py_DECREF(module);
        return NULL;
    }
    
    /* Create a simple ParseError exception class that can hold attributes.
       The Python code in _parser.py will define a proper ParseError class
       and _parse_number will be updated to import it from there. */
    ParseError = PyErr_NewException("_skip_ext.ParseError", PyExc_Exception, NULL);
    if (ParseError == NULL) {
        Py_DECREF(module);
        return NULL;
    }
    Py_INCREF(ParseError);
    if (PyModule_AddObject(module, "ParseError", ParseError) < 0) {
        Py_DECREF(ParseError);
        Py_DECREF(module);
        return NULL;
    }
    
    return module;
}
