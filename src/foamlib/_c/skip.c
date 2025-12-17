#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* Whitespace lookup tables */
static int is_whitespace[256] = {0};
static int is_whitespace_no_newline[256] = {0};

/* Initialize whitespace lookup tables */
static void init_whitespace_tables(void) {
    static int initialized = 0;
    if (initialized) return;
    
    /* Whitespace characters: space, newline, tab, carriage return, form feed, vertical tab */
    const char *ws = " \n\t\r\f\v";
    for (const char *c = ws; *c; c++) {
        is_whitespace[(unsigned char)*c] = 1;
        is_whitespace_no_newline[(unsigned char)*c] = 1;
    }
    is_whitespace_no_newline['\n'] = 0;
    
    initialized = 1;
}

/*
 * Skip whitespace and comments in OpenFOAM file content.
 * Returns the new position after skipping whitespace and comments.
 * 
 * Arguments:
 *   contents: bytes or bytearray object
 *   pos: starting position (int)
 *   newline_ok: whether newlines are considered whitespace (bool, default True)
 * 
 * Returns:
 *   New position (int)
 */
static PyObject *
skip(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *contents_obj;
    Py_ssize_t pos;
    int newline_ok = 1;
    
    static char *kwlist[] = {"contents", "pos", "newline_ok", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "On|p", kwlist,
                                     &contents_obj, &pos, &newline_ok)) {
        return NULL;
    }
    
    /* Initialize whitespace tables */
    init_whitespace_tables();
    
    /* Get buffer from bytes or bytearray */
    Py_buffer buffer;
    if (PyObject_GetBuffer(contents_obj, &buffer, PyBUF_SIMPLE) < 0) {
        return NULL;
    }
    
    const unsigned char *contents = (const unsigned char *)buffer.buf;
    Py_ssize_t len = buffer.len;
    
    /* Choose whitespace table based on newline_ok */
    const int *ws_table = newline_ok ? is_whitespace : is_whitespace_no_newline;
    
    /* Skip whitespace and comments */
    while (pos < len) {
        /* Skip whitespace */
        while (pos < len && ws_table[contents[pos]]) {
            pos++;
        }
        
        /* Check for comments */
        if (pos + 1 >= len) {
            break;
        }
        
        unsigned char next1 = contents[pos];
        unsigned char next2 = contents[pos + 1];
        
        /* Handle // comments */
        if (next1 == '/' && next2 == '/') {
            pos += 2;
            while (pos < len) {
                if (contents[pos] == '\n') {
                    if (newline_ok) {
                        pos++;
                    }
                    break;
                }
                if (contents[pos] == '\\' && pos + 1 < len && contents[pos + 1] == '\n') {
                    pos++;
                }
                pos++;
            }
            continue;
        }
        
        /* Handle block comments (slash-star ... star-slash) */
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
                /* Raise FoamFileDecodeError */
                PyBuffer_Release(&buffer);
                
                /* Import the exception class */
                PyObject *exceptions_module = PyImport_ImportModule("foamlib._files._parsing.exceptions");
                if (!exceptions_module) {
                    return NULL;
                }
                
                PyObject *FoamFileDecodeError_class = PyObject_GetAttrString(exceptions_module, "FoamFileDecodeError");
                Py_DECREF(exceptions_module);
                if (!FoamFileDecodeError_class) {
                    return NULL;
                }
                
                /* Create exception with keyword argument: FoamFileDecodeError(contents, len, expected="star-slash") */
                PyObject *kwargs_exc = Py_BuildValue("{s:s}", "expected", "*/");
                PyObject *args_exc = Py_BuildValue("(On)", contents_obj, len);
                
                if (!kwargs_exc || !args_exc) {
                    Py_XDECREF(kwargs_exc);
                    Py_XDECREF(args_exc);
                    Py_DECREF(FoamFileDecodeError_class);
                    return NULL;
                }
                
                PyObject *exc = PyObject_Call(FoamFileDecodeError_class, args_exc, kwargs_exc);
                Py_DECREF(kwargs_exc);
                Py_DECREF(args_exc);
                
                if (exc) {
                    PyErr_SetObject(FoamFileDecodeError_class, exc);
                    Py_DECREF(exc);
                }
                Py_DECREF(FoamFileDecodeError_class);
                return NULL;
            }
            continue;
        }
        
        /* No more whitespace or comments */
        break;
    }
    
    PyBuffer_Release(&buffer);
    return PyLong_FromSsize_t(pos);
}

/* Module method definitions */
static PyMethodDef skip_methods[] = {
    {"skip", (PyCFunction)skip, METH_VARARGS | METH_KEYWORDS,
     "Skip whitespace and comments in OpenFOAM file content.\n\n"
     "Args:\n"
     "    contents: bytes or bytearray object\n"
     "    pos: starting position (int)\n"
     "    newline_ok: whether newlines are considered whitespace (bool, default True)\n\n"
     "Returns:\n"
     "    New position (int)"},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef skip_module = {
    PyModuleDef_HEAD_INIT,
    "_skip",
    "C extension module for skipping whitespace and comments in OpenFOAM files",
    -1,
    skip_methods
};

/* Module initialization */
PyMODINIT_FUNC
PyInit__skip(void)
{
    return PyModule_Create(&skip_module);
}
