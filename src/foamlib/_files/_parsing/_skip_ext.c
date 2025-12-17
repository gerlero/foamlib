#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *FoamFileDecodeError = NULL;

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

static PyMethodDef skip_methods[] = {
    {"_skip", (PyCFunction)skip, METH_VARARGS | METH_KEYWORDS,
     "Skip whitespace and comments in OpenFOAM file contents.\n\n"
     "Args:\n"
     "    contents: bytes or bytearray to parse\n"
     "    pos: current position in contents\n"
     "    newline_ok: if False, newlines are not skipped (default: True)\n\n"
     "Returns:\n"
     "    New position after skipping whitespace and comments\n"},
    {NULL, NULL, 0, NULL}
};

static int
skip_module_traverse(PyObject *m, visitproc visit, void *arg)
{
    Py_VISIT(FoamFileDecodeError);
    return 0;
}

static int
skip_module_clear(PyObject *m)
{
    Py_CLEAR(FoamFileDecodeError);
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
    
    return module;
}
