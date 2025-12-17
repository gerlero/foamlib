use pyo3::prelude::*;

// Whitespace lookup table
static IS_WHITESPACE: [bool; 256] = {
    let mut table = [false; 256];
    table[b' ' as usize] = true;
    table[b'\n' as usize] = true;
    table[b'\t' as usize] = true;
    table[b'\r' as usize] = true;
    table[b'\x0C' as usize] = true; // \f
    table[b'\x0B' as usize] = true; // \v
    table
};

// Whitespace without newline lookup table
static IS_WHITESPACE_NO_NEWLINE: [bool; 256] = {
    let mut table = [false; 256];
    table[b' ' as usize] = true;
    table[b'\t' as usize] = true;
    table[b'\r' as usize] = true;
    table[b'\x0C' as usize] = true; // \f
    table[b'\x0B' as usize] = true; // \v
    table
};

/// Skip whitespace and comments in OpenFOAM files
/// 
/// This function skips over whitespace and comments (both // and /* */ style).
/// It handles line continuations with backslash in line comments.
/// 
/// # Arguments
/// * `contents` - The file contents as bytes or bytearray
/// * `pos` - Current position in the file
/// * `newline_ok` - Whether newlines should be skipped (default: True)
/// 
/// # Returns
/// The new position after skipping whitespace and comments
#[pyfunction]
#[pyo3(signature = (contents, pos, *, newline_ok=true))]
fn skip(contents: &Bound<'_, pyo3::types::PyAny>, mut pos: usize, newline_ok: bool) -> PyResult<usize> {
    // Extract bytes from either bytes or bytearray
    let contents = if let Ok(bytes) = contents.downcast::<pyo3::types::PyBytes>() {
        bytes.as_bytes()
    } else if let Ok(bytearray) = contents.downcast::<pyo3::types::PyByteArray>() {
        unsafe { bytearray.as_bytes() }
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "contents must be bytes or bytearray"
        ));
    };
    let is_whitespace = if newline_ok {
        &IS_WHITESPACE
    } else {
        &IS_WHITESPACE_NO_NEWLINE
    };
    
    loop {
        // Skip whitespace
        while pos < contents.len() && is_whitespace[contents[pos] as usize] {
            pos += 1;
        }
        
        // Check if we're at the end of content
        if pos >= contents.len() {
            break;
        }
        
        // Check for comments
        if pos + 1 < contents.len() {
            let next1 = contents[pos];
            let next2 = contents[pos + 1];
            
            // Single-line comment //
            if next1 == b'/' && next2 == b'/' {
                pos += 2;
                loop {
                    if pos >= contents.len() {
                        break;
                    }
                    
                    if contents[pos] == b'\n' {
                        if newline_ok {
                            pos += 1;
                        }
                        break;
                    }
                    
                    // Handle line continuation
                    if contents[pos] == b'\\' && pos + 1 < contents.len() && contents[pos + 1] == b'\n' {
                        pos += 1;
                    }
                    
                    pos += 1;
                }
                continue;
            }
            
            // Multi-line comment /* */
            if next1 == b'/' && next2 == b'*' {
                pos += 2;
                
                // Find the closing */
                let mut found = false;
                while pos + 1 < contents.len() {
                    if contents[pos] == b'*' && contents[pos + 1] == b'/' {
                        pos += 2;
                        found = true;
                        break;
                    }
                    pos += 1;
                }
                
                if !found {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Unterminated comment at position {}", pos)
                    ));
                }
                
                continue;
            }
        }
        
        // No more whitespace or comments
        break;
    }
    
    Ok(pos)
}

/// A Python module implemented in Rust for performance-critical parsing operations.
#[pymodule]
fn foamlib_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(skip, m)?)?;
    Ok(())
}
