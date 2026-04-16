//! Standard library for the Genovo scripting VM.
//!
//! Provides a comprehensive set of native functions organized into categories:
//! - **Math**: trigonometry, exponentials, rounding, constants
//! - **String**: manipulation, searching, formatting
//! - **Array**: manipulation, functional operations
//! - **Map**: key-value operations
//! - **Type**: type checking and conversion
//! - **IO**: sandboxed output functions
//! - **Functional**: higher-order function concepts

use std::collections::HashMap;
use std::sync::Arc;

use crate::vm::{ScriptError, ScriptValue};

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// A native function signature for the standard library.
pub type StdlibFn = Box<dyn Fn(&[ScriptValue]) -> Result<ScriptValue, ScriptError> + Send + Sync>;

/// Returns all standard library functions as (name, function) pairs.
pub fn get_stdlib() -> Vec<(String, StdlibFn)> {
    let mut fns: Vec<(String, StdlibFn)> = Vec::new();

    // -- Math --
    register_math(&mut fns);
    // -- String --
    register_string(&mut fns);
    // -- Array --
    register_array(&mut fns);
    // -- Map --
    register_map(&mut fns);
    // -- Type --
    register_type(&mut fns);
    // -- IO --
    register_io(&mut fns);
    // -- Functional --
    register_functional(&mut fns);

    fns
}

/// Returns the names of all standard library functions.
pub fn stdlib_function_names() -> Vec<&'static str> {
    vec![
        // Math
        "math_sin", "math_cos", "math_tan", "math_asin", "math_acos", "math_atan",
        "math_atan2", "math_exp", "math_log", "math_log2", "math_pow", "math_sqrt",
        "math_ceil", "math_floor", "math_round", "math_sign", "math_fract", "math_mod",
        "math_pi", "math_e", "math_inf", "math_lerp", "math_inverse_lerp", "math_remap",
        "math_deg_to_rad", "math_rad_to_deg",
        // String
        "str_len", "str_substr", "str_find", "str_replace", "str_split", "str_join",
        "str_trim", "str_upper", "str_lower", "str_starts_with", "str_ends_with",
        "str_contains", "str_repeat", "str_reverse", "str_char_at", "str_format",
        // Array
        "arr_len", "arr_push", "arr_pop", "arr_insert", "arr_remove", "arr_contains",
        "arr_find", "arr_sort", "arr_reverse", "arr_map", "arr_filter", "arr_reduce",
        "arr_any", "arr_all", "arr_flatten", "arr_zip", "arr_enumerate", "arr_slice",
        "arr_chunks",
        // Map
        "map_len", "map_has", "map_get", "map_set", "map_remove", "map_keys",
        "map_values", "map_entries", "map_merge",
        // Type
        "type_of", "is_nil", "is_number", "is_string", "is_array", "is_map",
        "to_int", "to_float", "to_string", "to_bool",
        // IO
        "print", "println", "format", "debug_print",
        // Functional
        "fn_compose", "fn_identity",
    ]
}

// ---------------------------------------------------------------------------
// Math functions
// ---------------------------------------------------------------------------

fn register_math(fns: &mut Vec<(String, StdlibFn)>) {
    fns.push(("math_sin".into(), Box::new(|args| {
        expect_args("math_sin", args, 1)?;
        let x = to_f64(&args[0], "math_sin")?;
        Ok(ScriptValue::Float(x.sin()))
    })));

    fns.push(("math_cos".into(), Box::new(|args| {
        expect_args("math_cos", args, 1)?;
        let x = to_f64(&args[0], "math_cos")?;
        Ok(ScriptValue::Float(x.cos()))
    })));

    fns.push(("math_tan".into(), Box::new(|args| {
        expect_args("math_tan", args, 1)?;
        let x = to_f64(&args[0], "math_tan")?;
        Ok(ScriptValue::Float(x.tan()))
    })));

    fns.push(("math_asin".into(), Box::new(|args| {
        expect_args("math_asin", args, 1)?;
        let x = to_f64(&args[0], "math_asin")?;
        if x < -1.0 || x > 1.0 {
            return Err(ScriptError::RuntimeError("asin: argument must be in [-1, 1]".into()));
        }
        Ok(ScriptValue::Float(x.asin()))
    })));

    fns.push(("math_acos".into(), Box::new(|args| {
        expect_args("math_acos", args, 1)?;
        let x = to_f64(&args[0], "math_acos")?;
        if x < -1.0 || x > 1.0 {
            return Err(ScriptError::RuntimeError("acos: argument must be in [-1, 1]".into()));
        }
        Ok(ScriptValue::Float(x.acos()))
    })));

    fns.push(("math_atan".into(), Box::new(|args| {
        expect_args("math_atan", args, 1)?;
        let x = to_f64(&args[0], "math_atan")?;
        Ok(ScriptValue::Float(x.atan()))
    })));

    fns.push(("math_atan2".into(), Box::new(|args| {
        expect_args("math_atan2", args, 2)?;
        let y = to_f64(&args[0], "math_atan2")?;
        let x = to_f64(&args[1], "math_atan2")?;
        Ok(ScriptValue::Float(y.atan2(x)))
    })));

    fns.push(("math_exp".into(), Box::new(|args| {
        expect_args("math_exp", args, 1)?;
        let x = to_f64(&args[0], "math_exp")?;
        Ok(ScriptValue::Float(x.exp()))
    })));

    fns.push(("math_log".into(), Box::new(|args| {
        expect_args("math_log", args, 1)?;
        let x = to_f64(&args[0], "math_log")?;
        if x <= 0.0 {
            return Err(ScriptError::RuntimeError("log: argument must be positive".into()));
        }
        Ok(ScriptValue::Float(x.ln()))
    })));

    fns.push(("math_log2".into(), Box::new(|args| {
        expect_args("math_log2", args, 1)?;
        let x = to_f64(&args[0], "math_log2")?;
        if x <= 0.0 {
            return Err(ScriptError::RuntimeError("log2: argument must be positive".into()));
        }
        Ok(ScriptValue::Float(x.log2()))
    })));

    fns.push(("math_pow".into(), Box::new(|args| {
        expect_args("math_pow", args, 2)?;
        let base = to_f64(&args[0], "math_pow")?;
        let exp = to_f64(&args[1], "math_pow")?;
        Ok(ScriptValue::Float(base.powf(exp)))
    })));

    fns.push(("math_sqrt".into(), Box::new(|args| {
        expect_args("math_sqrt", args, 1)?;
        let x = to_f64(&args[0], "math_sqrt")?;
        if x < 0.0 {
            return Err(ScriptError::RuntimeError("sqrt: argument must be non-negative".into()));
        }
        Ok(ScriptValue::Float(x.sqrt()))
    })));

    fns.push(("math_ceil".into(), Box::new(|args| {
        expect_args("math_ceil", args, 1)?;
        let x = to_f64(&args[0], "math_ceil")?;
        Ok(ScriptValue::Float(x.ceil()))
    })));

    fns.push(("math_floor".into(), Box::new(|args| {
        expect_args("math_floor", args, 1)?;
        let x = to_f64(&args[0], "math_floor")?;
        Ok(ScriptValue::Float(x.floor()))
    })));

    fns.push(("math_round".into(), Box::new(|args| {
        expect_args("math_round", args, 1)?;
        let x = to_f64(&args[0], "math_round")?;
        Ok(ScriptValue::Float(x.round()))
    })));

    fns.push(("math_sign".into(), Box::new(|args| {
        expect_args("math_sign", args, 1)?;
        let x = to_f64(&args[0], "math_sign")?;
        Ok(ScriptValue::Float(x.signum()))
    })));

    fns.push(("math_fract".into(), Box::new(|args| {
        expect_args("math_fract", args, 1)?;
        let x = to_f64(&args[0], "math_fract")?;
        Ok(ScriptValue::Float(x.fract()))
    })));

    fns.push(("math_mod".into(), Box::new(|args| {
        expect_args("math_mod", args, 2)?;
        let a = to_f64(&args[0], "math_mod")?;
        let b = to_f64(&args[1], "math_mod")?;
        if b.abs() < 1e-15 {
            return Err(ScriptError::RuntimeError("mod: division by zero".into()));
        }
        Ok(ScriptValue::Float(a % b))
    })));

    fns.push(("math_pi".into(), Box::new(|args| {
        expect_args("math_pi", args, 0)?;
        Ok(ScriptValue::Float(std::f64::consts::PI))
    })));

    fns.push(("math_e".into(), Box::new(|args| {
        expect_args("math_e", args, 0)?;
        Ok(ScriptValue::Float(std::f64::consts::E))
    })));

    fns.push(("math_inf".into(), Box::new(|args| {
        expect_args("math_inf", args, 0)?;
        Ok(ScriptValue::Float(f64::INFINITY))
    })));

    fns.push(("math_lerp".into(), Box::new(|args| {
        expect_args("math_lerp", args, 3)?;
        let a = to_f64(&args[0], "math_lerp")?;
        let b = to_f64(&args[1], "math_lerp")?;
        let t = to_f64(&args[2], "math_lerp")?;
        Ok(ScriptValue::Float(a + (b - a) * t))
    })));

    fns.push(("math_inverse_lerp".into(), Box::new(|args| {
        expect_args("math_inverse_lerp", args, 3)?;
        let a = to_f64(&args[0], "math_inverse_lerp")?;
        let b = to_f64(&args[1], "math_inverse_lerp")?;
        let v = to_f64(&args[2], "math_inverse_lerp")?;
        let range = b - a;
        if range.abs() < 1e-15 {
            return Ok(ScriptValue::Float(0.0));
        }
        Ok(ScriptValue::Float((v - a) / range))
    })));

    fns.push(("math_remap".into(), Box::new(|args| {
        expect_args("math_remap", args, 5)?;
        let value = to_f64(&args[0], "math_remap")?;
        let from_min = to_f64(&args[1], "math_remap")?;
        let from_max = to_f64(&args[2], "math_remap")?;
        let to_min = to_f64(&args[3], "math_remap")?;
        let to_max = to_f64(&args[4], "math_remap")?;
        let from_range = from_max - from_min;
        if from_range.abs() < 1e-15 {
            return Ok(ScriptValue::Float(to_min));
        }
        let t = (value - from_min) / from_range;
        Ok(ScriptValue::Float(to_min + (to_max - to_min) * t))
    })));

    fns.push(("math_deg_to_rad".into(), Box::new(|args| {
        expect_args("math_deg_to_rad", args, 1)?;
        let deg = to_f64(&args[0], "math_deg_to_rad")?;
        Ok(ScriptValue::Float(deg.to_radians()))
    })));

    fns.push(("math_rad_to_deg".into(), Box::new(|args| {
        expect_args("math_rad_to_deg", args, 1)?;
        let rad = to_f64(&args[0], "math_rad_to_deg")?;
        Ok(ScriptValue::Float(rad.to_degrees()))
    })));
}

// ---------------------------------------------------------------------------
// String functions
// ---------------------------------------------------------------------------

fn register_string(fns: &mut Vec<(String, StdlibFn)>) {
    fns.push(("str_len".into(), Box::new(|args| {
        expect_args("str_len", args, 1)?;
        let s = to_str(&args[0], "str_len")?;
        Ok(ScriptValue::Int(s.len() as i64))
    })));

    fns.push(("str_substr".into(), Box::new(|args| {
        if args.len() < 2 || args.len() > 3 {
            return Err(ScriptError::ArityMismatch { function: "str_substr".into(), expected: 2, got: args.len() as u8 });
        }
        let s = to_str(&args[0], "str_substr")?;
        let start = to_i64(&args[1], "str_substr")? as usize;
        let end = if args.len() == 3 {
            to_i64(&args[2], "str_substr")? as usize
        } else {
            s.len()
        };
        let start = start.min(s.len());
        let end = end.min(s.len());
        Ok(ScriptValue::from_string(&s[start..end]))
    })));

    fns.push(("str_find".into(), Box::new(|args| {
        expect_args("str_find", args, 2)?;
        let haystack = to_str(&args[0], "str_find")?;
        let needle = to_str(&args[1], "str_find")?;
        match haystack.find(&needle) {
            Some(idx) => Ok(ScriptValue::Int(idx as i64)),
            None => Ok(ScriptValue::Int(-1)),
        }
    })));

    fns.push(("str_replace".into(), Box::new(|args| {
        expect_args("str_replace", args, 3)?;
        let s = to_str(&args[0], "str_replace")?;
        let from = to_str(&args[1], "str_replace")?;
        let to = to_str(&args[2], "str_replace")?;
        Ok(ScriptValue::from_string(s.replace(&from, &to)))
    })));

    fns.push(("str_split".into(), Box::new(|args| {
        expect_args("str_split", args, 2)?;
        let s = to_str(&args[0], "str_split")?;
        let delim = to_str(&args[1], "str_split")?;
        let parts: Vec<ScriptValue> = s.split(&delim)
            .map(|p| ScriptValue::from_string(p))
            .collect();
        Ok(ScriptValue::Array(parts))
    })));

    fns.push(("str_join".into(), Box::new(|args| {
        expect_args("str_join", args, 2)?;
        let arr = to_array(&args[0], "str_join")?;
        let sep = to_str(&args[1], "str_join")?;
        let parts: Vec<String> = arr.iter().map(|v| format!("{v}")).collect();
        Ok(ScriptValue::from_string(parts.join(&sep)))
    })));

    fns.push(("str_trim".into(), Box::new(|args| {
        expect_args("str_trim", args, 1)?;
        let s = to_str(&args[0], "str_trim")?;
        Ok(ScriptValue::from_string(s.trim()))
    })));

    fns.push(("str_upper".into(), Box::new(|args| {
        expect_args("str_upper", args, 1)?;
        let s = to_str(&args[0], "str_upper")?;
        Ok(ScriptValue::from_string(s.to_uppercase()))
    })));

    fns.push(("str_lower".into(), Box::new(|args| {
        expect_args("str_lower", args, 1)?;
        let s = to_str(&args[0], "str_lower")?;
        Ok(ScriptValue::from_string(s.to_lowercase()))
    })));

    fns.push(("str_starts_with".into(), Box::new(|args| {
        expect_args("str_starts_with", args, 2)?;
        let s = to_str(&args[0], "str_starts_with")?;
        let prefix = to_str(&args[1], "str_starts_with")?;
        Ok(ScriptValue::Bool(s.starts_with(&prefix)))
    })));

    fns.push(("str_ends_with".into(), Box::new(|args| {
        expect_args("str_ends_with", args, 2)?;
        let s = to_str(&args[0], "str_ends_with")?;
        let suffix = to_str(&args[1], "str_ends_with")?;
        Ok(ScriptValue::Bool(s.ends_with(&suffix)))
    })));

    fns.push(("str_contains".into(), Box::new(|args| {
        expect_args("str_contains", args, 2)?;
        let s = to_str(&args[0], "str_contains")?;
        let sub = to_str(&args[1], "str_contains")?;
        Ok(ScriptValue::Bool(s.contains(&sub)))
    })));

    fns.push(("str_repeat".into(), Box::new(|args| {
        expect_args("str_repeat", args, 2)?;
        let s = to_str(&args[0], "str_repeat")?;
        let n = to_i64(&args[1], "str_repeat")? as usize;
        Ok(ScriptValue::from_string(s.repeat(n)))
    })));

    fns.push(("str_reverse".into(), Box::new(|args| {
        expect_args("str_reverse", args, 1)?;
        let s = to_str(&args[0], "str_reverse")?;
        Ok(ScriptValue::from_string(s.chars().rev().collect::<String>()))
    })));

    fns.push(("str_char_at".into(), Box::new(|args| {
        expect_args("str_char_at", args, 2)?;
        let s = to_str(&args[0], "str_char_at")?;
        let idx = to_i64(&args[1], "str_char_at")? as usize;
        match s.chars().nth(idx) {
            Some(c) => Ok(ScriptValue::from_string(c.to_string())),
            None => Ok(ScriptValue::Nil),
        }
    })));

    fns.push(("str_format".into(), Box::new(|args| {
        if args.is_empty() {
            return Err(ScriptError::ArityMismatch { function: "str_format".into(), expected: 1, got: 0 });
        }
        let template = to_str(&args[0], "str_format")?;
        let mut result = template.clone();
        for (i, arg) in args[1..].iter().enumerate() {
            let placeholder = format!("{{{}}}", i);
            result = result.replace(&placeholder, &format!("{arg}"));
        }
        Ok(ScriptValue::from_string(result))
    })));
}

// ---------------------------------------------------------------------------
// Array functions
// ---------------------------------------------------------------------------

fn register_array(fns: &mut Vec<(String, StdlibFn)>) {
    fns.push(("arr_len".into(), Box::new(|args| {
        expect_args("arr_len", args, 1)?;
        let arr = to_array(&args[0], "arr_len")?;
        Ok(ScriptValue::Int(arr.len() as i64))
    })));

    fns.push(("arr_push".into(), Box::new(|args| {
        expect_args("arr_push", args, 2)?;
        let mut arr = to_array(&args[0], "arr_push")?;
        arr.push(args[1].clone());
        Ok(ScriptValue::Array(arr))
    })));

    fns.push(("arr_pop".into(), Box::new(|args| {
        expect_args("arr_pop", args, 1)?;
        let mut arr = to_array(&args[0], "arr_pop")?;
        let popped = arr.pop().unwrap_or(ScriptValue::Nil);
        Ok(ScriptValue::Array(vec![
            ScriptValue::Array(arr),
            popped,
        ]))
    })));

    fns.push(("arr_insert".into(), Box::new(|args| {
        expect_args("arr_insert", args, 3)?;
        let mut arr = to_array(&args[0], "arr_insert")?;
        let idx = to_i64(&args[1], "arr_insert")? as usize;
        let idx = idx.min(arr.len());
        arr.insert(idx, args[2].clone());
        Ok(ScriptValue::Array(arr))
    })));

    fns.push(("arr_remove".into(), Box::new(|args| {
        expect_args("arr_remove", args, 2)?;
        let mut arr = to_array(&args[0], "arr_remove")?;
        let idx = to_i64(&args[1], "arr_remove")? as usize;
        if idx < arr.len() {
            let removed = arr.remove(idx);
            Ok(ScriptValue::Array(vec![
                ScriptValue::Array(arr),
                removed,
            ]))
        } else {
            Ok(ScriptValue::Array(vec![
                ScriptValue::Array(arr),
                ScriptValue::Nil,
            ]))
        }
    })));

    fns.push(("arr_contains".into(), Box::new(|args| {
        expect_args("arr_contains", args, 2)?;
        let arr = to_array(&args[0], "arr_contains")?;
        let target = &args[1];
        let found = arr.iter().any(|v| values_equal(v, target));
        Ok(ScriptValue::Bool(found))
    })));

    fns.push(("arr_find".into(), Box::new(|args| {
        expect_args("arr_find", args, 2)?;
        let arr = to_array(&args[0], "arr_find")?;
        let target = &args[1];
        for (i, v) in arr.iter().enumerate() {
            if values_equal(v, target) {
                return Ok(ScriptValue::Int(i as i64));
            }
        }
        Ok(ScriptValue::Int(-1))
    })));

    fns.push(("arr_sort".into(), Box::new(|args| {
        expect_args("arr_sort", args, 1)?;
        let mut arr = to_array(&args[0], "arr_sort")?;
        arr.sort_by(|a, b| compare_values(a, b));
        Ok(ScriptValue::Array(arr))
    })));

    fns.push(("arr_reverse".into(), Box::new(|args| {
        expect_args("arr_reverse", args, 1)?;
        let mut arr = to_array(&args[0], "arr_reverse")?;
        arr.reverse();
        Ok(ScriptValue::Array(arr))
    })));

    // arr_map, arr_filter, arr_reduce operate on arrays of values directly.
    // Since we don't have closures in the simple VM, these work with simple
    // arithmetic operations encoded as strings or use native callbacks.
    // For now, they provide structural operations.

    fns.push(("arr_map".into(), Box::new(|args| {
        // arr_map(array, "double") — applies named transformation.
        // Without first-class functions, we support basic string-named transforms.
        expect_args("arr_map", args, 2)?;
        let arr = to_array(&args[0], "arr_map")?;
        let op = to_str(&args[1], "arr_map")?;
        let result: Result<Vec<ScriptValue>, ScriptError> = arr.iter().map(|v| {
            apply_named_transform(v, &op)
        }).collect();
        Ok(ScriptValue::Array(result?))
    })));

    fns.push(("arr_filter".into(), Box::new(|args| {
        expect_args("arr_filter", args, 2)?;
        let arr = to_array(&args[0], "arr_filter")?;
        let op = to_str(&args[1], "arr_filter")?;
        let result: Vec<ScriptValue> = arr.into_iter().filter(|v| {
            apply_named_predicate(v, &op)
        }).collect();
        Ok(ScriptValue::Array(result))
    })));

    fns.push(("arr_reduce".into(), Box::new(|args| {
        expect_args("arr_reduce", args, 3)?;
        let arr = to_array(&args[0], "arr_reduce")?;
        let mut acc = args[1].clone();
        let op = to_str(&args[2], "arr_reduce")?;
        for v in &arr {
            acc = apply_named_binary_op(&acc, v, &op)?;
        }
        Ok(acc)
    })));

    fns.push(("arr_any".into(), Box::new(|args| {
        expect_args("arr_any", args, 2)?;
        let arr = to_array(&args[0], "arr_any")?;
        let op = to_str(&args[1], "arr_any")?;
        let result = arr.iter().any(|v| apply_named_predicate(v, &op));
        Ok(ScriptValue::Bool(result))
    })));

    fns.push(("arr_all".into(), Box::new(|args| {
        expect_args("arr_all", args, 2)?;
        let arr = to_array(&args[0], "arr_all")?;
        let op = to_str(&args[1], "arr_all")?;
        let result = arr.iter().all(|v| apply_named_predicate(v, &op));
        Ok(ScriptValue::Bool(result))
    })));

    fns.push(("arr_flatten".into(), Box::new(|args| {
        expect_args("arr_flatten", args, 1)?;
        let arr = to_array(&args[0], "arr_flatten")?;
        let mut result = Vec::new();
        for v in arr {
            match v {
                ScriptValue::Array(inner) => result.extend(inner),
                other => result.push(other),
            }
        }
        Ok(ScriptValue::Array(result))
    })));

    fns.push(("arr_zip".into(), Box::new(|args| {
        expect_args("arr_zip", args, 2)?;
        let a = to_array(&args[0], "arr_zip")?;
        let b = to_array(&args[1], "arr_zip")?;
        let len = a.len().min(b.len());
        let result: Vec<ScriptValue> = (0..len)
            .map(|i| ScriptValue::Array(vec![a[i].clone(), b[i].clone()]))
            .collect();
        Ok(ScriptValue::Array(result))
    })));

    fns.push(("arr_enumerate".into(), Box::new(|args| {
        expect_args("arr_enumerate", args, 1)?;
        let arr = to_array(&args[0], "arr_enumerate")?;
        let result: Vec<ScriptValue> = arr.into_iter().enumerate()
            .map(|(i, v)| ScriptValue::Array(vec![ScriptValue::Int(i as i64), v]))
            .collect();
        Ok(ScriptValue::Array(result))
    })));

    fns.push(("arr_slice".into(), Box::new(|args| {
        if args.len() < 2 || args.len() > 3 {
            return Err(ScriptError::ArityMismatch { function: "arr_slice".into(), expected: 2, got: args.len() as u8 });
        }
        let arr = to_array(&args[0], "arr_slice")?;
        let start = to_i64(&args[1], "arr_slice")? as usize;
        let end = if args.len() == 3 {
            to_i64(&args[2], "arr_slice")? as usize
        } else {
            arr.len()
        };
        let start = start.min(arr.len());
        let end = end.min(arr.len());
        Ok(ScriptValue::Array(arr[start..end].to_vec()))
    })));

    fns.push(("arr_chunks".into(), Box::new(|args| {
        expect_args("arr_chunks", args, 2)?;
        let arr = to_array(&args[0], "arr_chunks")?;
        let size = to_i64(&args[1], "arr_chunks")? as usize;
        if size == 0 {
            return Err(ScriptError::RuntimeError("arr_chunks: chunk size must be > 0".into()));
        }
        let result: Vec<ScriptValue> = arr.chunks(size)
            .map(|chunk| ScriptValue::Array(chunk.to_vec()))
            .collect();
        Ok(ScriptValue::Array(result))
    })));
}

// ---------------------------------------------------------------------------
// Map functions
// ---------------------------------------------------------------------------

fn register_map(fns: &mut Vec<(String, StdlibFn)>) {
    fns.push(("map_len".into(), Box::new(|args| {
        expect_args("map_len", args, 1)?;
        let map = to_map(&args[0], "map_len")?;
        Ok(ScriptValue::Int(map.len() as i64))
    })));

    fns.push(("map_has".into(), Box::new(|args| {
        expect_args("map_has", args, 2)?;
        let map = to_map(&args[0], "map_has")?;
        let key = to_str(&args[1], "map_has")?;
        Ok(ScriptValue::Bool(map.contains_key(&key)))
    })));

    fns.push(("map_get".into(), Box::new(|args| {
        if args.len() < 2 || args.len() > 3 {
            return Err(ScriptError::ArityMismatch { function: "map_get".into(), expected: 2, got: args.len() as u8 });
        }
        let map = to_map(&args[0], "map_get")?;
        let key = to_str(&args[1], "map_get")?;
        match map.get(&key) {
            Some(v) => Ok(v.clone()),
            None => {
                if args.len() == 3 {
                    Ok(args[2].clone())
                } else {
                    Ok(ScriptValue::Nil)
                }
            }
        }
    })));

    fns.push(("map_set".into(), Box::new(|args| {
        expect_args("map_set", args, 3)?;
        let mut map = to_map(&args[0], "map_set")?;
        let key = to_str(&args[1], "map_set")?;
        map.insert(key, args[2].clone());
        Ok(ScriptValue::Map(map))
    })));

    fns.push(("map_remove".into(), Box::new(|args| {
        expect_args("map_remove", args, 2)?;
        let mut map = to_map(&args[0], "map_remove")?;
        let key = to_str(&args[1], "map_remove")?;
        map.remove(&key);
        Ok(ScriptValue::Map(map))
    })));

    fns.push(("map_keys".into(), Box::new(|args| {
        expect_args("map_keys", args, 1)?;
        let map = to_map(&args[0], "map_keys")?;
        let keys: Vec<ScriptValue> = map.keys().map(|k| ScriptValue::from_string(k.as_str())).collect();
        Ok(ScriptValue::Array(keys))
    })));

    fns.push(("map_values".into(), Box::new(|args| {
        expect_args("map_values", args, 1)?;
        let map = to_map(&args[0], "map_values")?;
        let values: Vec<ScriptValue> = map.values().cloned().collect();
        Ok(ScriptValue::Array(values))
    })));

    fns.push(("map_entries".into(), Box::new(|args| {
        expect_args("map_entries", args, 1)?;
        let map = to_map(&args[0], "map_entries")?;
        let entries: Vec<ScriptValue> = map.iter()
            .map(|(k, v)| ScriptValue::Array(vec![ScriptValue::from_string(k.as_str()), v.clone()]))
            .collect();
        Ok(ScriptValue::Array(entries))
    })));

    fns.push(("map_merge".into(), Box::new(|args| {
        expect_args("map_merge", args, 2)?;
        let mut map1 = to_map(&args[0], "map_merge")?;
        let map2 = to_map(&args[1], "map_merge")?;
        for (k, v) in map2 {
            map1.insert(k, v);
        }
        Ok(ScriptValue::Map(map1))
    })));
}

// ---------------------------------------------------------------------------
// Type functions
// ---------------------------------------------------------------------------

fn register_type(fns: &mut Vec<(String, StdlibFn)>) {
    fns.push(("is_nil".into(), Box::new(|args| {
        expect_args("is_nil", args, 1)?;
        Ok(ScriptValue::Bool(args[0].is_nil()))
    })));

    fns.push(("is_number".into(), Box::new(|args| {
        expect_args("is_number", args, 1)?;
        Ok(ScriptValue::Bool(matches!(args[0], ScriptValue::Int(_) | ScriptValue::Float(_))))
    })));

    fns.push(("is_string".into(), Box::new(|args| {
        expect_args("is_string", args, 1)?;
        Ok(ScriptValue::Bool(matches!(args[0], ScriptValue::String(_))))
    })));

    fns.push(("is_array".into(), Box::new(|args| {
        expect_args("is_array", args, 1)?;
        Ok(ScriptValue::Bool(matches!(args[0], ScriptValue::Array(_))))
    })));

    fns.push(("is_map".into(), Box::new(|args| {
        expect_args("is_map", args, 1)?;
        Ok(ScriptValue::Bool(matches!(args[0], ScriptValue::Map(_))))
    })));

    fns.push(("to_bool".into(), Box::new(|args| {
        expect_args("to_bool", args, 1)?;
        Ok(ScriptValue::Bool(args[0].is_truthy()))
    })));
}

// ---------------------------------------------------------------------------
// IO functions
// ---------------------------------------------------------------------------

fn register_io(fns: &mut Vec<(String, StdlibFn)>) {
    fns.push(("println".into(), Box::new(|args| {
        let msg: Vec<String> = args.iter().map(|v| format!("{v}")).collect();
        log::info!("[script] {}", msg.join(" "));
        Ok(ScriptValue::Nil)
    })));

    fns.push(("format".into(), Box::new(|args| {
        if args.is_empty() {
            return Ok(ScriptValue::from_string(""));
        }
        let parts: Vec<String> = args.iter().map(|v| format!("{v}")).collect();
        Ok(ScriptValue::from_string(parts.join(" ")))
    })));

    fns.push(("debug_print".into(), Box::new(|args| {
        let msg: Vec<String> = args.iter().map(|v| format!("{v:?}")).collect();
        log::debug!("[script:debug] {}", msg.join(" "));
        Ok(ScriptValue::Nil)
    })));
}

// ---------------------------------------------------------------------------
// Functional helpers
// ---------------------------------------------------------------------------

fn register_functional(fns: &mut Vec<(String, StdlibFn)>) {
    fns.push(("fn_identity".into(), Box::new(|args| {
        expect_args("fn_identity", args, 1)?;
        Ok(args[0].clone())
    })));

    fns.push(("fn_compose".into(), Box::new(|args| {
        // fn_compose takes two strings representing operations and returns
        // a string representing their composition.
        expect_args("fn_compose", args, 2)?;
        let a = to_str(&args[0], "fn_compose")?;
        let b = to_str(&args[1], "fn_compose")?;
        Ok(ScriptValue::from_string(format!("{a}|{b}")))
    })));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn expect_args(name: &str, args: &[ScriptValue], expected: usize) -> Result<(), ScriptError> {
    if args.len() != expected {
        Err(ScriptError::ArityMismatch {
            function: name.into(),
            expected: expected as u8,
            got: args.len() as u8,
        })
    } else {
        Ok(())
    }
}

fn to_f64(val: &ScriptValue, func: &str) -> Result<f64, ScriptError> {
    val.as_float().ok_or_else(|| {
        ScriptError::TypeError(format!("{func}: expected number, got {}", val.type_name()))
    })
}

fn to_i64(val: &ScriptValue, func: &str) -> Result<i64, ScriptError> {
    match val {
        ScriptValue::Int(i) => Ok(*i),
        ScriptValue::Float(f) => Ok(*f as i64),
        _ => Err(ScriptError::TypeError(format!(
            "{func}: expected integer, got {}",
            val.type_name()
        ))),
    }
}

fn to_str(val: &ScriptValue, func: &str) -> Result<String, ScriptError> {
    match val {
        ScriptValue::String(s) => Ok(s.to_string()),
        _ => Err(ScriptError::TypeError(format!(
            "{func}: expected string, got {}",
            val.type_name()
        ))),
    }
}

fn to_array(val: &ScriptValue, func: &str) -> Result<Vec<ScriptValue>, ScriptError> {
    match val {
        ScriptValue::Array(a) => Ok(a.clone()),
        _ => Err(ScriptError::TypeError(format!(
            "{func}: expected array, got {}",
            val.type_name()
        ))),
    }
}

fn to_map(val: &ScriptValue, func: &str) -> Result<HashMap<String, ScriptValue>, ScriptError> {
    match val {
        ScriptValue::Map(m) => Ok(m.clone()),
        _ => Err(ScriptError::TypeError(format!(
            "{func}: expected map, got {}",
            val.type_name()
        ))),
    }
}

fn values_equal(a: &ScriptValue, b: &ScriptValue) -> bool {
    match (a, b) {
        (ScriptValue::Nil, ScriptValue::Nil) => true,
        (ScriptValue::Bool(a), ScriptValue::Bool(b)) => a == b,
        (ScriptValue::Int(a), ScriptValue::Int(b)) => a == b,
        (ScriptValue::Float(a), ScriptValue::Float(b)) => (a - b).abs() < 1e-10,
        (ScriptValue::Int(a), ScriptValue::Float(b)) => (*a as f64 - b).abs() < 1e-10,
        (ScriptValue::Float(a), ScriptValue::Int(b)) => (a - *b as f64).abs() < 1e-10,
        (ScriptValue::String(a), ScriptValue::String(b)) => a == b,
        _ => false,
    }
}

fn compare_values(a: &ScriptValue, b: &ScriptValue) -> std::cmp::Ordering {
    match (a, b) {
        (ScriptValue::Int(a), ScriptValue::Int(b)) => a.cmp(b),
        (ScriptValue::Float(a), ScriptValue::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
        (ScriptValue::Int(a), ScriptValue::Float(b)) => (*a as f64).partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
        (ScriptValue::Float(a), ScriptValue::Int(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(std::cmp::Ordering::Equal),
        (ScriptValue::String(a), ScriptValue::String(b)) => a.cmp(b),
        _ => std::cmp::Ordering::Equal,
    }
}

/// Apply a named transformation to a value.
fn apply_named_transform(val: &ScriptValue, op: &str) -> Result<ScriptValue, ScriptError> {
    match op {
        "double" => match val {
            ScriptValue::Int(i) => Ok(ScriptValue::Int(i * 2)),
            ScriptValue::Float(f) => Ok(ScriptValue::Float(f * 2.0)),
            _ => Ok(val.clone()),
        },
        "negate" => match val {
            ScriptValue::Int(i) => Ok(ScriptValue::Int(-i)),
            ScriptValue::Float(f) => Ok(ScriptValue::Float(-f)),
            _ => Ok(val.clone()),
        },
        "abs" => match val {
            ScriptValue::Int(i) => Ok(ScriptValue::Int(i.abs())),
            ScriptValue::Float(f) => Ok(ScriptValue::Float(f.abs())),
            _ => Ok(val.clone()),
        },
        "to_string" => Ok(ScriptValue::from_string(format!("{val}"))),
        "to_int" => match val {
            ScriptValue::Float(f) => Ok(ScriptValue::Int(*f as i64)),
            ScriptValue::Int(i) => Ok(ScriptValue::Int(*i)),
            _ => Ok(val.clone()),
        },
        "to_float" => match val {
            ScriptValue::Int(i) => Ok(ScriptValue::Float(*i as f64)),
            ScriptValue::Float(f) => Ok(ScriptValue::Float(*f)),
            _ => Ok(val.clone()),
        },
        "square" => match val {
            ScriptValue::Int(i) => Ok(ScriptValue::Int(i * i)),
            ScriptValue::Float(f) => Ok(ScriptValue::Float(f * f)),
            _ => Ok(val.clone()),
        },
        _ => Err(ScriptError::RuntimeError(format!("Unknown transform: '{op}'"))),
    }
}

/// Apply a named predicate to check a condition.
fn apply_named_predicate(val: &ScriptValue, op: &str) -> bool {
    match op {
        "is_positive" => match val {
            ScriptValue::Int(i) => *i > 0,
            ScriptValue::Float(f) => *f > 0.0,
            _ => false,
        },
        "is_negative" => match val {
            ScriptValue::Int(i) => *i < 0,
            ScriptValue::Float(f) => *f < 0.0,
            _ => false,
        },
        "is_zero" => match val {
            ScriptValue::Int(i) => *i == 0,
            ScriptValue::Float(f) => f.abs() < 1e-10,
            _ => false,
        },
        "is_even" => match val {
            ScriptValue::Int(i) => i % 2 == 0,
            _ => false,
        },
        "is_odd" => match val {
            ScriptValue::Int(i) => i % 2 != 0,
            _ => false,
        },
        "is_truthy" => val.is_truthy(),
        "not_nil" => !val.is_nil(),
        _ => true,
    }
}

/// Apply a named binary operation (for reduce).
fn apply_named_binary_op(
    a: &ScriptValue,
    b: &ScriptValue,
    op: &str,
) -> Result<ScriptValue, ScriptError> {
    match op {
        "add" | "sum" => a.add(b),
        "mul" | "product" => a.mul(b),
        "min" => {
            if compare_values(a, b) == std::cmp::Ordering::Less {
                Ok(a.clone())
            } else {
                Ok(b.clone())
            }
        }
        "max" => {
            if compare_values(a, b) == std::cmp::Ordering::Greater {
                Ok(a.clone())
            } else {
                Ok(b.clone())
            }
        }
        _ => Err(ScriptError::RuntimeError(format!("Unknown reduce operation: '{op}'"))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdlib_registration() {
        let stdlib = get_stdlib();
        assert!(stdlib.len() > 50);
    }

    #[test]
    fn test_math_sin_cos() {
        let stdlib = get_stdlib();
        let sin = stdlib.iter().find(|(n, _)| n == "math_sin").unwrap();
        let result = (sin.1)(&[ScriptValue::Float(0.0)]).unwrap();
        assert_eq!(result, ScriptValue::Float(0.0));

        let cos = stdlib.iter().find(|(n, _)| n == "math_cos").unwrap();
        let result = (cos.1)(&[ScriptValue::Float(0.0)]).unwrap();
        assert_eq!(result, ScriptValue::Float(1.0));
    }

    #[test]
    fn test_math_pow() {
        let stdlib = get_stdlib();
        let pow = stdlib.iter().find(|(n, _)| n == "math_pow").unwrap();
        let result = (pow.1)(&[ScriptValue::Float(2.0), ScriptValue::Float(3.0)]).unwrap();
        assert_eq!(result, ScriptValue::Float(8.0));
    }

    #[test]
    fn test_math_lerp() {
        let stdlib = get_stdlib();
        let lerp = stdlib.iter().find(|(n, _)| n == "math_lerp").unwrap();
        let result = (lerp.1)(&[
            ScriptValue::Float(0.0),
            ScriptValue::Float(10.0),
            ScriptValue::Float(0.5),
        ]).unwrap();
        assert_eq!(result, ScriptValue::Float(5.0));
    }

    #[test]
    fn test_math_constants() {
        let stdlib = get_stdlib();
        let pi = stdlib.iter().find(|(n, _)| n == "math_pi").unwrap();
        let result = (pi.1)(&[]).unwrap();
        assert!(matches!(result, ScriptValue::Float(v) if (v - std::f64::consts::PI).abs() < 1e-10));
    }

    #[test]
    fn test_str_operations() {
        let stdlib = get_stdlib();

        let upper = stdlib.iter().find(|(n, _)| n == "str_upper").unwrap();
        let result = (upper.1)(&[ScriptValue::from_string("hello")]).unwrap();
        assert_eq!(result, ScriptValue::from_string("HELLO"));

        let contains = stdlib.iter().find(|(n, _)| n == "str_contains").unwrap();
        let result = (contains.1)(&[
            ScriptValue::from_string("hello world"),
            ScriptValue::from_string("world"),
        ]).unwrap();
        assert_eq!(result, ScriptValue::Bool(true));
    }

    #[test]
    fn test_str_split_join() {
        let stdlib = get_stdlib();

        let split = stdlib.iter().find(|(n, _)| n == "str_split").unwrap();
        let result = (split.1)(&[
            ScriptValue::from_string("a,b,c"),
            ScriptValue::from_string(","),
        ]).unwrap();

        if let ScriptValue::Array(arr) = result {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], ScriptValue::from_string("a"));
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn test_arr_operations() {
        let stdlib = get_stdlib();

        let sort = stdlib.iter().find(|(n, _)| n == "arr_sort").unwrap();
        let result = (sort.1)(&[ScriptValue::Array(vec![
            ScriptValue::Int(3),
            ScriptValue::Int(1),
            ScriptValue::Int(2),
        ])]).unwrap();

        if let ScriptValue::Array(arr) = result {
            assert_eq!(arr[0], ScriptValue::Int(1));
            assert_eq!(arr[1], ScriptValue::Int(2));
            assert_eq!(arr[2], ScriptValue::Int(3));
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn test_arr_filter() {
        let stdlib = get_stdlib();
        let filter = stdlib.iter().find(|(n, _)| n == "arr_filter").unwrap();
        let result = (filter.1)(&[
            ScriptValue::Array(vec![
                ScriptValue::Int(-2),
                ScriptValue::Int(3),
                ScriptValue::Int(-1),
                ScriptValue::Int(5),
            ]),
            ScriptValue::from_string("is_positive"),
        ]).unwrap();

        if let ScriptValue::Array(arr) = result {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], ScriptValue::Int(3));
            assert_eq!(arr[1], ScriptValue::Int(5));
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn test_arr_reduce_sum() {
        let stdlib = get_stdlib();
        let reduce = stdlib.iter().find(|(n, _)| n == "arr_reduce").unwrap();
        let result = (reduce.1)(&[
            ScriptValue::Array(vec![
                ScriptValue::Int(1),
                ScriptValue::Int(2),
                ScriptValue::Int(3),
            ]),
            ScriptValue::Int(0),
            ScriptValue::from_string("sum"),
        ]).unwrap();
        assert_eq!(result, ScriptValue::Int(6));
    }

    #[test]
    fn test_arr_map_double() {
        let stdlib = get_stdlib();
        let map_fn = stdlib.iter().find(|(n, _)| n == "arr_map").unwrap();
        let result = (map_fn.1)(&[
            ScriptValue::Array(vec![
                ScriptValue::Int(1),
                ScriptValue::Int(2),
                ScriptValue::Int(3),
            ]),
            ScriptValue::from_string("double"),
        ]).unwrap();

        if let ScriptValue::Array(arr) = result {
            assert_eq!(arr[0], ScriptValue::Int(2));
            assert_eq!(arr[1], ScriptValue::Int(4));
            assert_eq!(arr[2], ScriptValue::Int(6));
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn test_map_operations() {
        let stdlib = get_stdlib();

        let set = stdlib.iter().find(|(n, _)| n == "map_set").unwrap();
        let map = ScriptValue::Map(HashMap::new());
        let result = (set.1)(&[
            map,
            ScriptValue::from_string("key"),
            ScriptValue::Int(42),
        ]).unwrap();

        let has = stdlib.iter().find(|(n, _)| n == "map_has").unwrap();
        let result2 = (has.1)(&[result.clone(), ScriptValue::from_string("key")]).unwrap();
        assert_eq!(result2, ScriptValue::Bool(true));

        let get = stdlib.iter().find(|(n, _)| n == "map_get").unwrap();
        let result3 = (get.1)(&[result, ScriptValue::from_string("key")]).unwrap();
        assert_eq!(result3, ScriptValue::Int(42));
    }

    #[test]
    fn test_type_checks() {
        let stdlib = get_stdlib();

        let is_nil = stdlib.iter().find(|(n, _)| n == "is_nil").unwrap();
        assert_eq!((is_nil.1)(&[ScriptValue::Nil]).unwrap(), ScriptValue::Bool(true));
        assert_eq!((is_nil.1)(&[ScriptValue::Int(1)]).unwrap(), ScriptValue::Bool(false));

        let is_number = stdlib.iter().find(|(n, _)| n == "is_number").unwrap();
        assert_eq!((is_number.1)(&[ScriptValue::Int(1)]).unwrap(), ScriptValue::Bool(true));
        assert_eq!((is_number.1)(&[ScriptValue::Float(1.0)]).unwrap(), ScriptValue::Bool(true));
        assert_eq!((is_number.1)(&[ScriptValue::Nil]).unwrap(), ScriptValue::Bool(false));
    }

    #[test]
    fn test_arr_flatten() {
        let stdlib = get_stdlib();
        let flatten = stdlib.iter().find(|(n, _)| n == "arr_flatten").unwrap();
        let result = (flatten.1)(&[ScriptValue::Array(vec![
            ScriptValue::Array(vec![ScriptValue::Int(1), ScriptValue::Int(2)]),
            ScriptValue::Int(3),
            ScriptValue::Array(vec![ScriptValue::Int(4)]),
        ])]).unwrap();

        if let ScriptValue::Array(arr) = result {
            assert_eq!(arr.len(), 4);
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn test_arr_zip() {
        let stdlib = get_stdlib();
        let zip = stdlib.iter().find(|(n, _)| n == "arr_zip").unwrap();
        let result = (zip.1)(&[
            ScriptValue::Array(vec![ScriptValue::Int(1), ScriptValue::Int(2)]),
            ScriptValue::Array(vec![ScriptValue::from_string("a"), ScriptValue::from_string("b")]),
        ]).unwrap();

        if let ScriptValue::Array(arr) = result {
            assert_eq!(arr.len(), 2);
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn test_str_format() {
        let stdlib = get_stdlib();
        let format_fn = stdlib.iter().find(|(n, _)| n == "str_format").unwrap();
        let result = (format_fn.1)(&[
            ScriptValue::from_string("Hello {0}, you are {1}!"),
            ScriptValue::from_string("World"),
            ScriptValue::Int(42),
        ]).unwrap();

        if let ScriptValue::String(s) = result {
            assert!(s.contains("Hello World"));
            assert!(s.contains("you are 42"));
        } else {
            panic!("expected string");
        }
    }

    #[test]
    fn test_math_error_handling() {
        let stdlib = get_stdlib();
        let asin = stdlib.iter().find(|(n, _)| n == "math_asin").unwrap();
        assert!((asin.1)(&[ScriptValue::Float(2.0)]).is_err());

        let log = stdlib.iter().find(|(n, _)| n == "math_log").unwrap();
        assert!((log.1)(&[ScriptValue::Float(-1.0)]).is_err());
    }
}
