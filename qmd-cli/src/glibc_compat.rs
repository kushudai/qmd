//! glibc 2.38+ C23 symbol compatibility shim.
//!
//! The pyke prebuilt of `onnxruntime` shipped with `ort 2.0.0-rc.11` is
//! compiled against glibc ≥ 2.38 with C23 mode enabled, which redirects
//! `strtoll` / `strtol` / `strtoull` to versioned wrappers
//! `__isoc23_strtoll` / `__isoc23_strtol` / `__isoc23_strtoull`.
//!
//! On older glibc (e.g. Ubuntu 22.04, which ships glibc 2.35) those symbols
//! don't exist, and linking the `qmd` binary fails with
//!
//! ```text
//! rust-lld: error: undefined symbol: __isoc23_strtoll
//! ```
//!
//! The C23 variants behave identically to the un-prefixed functions except
//! they additionally accept `0b...` binary literals when `base` is 0 or 2.
//! onnxruntime never feeds binary literals into these calls (it only parses
//! JSON numbers and integer config values), so we can safely satisfy the
//! references by delegating to the plain libc functions.
//!
//! These shims are only emitted on `target_os = "linux"` with the GNU libc
//! environment. Other targets (musl, macOS, Windows) don't need them and
//! the module compiles to nothing.

#![cfg(all(target_os = "linux", target_env = "gnu"))]

use std::os::raw::{c_char, c_int, c_long, c_longlong, c_ulonglong};

unsafe extern "C" {
    fn strtoll(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> c_longlong;
    fn strtol(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> c_long;
    fn strtoull(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> c_ulonglong;
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __isoc23_strtoll(
    nptr: *const c_char,
    endptr: *mut *mut c_char,
    base: c_int,
) -> c_longlong {
    unsafe { strtoll(nptr, endptr, base) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __isoc23_strtol(
    nptr: *const c_char,
    endptr: *mut *mut c_char,
    base: c_int,
) -> c_long {
    unsafe { strtol(nptr, endptr, base) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __isoc23_strtoull(
    nptr: *const c_char,
    endptr: *mut *mut c_char,
    base: c_int,
) -> c_ulonglong {
    unsafe { strtoull(nptr, endptr, base) }
}
