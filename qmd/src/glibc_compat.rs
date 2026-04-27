//! glibc 2.38+ C23 symbol compatibility shim.
//!
//! The pyke prebuilt of `onnxruntime` shipped with `ort 2.0.0-rc.11` is
//! compiled against glibc ≥ 2.38 with C23 mode enabled, which redirects
//! `strtoll` / `strtol` / `strtoull` to versioned wrappers
//! `__isoc23_strtoll` / `__isoc23_strtol` / `__isoc23_strtoull`.
//!
//! On older glibc (e.g. Ubuntu 22.04, which ships glibc 2.35) those symbols
//! don't exist, and linking any binary that pulls in `qmd` fails with
//!
//! ```text
//! rust-lld: error: undefined symbol: __isoc23_strtoll
//! ```
//!
//! That includes `qmd-cli`, downstream consumers of this crate, **and**
//! qmd's own doctests/benches/integration tests, since they all link the
//! crate's transitive deps (fastembed → ort).
//!
//! The C23 variants behave identically to the un-prefixed functions except
//! they additionally accept `0b...` binary literals when `base` is 0 or 2.
//! onnxruntime never feeds binary literals into these calls (it only parses
//! JSON numbers and integer config values), so we can safely satisfy the
//! references by delegating to the plain libc functions.
//!
//! Gated behind the default-on `glibc-compat` feature and only emitted on
//! `target_os = "linux"` with the GNU libc environment. Downstream users
//! on glibc ≥ 2.38 can disable the feature with `default-features = false`
//! if they want the real glibc implementations to win.

#![cfg(all(feature = "glibc-compat", target_os = "linux", target_env = "gnu"))]
// The whole module is one big intentional pile of FFI: `unsafe extern` blocks,
// `#[no_mangle]` symbol exports, `unsafe fn` declarations, and `unsafe { … }`
// calls. These are exactly what the workspace lints (`unsafe_code`,
// `clippy::multiple_unsafe_ops_per_block`, `clippy::missing_docs_in_private_items`,
// etc.) flag — and they cannot be avoided here.
#![allow(
    unsafe_code,
    missing_docs,
    clippy::missing_docs_in_private_items,
    clippy::missing_safety_doc,
    clippy::multiple_unsafe_ops_per_block,
    clippy::undocumented_unsafe_blocks,
    clippy::unnecessary_safety_comment
)]

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
