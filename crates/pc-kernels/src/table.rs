//! Instantiation lookup: one `KernelTable` per (word, ModMul) built into the archive (Cargo features), resolved once
//! (the caller keeps the `&'static` reference; there is no global cache) and checked against this crate's ABI.
use crate::args::{KernelTable, ABI, LOG_N};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Instance {
    U32Barrett64,
    U32Barrett32,
    U32Mont,
    U64Barrett64,
    U16Barrett32,
}

#[derive(Debug, thiserror::Error)]
pub enum TableError {
    #[error("kernel instantiation {0} is not built (enable Cargo feature `{1}` of pc-kernels)")]
    NotBuilt(&'static str, &'static str),
    #[error("kernel table {name}: {field} = {got}, expected {want}")]
    Mismatch { name: &'static str, field: &'static str, got: u32, want: u32 },
}

extern "C" {
    #[cfg(feature = "w32")]
    fn pc_table_u32_barrett64() -> *const KernelTable;
    #[cfg(feature = "w32")]
    fn pc_table_u32_barrett32() -> *const KernelTable;
    #[cfg(feature = "mont")]
    fn pc_table_u32_mont() -> *const KernelTable;
    #[cfg(feature = "w64")]
    fn pc_table_u64_barrett64() -> *const KernelTable;
    #[cfg(feature = "w16")]
    fn pc_table_u16_barrett32() -> *const KernelTable;
}

impl Instance {
    pub const ALL: [Instance; 5] =
        [Instance::U32Barrett64, Instance::U32Barrett32, Instance::U32Mont, Instance::U64Barrett64, Instance::U16Barrett32];

    pub fn name(self) -> &'static str {
        match self {
            Instance::U32Barrett64 => "u32xBarrett64",
            Instance::U32Barrett32 => "u32xBarrett32",
            Instance::U32Mont => "u32xMontgomery",
            Instance::U64Barrett64 => "u64xBarrett64",
            Instance::U16Barrett32 => "u16xBarrett32",
        }
    }
    pub fn feature(self) -> &'static str {
        match self {
            Instance::U32Barrett64 | Instance::U32Barrett32 => "w32",
            Instance::U32Mont => "mont",
            Instance::U64Barrett64 => "w64",
            Instance::U16Barrett32 => "w16",
        }
    }
    pub fn word_bytes(self) -> u32 {
        match self {
            Instance::U64Barrett64 => 8,
            Instance::U16Barrett32 => 2,
            _ => 4,
        }
    }
    /// ModMul kind as in KernelTable::mm (0 Barrett64, 1 Barrett32, 2 Montgomery)
    pub fn mm(self) -> u32 {
        match self {
            Instance::U32Barrett64 | Instance::U64Barrett64 => 0,
            Instance::U32Barrett32 | Instance::U16Barrett32 => 1,
            Instance::U32Mont => 2,
        }
    }
    /// The instantiations compiled into this build.
    pub fn built() -> Vec<Instance> {
        Instance::ALL.into_iter().filter(|i| table(*i).is_ok()).collect()
    }
    pub fn from_name(s: &str) -> Option<Instance> {
        Instance::ALL.into_iter().find(|i| i.name().eq_ignore_ascii_case(s))
    }
}

fn raw(inst: Instance) -> Option<*const KernelTable> {
    // SAFETY: the exported functions return the address of a constant-initialized table with static storage
    unsafe {
        match inst {
            #[cfg(feature = "w32")]
            Instance::U32Barrett64 => Some(pc_table_u32_barrett64()),
            #[cfg(feature = "w32")]
            Instance::U32Barrett32 => Some(pc_table_u32_barrett32()),
            #[cfg(feature = "mont")]
            Instance::U32Mont => Some(pc_table_u32_mont()),
            #[cfg(feature = "w64")]
            Instance::U64Barrett64 => Some(pc_table_u64_barrett64()),
            #[cfg(feature = "w16")]
            Instance::U16Barrett32 => Some(pc_table_u16_barrett32()),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }
}

/// The kernel table of `inst`, checked against this crate's ABI, word size, ModMul kind and ring dimension.
pub fn table(inst: Instance) -> Result<&'static KernelTable, TableError> {
    let p = raw(inst).ok_or(TableError::NotBuilt(inst.name(), inst.feature()))?;
    // SAFETY: non-null pointer to an immutable static table (see `raw`)
    let t: &'static KernelTable = unsafe { &*p };
    let name = inst.name();
    let chk = |field, got, want| if got == want { Ok(()) } else { Err(TableError::Mismatch { name, field, got, want }) };
    chk("abi", t.abi, ABI)?;
    chk("word_bytes", t.word_bytes, inst.word_bytes())?;
    chk("mm", t.mm, inst.mm())?;
    chk("log_n", t.log_n, LOG_N)?;
    Ok(t)
}
