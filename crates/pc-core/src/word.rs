//! The machine word an RNS limb is stored in (seam "word size"). Host code is generic over `Word`; device kernels are
//! instantiated per `WordId`. Every residue is `< p < 2^MAX_PRIME_BITS`, which leaves the headroom the lazy kernels need.
use std::fmt::{Debug, Display};
use std::hash::Hash;

/// Runtime tag of a word type (stamped into plans and kernel table names).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum WordId {
    U16 = 16,
    U32 = 32,
    U64 = 64,
}

impl WordId {
    pub fn bits(self) -> u32 {
        self as u32
    }
}

pub trait Word: Copy + Clone + Debug + Display + Default + Eq + Ord + Hash + Send + Sync + 'static {
    /// The double-width type a full product `a * b` lives in.
    type Wide: Copy + Debug + Eq + Ord;
    const BITS: u32;
    /// Largest prime size any `ModMul` over this word accepts. One bit below the word for u32/u16 (lazy adds and
    /// Shoup need `p < 2^(BITS-1)`); 60 bits for u64, OpenFHE's NativeInteger modulus limit.
    const MAX_PRIME_BITS: u32;
    const ID: WordId;
    const ZERO: Self;
    const ONE: Self;

    /// Narrowing conversion; the caller guarantees `x < 2^BITS` (debug-asserted).
    fn from_u64(x: u64) -> Self;
    fn to_u64(self) -> u64;
    fn to_wide(self) -> Self::Wide;
    fn wide_to_u128(w: Self::Wide) -> u128;
    /// `x mod 2^BITS` of a u128 (wrapping narrow).
    fn from_u128_wrapping(x: u128) -> Self;
}

macro_rules! impl_word {
    ($t:ty, $w:ty, $bits:expr, $maxp:expr, $id:expr) => {
        impl Word for $t {
            type Wide = $w;
            const BITS: u32 = $bits;
            const MAX_PRIME_BITS: u32 = $maxp;
            const ID: WordId = $id;
            const ZERO: Self = 0;
            const ONE: Self = 1;
            #[inline]
            fn from_u64(x: u64) -> Self {
                debug_assert!(<$t>::try_from(x).is_ok(), "value {x} does not fit a {}-bit word", $bits);
                x as $t
            }
            #[inline]
            fn to_u64(self) -> u64 {
                self as u64
            }
            #[inline]
            fn to_wide(self) -> $w {
                self as $w
            }
            #[inline]
            fn wide_to_u128(w: $w) -> u128 {
                w as u128
            }
            #[inline]
            fn from_u128_wrapping(x: u128) -> Self {
                x as $t
            }
        }
    };
}

impl_word!(u16, u32, 16, 15, WordId::U16);
impl_word!(u32, u64, 32, 31, WordId::U32);
impl_word!(u64, u128, 64, 60, WordId::U64);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ids_and_bits() {
        assert_eq!(<u32 as Word>::ID.bits(), 32);
        assert_eq!(<u64 as Word>::BITS, 64);
        assert_eq!(<u16 as Word>::from_u64(65535), u16::MAX);
        assert_eq!(<u32 as Word>::wide_to_u128(7u32.to_wide() * 9), 63);
    }
}
