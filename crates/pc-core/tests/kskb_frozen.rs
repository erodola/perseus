//! The u64 KSK expander (KSKB lane) is frozen. Pins its keystream on the n64 chain's primes; a change here
//! invalidates every serialized u64 key. The row form must equal the per-slot form (the device twin is checked
//! against `kskb_expand` in pc-kernels-selftest).
use pc_core::keys::{check_kskb, ksk_expand_row, kskb_expand, kskb_expand_row, KskSeed};
use pc_core::params::{Chain, ChainSpec};

const SEED: KskSeed = KskSeed([0x0123_4567, 0x89ab_cdef, 0xdead_beef, 0x0bad_f00d, 1, 2, 3, 0xffff_ffff]);

fn fnv(v: &[u64]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x).wrapping_mul(0x0000_0100_0000_01b3))
}

#[test]
fn kskb_row_equals_slot_form_and_is_pinned() {
    let chain = Chain::generate(&ChainSpec::n64()).unwrap();
    let n = chain.n;
    let primes: Vec<u64> = chain.q.iter().chain(&chain.p).copied().collect();
    check_kskb(chain.dnum(), &primes).unwrap();
    let mut hashes = Vec::new();
    for (digit, p) in [(0u32, primes[0]), (3, primes[17]), (6, *primes.last().unwrap())] {
        let mut row = vec![0u64; n];
        kskb_expand_row(&SEED, digit, p, &mut row);
        for s in (0..n).step_by(97).chain(n - 8..n) {
            assert_eq!(row[s], kskb_expand(&SEED, digit, p, s as u32, n), "digit {digit} p {p} slot {s}");
        }
        assert!(row.iter().all(|&x| x < p));
        let mut via = vec![0u64; n];
        ksk_expand_row(&SEED, digit, p, 64, &mut via).unwrap();
        assert_eq!(via, row);
        hashes.push((p, row[0], row[1], row[n - 1], fnv(&row)));
    }
    assert_eq!(hashes, PINNED.to_vec());
}

#[test]
fn kskb_domain_checks() {
    assert!(check_kskb(17, &[(1 << 59) + 1]).is_err());
    assert!(check_kskb(7, &[1 << 60]).is_err());
    assert!(ksk_expand_row(&SEED, 0, 97, 16, &mut [0u64; 16]).is_err());
    // the u32 lane through the dispatcher is KSKA
    let mut a = vec![0u64; 64];
    ksk_expand_row(&SEED, 2, 268_369_921, 32, &mut a).unwrap();
    assert_eq!(a[5], pc_core::keys::kska_expand(&SEED, 2, 268_369_921, 5, 64) as u64);
}

/// (p, a[0], a[1], a[n-1], FNV-1a of the row), digits 0 / 3 / 6. Produced by FIDESlib `KskSeedExpand.cuh::
/// expand_coeff64` compiled on the host with the same seed: identical.
const PINNED: [(u64, u64, u64, u64, u64); 3] = [
    (1152921504606584833, 887896118268210575, 381724958222899765, 166443527968165288, 17347891236941834349),
    (9007199273091073, 3071635100410335, 4572704283048939, 2445311928879128, 6946775773964342884),
    (1152921504592822273, 229142879966039658, 808485329004953024, 1085433559467732095, 2359252148008581173),
];
