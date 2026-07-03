use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::ops::Range;
#[cfg(test)]
use std::sync::MutexGuard;
use std::sync::{Mutex, OnceLock};

static RNG: OnceLock<Mutex<StdRng>> = OnceLock::new();

fn rng() -> &'static Mutex<StdRng> {
    RNG.get_or_init(|| Mutex::new(StdRng::seed_from_u64(0)))
}

#[cfg(test)]
pub(crate) fn test_lock() -> MutexGuard<'static, ()> {
    static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    TEST_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(crate) fn set_seed(seed: u64) {
    *rng().lock().expect("global RNG mutex poisoned") = StdRng::seed_from_u64(seed);
}

pub(crate) fn gen_f32() -> f32 {
    rng().lock().expect("global RNG mutex poisoned").gen()
}

pub(crate) fn gen_range_f32(range: Range<f32>) -> f32 {
    rng()
        .lock()
        .expect("global RNG mutex poisoned")
        .gen_range(range)
}

pub(crate) fn gen_range_usize(range: Range<usize>) -> usize {
    rng()
        .lock()
        .expect("global RNG mutex poisoned")
        .gen_range(range)
}

pub(crate) fn shuffle<T>(slice: &mut [T]) {
    slice.shuffle(&mut *rng().lock().expect("global RNG mutex poisoned"));
}

pub(crate) fn sample<T, D>(distribution: D) -> T
where
    D: rand_distr::Distribution<T>,
{
    distribution.sample(&mut *rng().lock().expect("global RNG mutex poisoned"))
}
