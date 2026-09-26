use anyhow::{anyhow, Result};
use burn::prelude::*;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};

use super::dataset::Batch;
use super::source::{prepare_sample, PreparedSample, SharedSampleStore};

#[derive(Clone, Copy, Debug)]
pub struct PlannedSample {
    pub key: usize,
    pub orientation: u8,
}

/// Deterministic D4 coverage. A sample sees every square symmetry once over
/// eight epochs, while a keyed hash prevents one orientation from filling an
/// entire batch.
pub fn orientation_for(seed: u64, epoch: usize, key: usize) -> u8 {
    let mut value = seed ^ (key as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    ((value as usize + epoch) & 7) as u8
}

pub struct HostBatch {
    samples: Vec<PreparedSample>,
    input_size: usize,
}

impl HostBatch {
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Collate on the host and upload one contiguous image tensor per batch.
    pub fn upload(self, device: &Device) -> Batch {
        let batch_size = self.samples.len();
        let image_len = 3 * self.input_size * self.input_size;
        let target_count: usize = self.samples.iter().map(|sample| sample.cls.len()).sum();
        let mut images = Vec::with_capacity(batch_size * image_len);
        let mut cls = Vec::with_capacity(target_count);
        let mut bbox = Vec::with_capacity(target_count * 4);
        let mut idx = Vec::with_capacity(target_count);
        for (batch_index, sample) in self.samples.into_iter().enumerate() {
            images.extend(sample.image);
            for (class, bounds) in sample.cls.into_iter().zip(sample.bbox) {
                cls.push(class);
                bbox.extend_from_slice(&bounds);
                idx.push(batch_index as f32);
            }
        }
        let images = Tensor::<1>::from_floats(images.as_slice(), device).reshape([
            batch_size,
            3,
            self.input_size,
            self.input_size,
        ]);
        let (cls, bbox, idx) = if target_count == 0 {
            (
                Tensor::<2>::zeros([1, 1], device),
                Tensor::<2>::zeros([1, 4], device),
                Tensor::<1>::from_floats([-1.0f32].as_slice(), device),
            )
        } else {
            (
                Tensor::<1>::from_floats(cls.as_slice(), device).reshape([target_count, 1]),
                Tensor::<1>::from_floats(bbox.as_slice(), device).reshape([target_count, 4]),
                Tensor::<1>::from_floats(idx.as_slice(), device),
            )
        };
        Batch {
            images,
            cls,
            bbox,
            idx,
        }
    }
}

type Loaded = (usize, Result<PreparedSample>);

/// Bounded, ordered loader. Storage and CPU transforms run on worker threads;
/// the optimizer thread only collates and uploads completed host batches.
pub struct ParallelBatchLoader {
    store: SharedSampleStore,
    plan: Arc<Vec<PlannedSample>>,
    receiver: mpsc::Receiver<Loaded>,
    pending: BTreeMap<usize, Result<PreparedSample>>,
    _handles: Vec<JoinHandle<()>>,
    next_position: usize,
    plan_len: usize,
    batch_size: usize,
    input_size: usize,
    prefetched_until: usize,
}

impl ParallelBatchLoader {
    pub fn new(
        store: SharedSampleStore,
        plan: Vec<PlannedSample>,
        input_size: usize,
        batch_size: usize,
        workers: usize,
        queue_batches: usize,
    ) -> Result<Self> {
        anyhow::ensure!(batch_size > 0, "batch size must be positive");
        anyhow::ensure!(workers > 0, "loader worker count must be positive");
        let prefetch_samples = batch_size.saturating_mul(queue_batches.max(1));
        let keys: Vec<_> = plan
            .iter()
            .take(prefetch_samples)
            .map(|sample| sample.key)
            .collect();
        store.prefetch(&keys)?;

        let plan = Arc::new(plan);
        let next = Arc::new(AtomicUsize::new(0));
        let (sender, receiver) = mpsc::sync_channel::<Loaded>(prefetch_samples.max(workers));
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let store = Arc::clone(&store);
            let plan = Arc::clone(&plan);
            let next = Arc::clone(&next);
            let sender = sender.clone();
            handles.push(thread::spawn(move || loop {
                let position = next.fetch_add(1, Ordering::Relaxed);
                let Some(sample) = plan.get(position).copied() else {
                    break;
                };
                let loaded = store
                    .load(sample.key)
                    .and_then(|raw| prepare_sample(raw, input_size, sample.orientation));
                if sender.send((position, loaded)).is_err() {
                    break;
                }
            }));
        }
        drop(sender);
        let plan_len = plan.len();
        Ok(Self {
            store,
            plan,
            receiver,
            pending: BTreeMap::new(),
            _handles: handles,
            next_position: 0,
            plan_len,
            batch_size,
            input_size,
            prefetched_until: keys.len(),
        })
    }

    pub fn num_batches(&self) -> usize {
        self.plan_len.div_ceil(self.batch_size)
    }

    pub fn next_batch(&mut self) -> Result<Option<HostBatch>> {
        if self.next_position == self.plan_len {
            return Ok(None);
        }
        let end = (self.next_position + self.batch_size).min(self.plan_len);
        let mut samples = Vec::with_capacity(end - self.next_position);
        while self.next_position < end {
            if let Some(sample) = self.pending.remove(&self.next_position) {
                samples.push(sample?);
                self.next_position += 1;
                continue;
            }
            let (position, sample) = self
                .receiver
                .recv()
                .map_err(|_| anyhow!("all loader workers stopped before the epoch completed"))?;
            self.pending.insert(position, sample);
        }
        let new_prefetch_end = (self.prefetched_until + self.batch_size).min(self.plan_len);
        if new_prefetch_end > self.prefetched_until {
            let keys: Vec<_> = self.plan[self.prefetched_until..new_prefetch_end]
                .iter()
                .map(|sample| sample.key)
                .collect();
            self.store.prefetch(&keys)?;
            self.prefetched_until = new_prefetch_end;
        }
        Ok(Some(HostBatch {
            samples,
            input_size: self.input_size,
        }))
    }
}
