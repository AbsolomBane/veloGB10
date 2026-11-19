//! KVPOOL (fork, 2026-11): a shared physical pool behind the per-lane position planes.
//!
//! The EXL3 engine allocates every position plane (trunk KV, raw/pooled indexer keys, the draft
//! head's KV and keys) as one `[width][segs][max_pos][row]` slab, so `--max-batch N` pins N full
//! windows of memory whether or not a lane holds any tokens. This module keeps the SAME virtual
//! layout — every kernel, launch argument and captured CUDA graph is untouched — but backs it with
//! CUDA virtual memory: the address range is reserved up front and 2 MiB physical pages from ONE
//! fixed pool are mapped under the rows a lane actually uses, and handed back when the lane resets.
//!
//! Design rules (measured on GB10, see docs in the fork):
//! * The physical pool is created ONCE at boot. `cuMemCreate` under memory pressure takes 100-400 ms
//!   per 32 MiB and can fail (CUDA sees MemFree, not page cache); moving an existing page between
//!   addresses is ~0.06 ms and never allocates.
//! * Addresses never move: graphs captured before a mapping replay correctly after it, and after an
//!   unmap+remap with different physical pages.
//! * Touching an unmapped page is a fatal device fault, not an error return. Callers map AHEAD of the
//!   highest row a round can touch (`FwdModel::vmm_ensure`), and `Mode::Full` maps everything at boot
//!   so the plumbing can be proven bit-exact before on-demand mapping is switched on.
//!
//! Off by default: `--kv-pool full|lazy`, pool size `--kv-pool-gb` (lazy; default = everything). One box only.

use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::sys as cu;

/// Physical page size: the GB10 VMM allocation granularity (verified at `new`).
pub const PAGE: usize = 2 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Every page of every plane mapped at reserve time. Same memory as the slab allocation; proves
    /// the VMM plumbing changes nothing.
    Full,
    /// Pages mapped on demand from the fixed pool.
    Lazy,
}

/// `--kv-pool off|full|lazy`: off (the default) = None (the stock slab allocation).
pub fn mode_from_opts() -> Option<Mode> {
    match crate::opts::var(crate::opt!("kv-pool")).ok().as_deref() {
        Some("full") => Some(Mode::Full),
        Some("lazy") => Some(Mode::Lazy),
        _ => None,
    }
}

/// `--kv-pool full|lazy` is refused with any TP role: a lane the pool cannot grow is ended on the rank that ran out,
/// and no TP event mirrors that to the peers yet (their lane sets would diverge). Called from the startup checks.
pub fn tp_topology(pool_set: bool, args: &[String]) -> Result<(), String> {
    if !pool_set { return Ok(()); }
    match args.iter().find(|a| matches!(a.as_str(), "--tp" | "--head" | "--node") || a.starts_with("--tp=")) {
        Some(role) => Err(format!("--kv-pool is one-box only and is refused with {} (a full pool ends a lane on one rank only)",
                                  role.split('=').next().unwrap_or(role))),
        None => Ok(()),
    }
}

/// `--kv-pool-gb` (decimal GB, like the boot fit message). None = size the pool to cover every plane.
pub fn pool_bytes_from_opts() -> Option<usize> {
    crate::opts::var(crate::opt!("kv-pool-gb")).ok().and_then(|v| v.parse::<u64>().ok()).filter(|g| *g > 0).map(|g| g as f64)
        .map(|g| (g * 1e9) as usize)
}

struct Plane {
    base: u64,
    /// Reserved bytes (a multiple of PAGE; >= the plane's logical length).
    bytes: usize,
    /// Fresh pages are zero-filled (the indexer key planes: `qsa_pool_update_b` reads the whole
    /// `ratio`-row block, rows past the lane's position included, and the stock boot uploads zeros).
    zero: bool,
    /// Per page: index into `KvVmm::handles` + 1, 0 = unmapped.
    pages: Vec<u32>,
}

/// KVPOOL (v2 dry-run): page bounds exactly as `ensure` walks them (every page the range touches) and as
/// `drop_range` frees them (only pages wholly inside the range). Pure arithmetic, shared by the real calls
/// and the dry-run queries below, so the head's capacity gate and the apply path cannot disagree.
pub fn ensure_page_bounds(off: usize, len: usize) -> (usize, usize) {
    (off / PAGE, (off + len).div_ceil(PAGE))
}
pub fn drop_page_bounds(off: usize, len: usize) -> (usize, usize) {
    (off.div_ceil(PAGE), (off + len) / PAGE)
}
/// Pages in `pages[p0..p1]` that are unmapped (`want_mapped == false`) or mapped (`true`).
pub fn count_pages(pages: &[u32], (p0, p1): (usize, usize), want_mapped: bool) -> usize {
    let p1 = p1.min(pages.len());
    if p0 >= p1 { return 0; }
    pages[p0..p1].iter().filter(|&&h| (h != 0) == want_mapped).count()
}

pub struct KvVmm {
    mode: Mode,
    dev_id: i32,
    planes: Vec<Plane>,
    handles: Vec<cu::CUmemGenericAllocationHandle>,
    free: Vec<u32>,
    /// Lazy with no explicit pool size: grow the pool to cover each reserved plane (boot only).
    cover_all: bool,
    sealed: bool,
    pub stat_maps: u64,
    pub stat_unmaps: u64,
}

// Raw CUDA handles and addresses; every use is under the model's Mutex with the context bound.
unsafe impl Send for KvVmm {}

fn ck(r: cu::CUresult, what: &str) -> Result<()> {
    if r == cu::CUresult::CUDA_SUCCESS { Ok(()) } else { bail!("kv-vmm: {what} failed: {r:?}") }
}

impl KvVmm {
    /// `pool_bytes`: the physical pool for `Mode::Lazy` (None = grow to cover every plane reserved before
    /// `seal`). `Mode::Full` ignores it (each plane brings its own pages).
    pub fn new(mode: Mode, dev_ordinal: usize, pool_bytes: Option<usize>) -> Result<Self> {
        let mut v = KvVmm { mode, dev_id: dev_ordinal as i32, planes: Vec::new(), handles: Vec::new(), free: Vec::new(),
                            cover_all: mode == Mode::Full || pool_bytes.is_none(), sealed: false, stat_maps: 0, stat_unmaps: 0 };
        let prop = v.prop();
        let mut gran = 0usize;
        ck(unsafe { cu::cuMemGetAllocationGranularity(&mut gran, &prop,
                    cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM) }, "cuMemGetAllocationGranularity")?;
        ensure!(gran > 0 && PAGE % gran == 0, "kv-vmm: device granularity {gran} does not divide the {PAGE}-byte page");
        if mode == Mode::Lazy {
            if let Some(b) = pool_bytes { v.grow(b.div_ceil(PAGE))?; }
        }
        Ok(v)
    }

    fn prop(&self) -> cu::CUmemAllocationProp {
        let mut p: cu::CUmemAllocationProp = unsafe { std::mem::zeroed() };
        p.type_ = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        p.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        p.location.id = self.dev_id;
        p
    }

    fn grow(&mut self, pages: usize) -> Result<()> {
        let prop = self.prop();
        for _ in 0..pages {
            let mut h: cu::CUmemGenericAllocationHandle = 0;
            ck(unsafe { cu::cuMemCreate(&mut h, PAGE, &prop, 0) }, "cuMemCreate (kv pool page)")
                .with_context(|| format!("after {} pages ({:.2} GB)", self.handles.len(), (self.handles.len() * PAGE) as f64 / 1e9))?;
            self.free.push(self.handles.len() as u32);
            self.handles.push(h);
        }
        Ok(())
    }

    /// Reserve the address range of one plane of `bytes` logical bytes. Returns its base address.
    pub fn reserve(&mut self, bytes: usize, zero: bool) -> Result<u64> {
        ensure!(!self.sealed, "kv-vmm: reserve after seal");
        ensure!(bytes > 0, "kv-vmm: empty plane");
        let rbytes = bytes.div_ceil(PAGE) * PAGE;
        let mut base: cu::CUdeviceptr = 0;
        ck(unsafe { cu::cuMemAddressReserve(&mut base, rbytes, PAGE, 0, 0) }, "cuMemAddressReserve")?;
        let npages = rbytes / PAGE;
        self.planes.push(Plane { base: base as u64, bytes: rbytes, zero, pages: vec![0; npages] });
        if self.cover_all { self.grow(npages)?; }
        if self.mode == Mode::Full { self.ensure(base as u64, rbytes)?; }
        Ok(base as u64)
    }

    /// No more planes: the pool is final.
    pub fn seal(&mut self) { self.sealed = true; }

    fn plane_of(&self, at: u64, len: usize) -> Result<usize> {
        for (i, p) in self.planes.iter().enumerate() {
            if at >= p.base && at + len as u64 <= p.base + p.bytes as u64 { return Ok(i); }
        }
        bail!("kv-vmm: range {at:#x}+{len} is not inside a reserved plane")
    }

    /// Make every page under `[at, at+len)` mapped. Returns the pages newly mapped (0 = nothing to do).
    /// Err (nothing half-done that matters: pages mapped so far stay mapped) when the pool is exhausted.
    /// KVPOOL (v2): how many pages `ensure(at, len)` would map right now (0 = nothing to do). Dry run: changes nothing.
    pub fn pages_needed(&self, at: u64, len: usize) -> Result<usize> {
        if len == 0 { return Ok(0); }
        let pi = self.plane_of(at, len)?;
        let base = self.planes[pi].base;
        Ok(count_pages(&self.planes[pi].pages, ensure_page_bounds((at - base) as usize, len), false))
    }
    /// KVPOOL (v2): how many mapped pages `drop_range(at, len)` would return to the pool. Dry run: changes nothing.
    /// Full mode maps everything and never drops, so it frees 0.
    pub fn pages_freed(&self, at: u64, len: usize) -> Result<usize> {
        if self.mode == Mode::Full || len == 0 { return Ok(0); }
        let pi = self.plane_of(at, len)?;
        let base = self.planes[pi].base;
        Ok(count_pages(&self.planes[pi].pages, drop_page_bounds((at - base) as usize, len), true))
    }
    pub fn ensure(&mut self, at: u64, len: usize) -> Result<usize> {
        if len == 0 { return Ok(0); }
        let pi = self.plane_of(at, len)?;
        let (base, zero) = (self.planes[pi].base, self.planes[pi].zero);
        let (p0, p1) = ensure_page_bounds((at - base) as usize, len);
        let mut acc: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
        acc.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        acc.location.id = self.dev_id;
        acc.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        let mut n = 0usize;
        for pg in p0..p1 {
            if self.planes[pi].pages[pg] != 0 { continue; }
            let Some(hi) = self.free.pop() else {
                bail!("kv-vmm: pool exhausted ({} pages = {:.2} GB all mapped)", self.handles.len(), (self.handles.len() * PAGE) as f64 / 1e9)
            };
            let addr = (base + (pg * PAGE) as u64) as cu::CUdeviceptr;
            let r = unsafe { cu::cuMemMap(addr, PAGE, 0, self.handles[hi as usize], 0) };
            if r != cu::CUresult::CUDA_SUCCESS { self.free.push(hi); ck(r, "cuMemMap")?; }
            // Ownership is published the moment the mapping exists: a later access/zero failure must leave the page
            // registered (drop_range and Drop unmap it, a retry does not map over it) — the error still propagates.
            self.planes[pi].pages[pg] = hi + 1;
            self.stat_maps += 1;
            ck(unsafe { cu::cuMemSetAccess(addr, PAGE, &acc, 1) }, "cuMemSetAccess")?;
            if zero { ck(unsafe { cu::cuMemsetD8_v2(addr, 0, PAGE) }, "cuMemsetD8 (fresh key page)")?; }
            n += 1;
        }
        Ok(n)
    }

    /// Unmap every page that lies WHOLLY inside `[at, at+len)` and return it to the pool. Pages that
    /// straddle the range's edges belong partly to a neighbour (the next segment or lane) and stay.
    /// The caller has synchronized the device: no queued kernel may still reference these rows.
    pub fn drop_range(&mut self, at: u64, len: usize) -> Result<usize> {
        if self.mode == Mode::Full || len == 0 { return Ok(0); }
        let pi = self.plane_of(at, len)?;
        let base = self.planes[pi].base;
        let (p0, p1) = drop_page_bounds((at - base) as usize, len);
        let mut n = 0usize;
        for pg in p0..p1 {
            let h = self.planes[pi].pages[pg];
            if h == 0 { continue; }
            ck(unsafe { cu::cuMemUnmap((base + (pg * PAGE) as u64) as cu::CUdeviceptr, PAGE) }, "cuMemUnmap")?;
            self.planes[pi].pages[pg] = 0;
            self.free.push(h - 1);
            n += 1;
        }
        self.stat_unmaps += n as u64;
        Ok(n)
    }

    pub fn mode(&self) -> Mode { self.mode }
    pub fn pool_pages(&self) -> usize { self.handles.len() }
    pub fn free_pages(&self) -> usize { self.free.len() }
    pub fn mapped_bytes(&self) -> usize { (self.handles.len() - self.free.len()) * PAGE }
    pub fn pool_bytes(&self) -> usize { self.handles.len() * PAGE }
    pub fn reserved_bytes(&self) -> usize { self.planes.iter().map(|p| p.bytes).sum() }
}

impl Drop for KvVmm {
    fn drop(&mut self) {
        // Best effort at teardown (the slices over these planes were leaked by the model's Drop).
        unsafe {
            for p in &self.planes {
                for (pg, h) in p.pages.iter().enumerate() {
                    if *h != 0 { let _ = cu::cuMemUnmap((p.base + (pg * PAGE) as u64) as cu::CUdeviceptr, PAGE); }
                }
                let _ = cu::cuMemAddressFree(p.base as cu::CUdeviceptr, p.bytes);
            }
            for h in &self.handles { let _ = cu::cuMemRelease(*h); }
        }
    }
}

/// True for the error `KvVmm::ensure` returns when every page of the pool is mapped (the caller may free a lane and retry).
pub fn is_exhausted(e: &anyhow::Error) -> bool { format!("{e:#}").contains("kv-vmm: pool exhausted") }

/// Rows of a position plane to keep mapped for a lane that will touch rows `< rows`: rounded up to a
/// grid so the fast path (an atomic compare) absorbs almost every call, capped at the plane.
pub fn rows_grid(rows: usize, cap: usize) -> usize {
    const GRID: usize = 4096;
    (rows.div_ceil(GRID) * GRID).min(cap)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rows_grid_rounds_up_and_caps() {
        assert_eq!(rows_grid(0, 1000), 0);
        assert_eq!(rows_grid(1, 100_000), 4096);
        assert_eq!(rows_grid(4096, 100_000), 4096);
        assert_eq!(rows_grid(4097, 100_000), 8192);
        assert_eq!(rows_grid(99_999, 100_000), 100_000);
    }
    /// The pool is opt-in (default off = today's allocation byte-for-byte) and one-box only.
    #[test]
    fn kv_pool_is_off_by_default_and_refused_under_tp() {
        use crate::opts::{Scope, Ty};
        let o = crate::opt!("kv-pool").def();
        assert_eq!((o.def, o.ty, o.scope), ("off", Ty::Choice(&["off", "full", "lazy"]), Scope::Spmd));
        assert_eq!(crate::opt!("kv-pool-gb").def().def, "unset");
        assert_eq!(crate::opt!("fit-floor-gb").def().def, "20");
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        assert!(tp_topology(false, &a(&["--tp", "2"])).is_ok(), "pool off: TP untouched");
        assert!(tp_topology(true, &a(&["--server"])).is_ok());
        for role in ["--tp", "--tp=2", "--head", "--node"] {
            let e = tp_topology(true, &a(&["--server", role])).unwrap_err();
            assert!(e.starts_with("--kv-pool is one-box only and is refused with --"), "{e}");
        }
    }
}

#[cfg(test)]
mod dry_run_tests {
    use super::*;
    #[test]
    fn ensure_bounds_cover_every_touched_page() {
        assert_eq!(ensure_page_bounds(0, 1), (0, 1));
        assert_eq!(ensure_page_bounds(PAGE - 1, 2), (0, 2));          // straddles a boundary: both pages
        assert_eq!(ensure_page_bounds(PAGE, PAGE), (1, 2));
        assert_eq!(ensure_page_bounds(3 * PAGE, 0), (3, 3));          // empty
    }
    #[test]
    fn drop_bounds_free_only_wholly_covered_pages() {
        assert_eq!(drop_page_bounds(0, PAGE), (0, 1));
        assert_eq!(drop_page_bounds(1, PAGE), (1, 1));                 // straddles both edges: nothing wholly inside
        assert_eq!(drop_page_bounds(PAGE, 2 * PAGE), (1, 3));
        assert_eq!(drop_page_bounds(PAGE - 1, PAGE + 2), (1, 2));      // covers page 1 wholly
    }
    #[test]
    fn boundary_page_is_counted_once_in_a_union_not_per_lane() {
        // lane A: [0, PAGE+1) needs pages 0,1; lane B: [PAGE+1, 2*PAGE+5) needs pages 1,2 (page 1 is shared)
        let pages = [0u32, 0, 0, 0];
        let a = count_pages(&pages, ensure_page_bounds(0, PAGE + 1), false);
        let b = count_pages(&pages, ensure_page_bounds(PAGE + 1, PAGE + 4), false);
        let union = count_pages(&pages, (0, 3), false);
        assert_eq!((a, b, union), (2, 2, 3));   // per-lane sum double-counts page 1: never sum ensure queries across lanes
    }
    #[test]
    fn count_pages_respects_mapped_and_unmapped() {
        let pages = [0u32, 7, 0, 3, 0];
        assert_eq!(count_pages(&pages, (0, 5), false), 3);
        assert_eq!(count_pages(&pages, (0, 5), true), 2);
        assert_eq!(count_pages(&pages, (2, 2), false), 0);            // empty range
        assert_eq!(count_pages(&pages, (3, 99), true), 1);            // clamped to the plane
    }
}
