use crate::constants::INITIAL_COEFFICIENTS;
use crate::tables::{Tables, delta, high_delta, short_delta, tables};

#[cfg(target_arch = "x86")]
use std::arch::x86 as intrinsics;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as intrinsics;

// Dispatch once per stage/frame, outside the per-sample adaptive filters.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn has_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

fn clamp16(x: i32) -> i16 {
    x.clamp(-32768, 32767) as i16
}
fn rounded(x: i32, shift: u32) -> i32 {
    x.wrapping_add(1 << (shift - 1)) >> shift
}

fn short_deltas(samples: [i16; 4]) -> [i16; 4] {
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "x86", target_feature = "sse2")
    ))]
    // SAFETY: SSE2 is enabled for this target. The load/store access four i16s.
    unsafe {
        use intrinsics::*;
        let zero = _mm_setzero_si128();
        let values = _mm_loadl_epi64(samples.as_ptr().cast());
        let sign = _mm_srai_epi16(values, 15);
        let magnitude = _mm_sub_epi16(_mm_xor_si128(values, sign), sign);
        // Compare unsigned magnitudes so abs(i16::MIN) is treated as 32768.
        let magnitude = _mm_xor_si128(magnitude, _mm_set1_epi16(i16::MIN));
        let mut steps = _mm_set1_epi16(1);
        for threshold in [1024i16, 3000, 5000] {
            steps = _mm_sub_epi16(
                steps,
                _mm_cmpgt_epi16(magnitude, _mm_set1_epi16(threshold ^ i16::MIN)),
            );
        }
        let deltas = _mm_sub_epi16(sign, _mm_xor_si128(steps, sign));
        let deltas = _mm_andnot_si128(_mm_cmpeq_epi16(values, zero), deltas);
        let mut result = [0; 4];
        _mm_storel_epi64(result.as_mut_ptr().cast(), deltas);
        result
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is mandatory on AArch64. The load/store access four i16s.
    unsafe {
        use std::arch::aarch64::*;
        let zero = vdup_n_s16(0);
        let values = vld1_s16(samples.as_ptr());
        let magnitude = vreinterpret_u16_s16(vabs_s16(values));
        let mut steps = vdup_n_s16(1);
        for threshold in [1024, 3000, 5000] {
            steps = vsub_s16(
                steps,
                vreinterpret_s16_u16(vcgt_u16(magnitude, vdup_n_u16(threshold))),
            );
        }
        let signs = vsub_s16(
            vreinterpret_s16_u16(vcgt_s16(values, zero)),
            vreinterpret_s16_u16(vclt_s16(values, zero)),
        );
        let mut result = [0; 4];
        vst1_s16(result.as_mut_ptr(), vmul_s16(steps, signs));
        result
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "x86", target_feature = "sse2")
    )))]
    samples.map(|sample| short_delta(sample as i32))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sample_deltas_avx2(samples: [i16; 4], scale: i32) -> [i16; 4] {
    // SAFETY: the caller checks AVX2. The load/store each access exactly the
    // four i16 elements; all intermediate arithmetic intentionally wraps.
    unsafe {
        use intrinsics::*;
        let zero = _mm_setzero_si128();
        let one = _mm_set1_epi32(1);
        let values = _mm_cvtepi16_epi32(_mm_loadl_epi64(samples.as_ptr().cast()));
        let values = _mm_srai_epi32(
            _mm_add_epi32(
                _mm_mullo_epi32(values, _mm_set1_epi32(scale)),
                _mm_set1_epi32(8192),
            ),
            14,
        );
        let values = _mm_min_epi32(
            _mm_max_epi32(values, _mm_set1_epi32(-65536)),
            _mm_set1_epi32(65535),
        );
        let magnitude = _mm_abs_epi32(values);
        let mut steps = _mm_set1_epi32(16);
        for threshold in [400, 700, 1224, 4096, 16384] {
            steps = _mm_add_epi32(
                steps,
                _mm_and_si128(
                    _mm_cmpgt_epi32(magnitude, _mm_set1_epi32(threshold)),
                    _mm_set1_epi32(16),
                ),
            );
        }
        let shifts = _mm_add_epi32(
            _mm_and_si128(_mm_cmpgt_epi32(_mm_set1_epi32(16), magnitude), one),
            _mm_and_si128(_mm_cmpgt_epi32(_mm_set1_epi32(4), magnitude), one),
        );
        let steps = _mm_srlv_epi32(steps, shifts);
        let deltas = _mm_sign_epi32(steps, _mm_sub_epi32(zero, values));
        let mut result = [0; 4];
        _mm_storel_epi64(result.as_mut_ptr().cast(), _mm_packs_epi32(deltas, zero));
        result
    }
}

#[cfg(target_arch = "aarch64")]
fn sample_deltas_neon(samples: [i16; 4], scale: i32) -> [i16; 4] {
    // SAFETY: NEON is mandatory on AArch64. The load/store access four i16s.
    unsafe {
        use std::arch::aarch64::*;
        let zero = vdupq_n_s32(0);
        let values = vmovl_s16(vld1_s16(samples.as_ptr()));
        let values = vshrq_n_s32(vaddq_s32(vmulq_n_s32(values, scale), vdupq_n_s32(8192)), 14);
        let values = vminq_s32(vmaxq_s32(values, vdupq_n_s32(-65536)), vdupq_n_s32(65535));
        let magnitude = vabsq_s32(values);
        let mut steps = vdupq_n_s32(16);
        for threshold in [400, 700, 1224, 4096, 16384] {
            steps = vaddq_s32(
                steps,
                vreinterpretq_s32_u32(vandq_u32(
                    vcgtq_s32(magnitude, vdupq_n_s32(threshold)),
                    vdupq_n_u32(16),
                )),
            );
        }
        let shifts = vaddq_u32(
            vandq_u32(vcltq_s32(magnitude, vdupq_n_s32(16)), vdupq_n_u32(1)),
            vandq_u32(vcltq_s32(magnitude, vdupq_n_s32(4)), vdupq_n_u32(1)),
        );
        let steps = vshlq_s32(steps, vnegq_s32(vreinterpretq_s32_u32(shifts)));
        let signs = vsubq_s32(
            vreinterpretq_s32_u32(vcgtq_s32(values, zero)),
            vreinterpretq_s32_u32(vcltq_s32(values, zero)),
        );
        let mut result = [0; 4];
        vst1_s16(result.as_mut_ptr(), vmovn_s32(vmulq_s32(steps, signs)));
        result
    }
}

#[derive(Clone)]
struct History {
    data: Vec<i16>,
    pos: usize,
    taps: usize,
}
impl History {
    fn new(taps: usize) -> Self {
        Self {
            data: vec![0; taps + 512],
            pos: taps,
            taps,
        }
    }
    fn advance(&mut self) {
        self.pos += 1;
        if self.pos == self.data.len() {
            self.data.copy_within(self.pos - self.taps..self.pos, 0);
            self.pos = self.taps;
        }
    }
    fn get(&self, back: usize) -> i16 {
        self.data[self.pos - back]
    }
    fn set(&mut self, back: usize, value: i16) {
        self.data[self.pos - back] = value;
    }
    fn window(&self) -> &[i16] {
        &self.data[self.pos + 1 - self.taps..=self.pos]
    }
    fn halve(&mut self, offsets: &[usize]) {
        for &offset in offsets {
            self.data[self.pos - offset] >>= 1;
        }
    }
}
#[inline(always)]
fn dot<const TAPS: usize, const AVX2: bool>(a: &[i16], b: &[i16]) -> i32 {
    let len = if TAPS == 0 {
        a.len().min(b.len())
    } else {
        TAPS
    };
    let (a, b) = (&a[..len], &b[..len]);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if AVX2 && len >= 96 {
        // SAFETY: AVX2=true is selected only after the CPU/OS feature check.
        return unsafe { dot_avx2::<TAPS>(a, b) };
    }
    dot_baseline(a, b)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot_avx2<const TAPS: usize>(a: &[i16], b: &[i16]) -> i32 {
    // SAFETY: the caller guarantees AVX2 and at least TAPS elements in both
    // slices (TAPS=0 selects their minimum length). The remainder is scalar.
    unsafe {
        use intrinsics::*;
        let len = if TAPS == 0 {
            a.len().min(b.len())
        } else {
            TAPS
        };
        let mut i = 0;
        let mut s0 = _mm256_setzero_si256();
        let mut s1 = s0;

        while i + 32 <= len {
            s0 = _mm256_add_epi32(
                s0,
                _mm256_madd_epi16(
                    _mm256_loadu_si256(a.as_ptr().add(i).cast()),
                    _mm256_loadu_si256(b.as_ptr().add(i).cast()),
                ),
            );
            s1 = _mm256_add_epi32(
                s1,
                _mm256_madd_epi16(
                    _mm256_loadu_si256(a.as_ptr().add(i + 16).cast()),
                    _mm256_loadu_si256(b.as_ptr().add(i + 16).cast()),
                ),
            );
            i += 32;
        }
        s0 = _mm256_add_epi32(s0, s1);
        if i + 16 <= len {
            s0 = _mm256_add_epi32(
                s0,
                _mm256_madd_epi16(
                    _mm256_loadu_si256(a.as_ptr().add(i).cast()),
                    _mm256_loadu_si256(b.as_ptr().add(i).cast()),
                ),
            );
            i += 16;
        }
        let mut sum = _mm_add_epi32(_mm256_castsi256_si128(s0), _mm256_extracti128_si256(s0, 1));
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
        a[i..len]
            .iter()
            .zip(&b[i..len])
            .fold(_mm_cvtsi128_si32(sum), |sum, (&x, &y)| {
                sum.wrapping_add(x as i32 * y as i32)
            })
    }
}

fn dot_baseline(a: &[i16], b: &[i16]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is mandatory on AArch64. Loads stay within the shorter slice.
    unsafe {
        use std::arch::aarch64::*;
        let len = a.len().min(b.len());
        let mut sum = vdupq_n_s32(0);
        let mut i = 0;
        while i + 8 <= len {
            let x = vld1q_s16(a.as_ptr().add(i));
            let y = vld1q_s16(b.as_ptr().add(i));
            sum = vmlal_s16(sum, vget_low_s16(x), vget_low_s16(y));
            sum = vmlal_high_s16(sum, x, y);
            i += 8;
        }
        a[i..len]
            .iter()
            .zip(&b[i..len])
            .fold(vaddvq_s32(sum), |sum, (&x, &y)| {
                sum.wrapping_add(x as i32 * y as i32)
            })
    }
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "x86", target_feature = "sse2")
    ))]
    {
        use intrinsics::*;
        let len = a.len().min(b.len());
        let (a, a_remainder) = a[..len].as_chunks::<8>();
        let (b, b_remainder) = b[..len].as_chunks::<8>();

        unsafe {
            let mut sum = _mm_setzero_si128();
            for (a, b) in a.iter().zip(b) {
                sum = _mm_add_epi32(
                    sum,
                    _mm_madd_epi16(
                        _mm_loadu_si128(a.as_ptr().cast()),
                        _mm_loadu_si128(b.as_ptr().cast()),
                    ),
                );
            }
            sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
            sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
            a_remainder
                .iter()
                .zip(b_remainder)
                .fold(_mm_cvtsi128_si32(sum), |sum, (&x, &y)| {
                    sum.wrapping_add(x as i32 * y as i32)
                })
        }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "x86", target_feature = "sse2")
    )))]
    a.iter()
        .zip(b)
        .fold(0i32, |sum, (&x, &y)| sum.wrapping_add(x as i32 * y as i32))
}

// Updating and multiplying in one pass avoids loading the weights twice.
// AVX2=true is only instantiated behind the per-frame CPU/OS check.
#[inline(always)]
fn adapt_dot<const TAPS: usize, const AVX2: bool>(
    weights: &mut [i16],
    deltas: &[i16],
    data: &[i16],
    amount: i32,
) -> i32 {
    debug_assert_eq!(weights.len(), deltas.len());
    debug_assert_eq!(weights.len(), data.len());
    let len = if TAPS == 0 { weights.len() } else { TAPS };
    let weights = &mut weights[..len];
    let (deltas, data) = (&deltas[..len], &data[..len]);
    if amount == 0 {
        return dot::<TAPS, AVX2>(data, weights);
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if AVX2 && len >= 96 {
        // SAFETY: AVX2 is selected only by process_avx2 after feature detection.
        return unsafe { adapt_dot_avx2::<TAPS>(weights, deltas, data, amount) };
    }
    adapt_dot_baseline(weights, deltas, data, amount)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn adapt_dot_avx2<const TAPS: usize>(
    weights: &mut [i16],
    deltas: &[i16],
    data: &[i16],
    amount: i32,
) -> i32 {
    // SAFETY: the caller guarantees AVX2, equal slice lengths and at least TAPS
    // elements (TAPS=0 uses the slice length). The remainder is scalar.
    unsafe {
        use intrinsics::*;
        let len = if TAPS == 0 { weights.len() } else { TAPS };
        let step = _mm256_set1_epi16(amount as i16);
        let mut sum0 = _mm256_setzero_si256();
        let mut sum1 = sum0;
        let mut i = 0;
        while i + 32 <= len {
            let w0 = _mm256_sub_epi16(
                _mm256_loadu_si256(weights.as_ptr().add(i).cast()),
                _mm256_mullo_epi16(_mm256_loadu_si256(deltas.as_ptr().add(i).cast()), step),
            );
            let w1 = _mm256_sub_epi16(
                _mm256_loadu_si256(weights.as_ptr().add(i + 16).cast()),
                _mm256_mullo_epi16(_mm256_loadu_si256(deltas.as_ptr().add(i + 16).cast()), step),
            );
            _mm256_storeu_si256(weights.as_mut_ptr().add(i).cast(), w0);
            _mm256_storeu_si256(weights.as_mut_ptr().add(i + 16).cast(), w1);
            sum0 = _mm256_add_epi32(
                sum0,
                _mm256_madd_epi16(w0, _mm256_loadu_si256(data.as_ptr().add(i).cast())),
            );
            sum1 = _mm256_add_epi32(
                sum1,
                _mm256_madd_epi16(w1, _mm256_loadu_si256(data.as_ptr().add(i + 16).cast())),
            );
            i += 32;
        }
        if i + 16 <= len {
            let w = _mm256_sub_epi16(
                _mm256_loadu_si256(weights.as_ptr().add(i).cast()),
                _mm256_mullo_epi16(_mm256_loadu_si256(deltas.as_ptr().add(i).cast()), step),
            );
            _mm256_storeu_si256(weights.as_mut_ptr().add(i).cast(), w);
            sum0 = _mm256_add_epi32(
                sum0,
                _mm256_madd_epi16(w, _mm256_loadu_si256(data.as_ptr().add(i).cast())),
            );
            i += 16;
        }
        sum0 = _mm256_add_epi32(sum0, sum1);
        let mut sum = _mm_add_epi32(
            _mm256_castsi256_si128(sum0),
            _mm256_extracti128_si256(sum0, 1),
        );
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
        let mut sum = _mm_cvtsi128_si32(sum);
        for ((w, &d), &x) in weights[i..].iter_mut().zip(&deltas[i..]).zip(&data[i..]) {
            *w = w.wrapping_sub(d.wrapping_mul(amount as i16));
            sum = sum.wrapping_add(*w as i32 * x as i32);
        }
        sum
    }
}

fn adapt_dot_baseline(weights: &mut [i16], deltas: &[i16], data: &[i16], amount: i32) -> i32 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is mandatory on AArch64; the caller supplies equal lengths.
    let (i, mut sum) = unsafe {
        use std::arch::aarch64::*;
        let mut i = 0;
        let step = vdupq_n_s16(amount as i16);
        let mut sum = vdupq_n_s32(0);
        while i + 8 <= weights.len() {
            let w = vsubq_s16(
                vld1q_s16(weights.as_ptr().add(i)),
                vmulq_s16(vld1q_s16(deltas.as_ptr().add(i)), step),
            );
            vst1q_s16(weights.as_mut_ptr().add(i), w);
            let x = vld1q_s16(data.as_ptr().add(i));
            sum = vmlal_s16(sum, vget_low_s16(w), vget_low_s16(x));
            sum = vmlal_high_s16(sum, w, x);
            i += 8;
        }
        (i, vaddvq_s32(sum))
    };
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "x86", target_feature = "sse2")
    ))]
    // SAFETY: SSE2 is enabled for this target and the caller supplies equal
    // lengths. Each vector stays within the slices; the remainder is scalar.
    let (i, mut sum) = unsafe {
        use intrinsics::*;
        let mut i = 0;
        let step = _mm_set1_epi16(amount as i16);
        let mut acc = _mm_setzero_si128();
        while i + 8 <= weights.len() {
            let w = _mm_sub_epi16(
                _mm_loadu_si128(weights.as_ptr().add(i).cast()),
                _mm_mullo_epi16(_mm_loadu_si128(deltas.as_ptr().add(i).cast()), step),
            );
            _mm_storeu_si128(weights.as_mut_ptr().add(i).cast(), w);
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(w, _mm_loadu_si128(data.as_ptr().add(i).cast())),
            );
            i += 8;
        }
        acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 8));
        acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 4));
        (i, _mm_cvtsi128_si32(acc))
    };
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "x86", target_feature = "sse2")
    )))]
    let (i, mut sum) = (0, 0i32);
    for ((w, &d), &x) in weights[i..].iter_mut().zip(&deltas[i..]).zip(&data[i..]) {
        *w = w.wrapping_sub(d.wrapping_mul(amount as i16));
        sum = sum.wrapping_add(*w as i32 * x as i32);
    }
    sum
}

#[inline]
fn limit_weights(weights: &mut [i16], limit: i16) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is mandatory on AArch64; loads/stores stay within weights.
    let i = unsafe {
        use std::arch::aarch64::*;
        let mut i = 0;
        let lo = vdupq_n_s16(-limit);
        let hi = vdupq_n_s16(limit);
        while i + 8 <= weights.len() {
            let w = vld1q_s16(weights.as_ptr().add(i));
            vst1q_s16(weights.as_mut_ptr().add(i), vminq_s16(vmaxq_s16(w, lo), hi));
            i += 8;
        }
        i
    };
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "x86", target_feature = "sse2")
    ))]
    // SAFETY: SSE2 is enabled for this target; every load/store stays in weights.
    let i = unsafe {
        use intrinsics::*;
        let mut i = 0;
        let lo = _mm_set1_epi16(-limit);
        let hi = _mm_set1_epi16(limit);
        while i + 8 <= weights.len() {
            let w = _mm_loadu_si128(weights.as_ptr().add(i).cast());
            _mm_storeu_si128(
                weights.as_mut_ptr().add(i).cast(),
                _mm_min_epi16(_mm_max_epi16(w, lo), hi),
            );
            i += 8;
        }
        i
    };
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "x86", target_feature = "sse2")
    )))]
    let i = 0;
    for w in &mut weights[i..] {
        *w = (*w).clamp(-limit, limit);
    }
}

#[derive(Clone)]
struct MovingSum {
    data: Vec<i32>,
    pos: usize,
    sum: i32,
}
impl MovingSum {
    fn new(len: usize) -> Self {
        Self {
            data: vec![0; len],
            pos: 0,
            sum: 0,
        }
    }
    fn push(&mut self, value: i32) {
        self.sum = self
            .sum
            .wrapping_add(value)
            .wrapping_sub(self.data[self.pos]);
        self.data[self.pos] = value;
        self.pos += 1;
        if self.pos == self.data.len() {
            self.pos = 0;
        }
    }
}

#[derive(Clone)]
struct Simple {
    last: i32,
    older: i32,
    energy: i64,
    count: usize,
    gain: i32,
}
impl Simple {
    fn new() -> Self {
        Self {
            last: 0,
            older: 0,
            energy: 0,
            count: 0,
            gain: 64,
        }
    }
    fn predict(&mut self, sample: i16) -> i32 {
        let sample = sample as i32;
        let error = (self.last * 2 - self.older - sample).abs();
        self.energy = ((self.energy * 8191) >> 13) + error as i64;
        if self.count.is_multiple_of(16) {
            let average = (self.energy + 4096) >> 13;
            self.gain = match average {
                0..=64 => 256,
                65..=128 => 254,
                129..=192 => 250,
                193..=1024 => 248,
                _ => 246,
            };
        }
        self.count += 1;
        self.older = self.last;
        self.last = sample;
        rounded(self.gain * sample, 8)
    }
}

#[derive(Clone)]
struct Short {
    data: History,
    deltas: History,
    weights: Vec<i16>,
    errors: MovingSum,
    prediction: i32,
    count: usize,
}
impl Short {
    fn new(stereo: bool) -> Self {
        let n = if stereo { 32 } else { 16 };
        let mut errors = MovingSum::new(2048);
        for _ in 0..64 {
            errors.push(1024);
        }
        Self {
            data: History::new(n),
            deltas: History::new(n),
            weights: INITIAL_COEFFICIENTS[..n].to_vec(),
            errors,
            prediction: 0,
            count: 0,
        }
    }
    fn predict<const TAPS: usize, const AVX2: bool>(
        &mut self,
        sample: i16,
        other: i16,
        tables: &Tables,
    ) -> i32 {
        let error = clamp16(sample as i32 - self.prediction) as i32;
        let sum = self.errors.sum + 1024;
        let average = sum >> 11;
        self.errors.push(error.abs());
        let index = (error / ((sum >> 13) + 24) + 8192).clamp(0, 16383) as usize;
        let amount = tables.short_gradient[index] as i32 * error.signum();
        self.data.advance();
        self.data.set(0, sample);
        self.data
            .set(3, clamp16(sample as i32 - self.data.get(1) as i32));
        if TAPS == 32 {
            self.data.set(16, other);
            self.data
                .set(19, clamp16(other as i32 - self.data.get(17) as i32));
        }
        self.prediction = rounded(
            adapt_dot::<TAPS, AVX2>(
                &mut self.weights,
                self.deltas.window(),
                self.data.window(),
                amount,
            ),
            14,
        );
        self.deltas.advance();
        let deltas = if TAPS == 32 {
            short_deltas([sample, self.data.get(3), other, self.data.get(19)])
        } else {
            [
                short_delta(sample as i32),
                short_delta(self.data.get(3) as i32),
                0,
                0,
            ]
        };
        self.deltas.set(0, deltas[0]);
        self.deltas.set(3, deltas[1]);
        self.deltas.halve(&[6]);
        if TAPS == 32 {
            self.deltas.set(16, deltas[2]);
            self.deltas.set(19, deltas[3]);
            self.deltas.halve(&[22]);
        }
        if self.count.is_multiple_of(128) {
            let limit = match average {
                i32::MIN..=31 => 6000,
                32..=63 => 8000,
                64..=127 => 10000,
                128..=256 => 12000,
                257..=512 => 14000,
                513..=1024 => 16000,
                _ => 18000,
            };
            limit_weights(&mut self.weights, limit);
        }
        self.count += 1;
        self.prediction
    }
}

#[derive(Clone)]
struct Lms {
    data: History,
    deltas: History,
    weights: Vec<i16>,
    prediction: i32,
    count: usize,
    fast: i32,
    slow: i32,
    rate: i32,
    scale: i32,
    last: [i16; 2],
    cross: usize,
    shift: u32,
}
impl Lms {
    fn new(taps: usize, cross: usize, shift: u32) -> Self {
        Self {
            data: History::new(taps),
            deltas: History::new(taps),
            weights: vec![0; taps],
            prediction: 99999999,
            count: 0,
            fast: 1024,
            slow: 1024,
            rate: 512,
            scale: 16384,
            last: [0; 2],
            cross,
            shift,
        }
    }
    fn sample_delta(&self, sample: i16) -> i16 {
        let x = rounded(self.scale.wrapping_mul(sample as i32), 14).clamp(-65536, 65535);
        delta(x)
    }
    #[inline]
    fn sample_deltas<const AVX2: bool>(&self, samples: [i16; 4]) -> [i16; 4] {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if AVX2 {
            // SAFETY: AVX2=true is selected only after the per-frame CPU/OS check.
            return unsafe { sample_deltas_avx2(samples, self.scale) };
        }
        #[cfg(target_arch = "aarch64")]
        {
            sample_deltas_neon(samples, self.scale)
        }
        #[cfg(not(target_arch = "aarch64"))]
        samples.map(|sample| self.sample_delta(sample))
    }
    #[inline(always)]
    fn predict_with<const TAPS: usize, const CROSS: usize, const SHIFT: u32, const AVX2: bool>(
        &mut self,
        sample: i16,
        other: i16,
        tables: &Tables,
    ) -> i32 {
        let taps = TAPS;
        let cross = CROSS;
        let shift = SHIFT;
        let mut amount = 0;
        if self.count != 0 {
            let error = (sample as i32).wrapping_sub(self.prediction);
            self.fast = (self.fast.wrapping_mul(63) >> 6).wrapping_add(error.wrapping_abs());
            self.slow = (self.slow.wrapping_mul(1023) >> 10).wrapping_add(error.wrapping_abs());
            if self.count.is_multiple_of(4) {
                let average = rounded(self.fast, 6).clamp(0, 65535);
                self.rate = tables.learning_rate[average as usize] as i32;
                if self.count.is_multiple_of(256) {
                    let average = rounded(self.slow, 10);
                    self.scale = crate::tables::sample_scale(average);
                    limit_weights(&mut self.weights, 24000);
                }
            }
            let index =
                (rounded(self.rate.wrapping_mul(error), 15) + 16384).clamp(0, 32767) as usize;
            if error != 0 {
                amount = tables.gradient[index] as i32 * if error < 0 { -1 } else { 1 };
            }
            self.data.advance();
            self.data.set(0, sample);
            if cross != 0 {
                self.data.set(cross, other);
            }
            if taps <= 32 {
                self.data
                    .set(2, clamp16(sample as i32 - self.last[0] as i32));
                if cross != 0 {
                    self.data
                        .set(cross + 2, clamp16(other as i32 - self.last[1] as i32));
                }
            } else {
                if taps == 96 || taps == 192 {
                    self.data.halve(&[40]);
                }
                if cross != 0 {
                    self.data.halve(&[cross + 40]);
                }
            }
        }
        self.prediction = rounded(
            adapt_dot::<TAPS, AVX2>(
                &mut self.weights,
                self.deltas.window(),
                self.data.window(),
                amount,
            ),
            shift,
        );
        if self.count != 0 {
            self.deltas.advance();
            if taps <= 32 {
                if cross != 0 {
                    let d = self.sample_deltas::<AVX2>([
                        sample,
                        self.data.get(2),
                        other,
                        self.data.get(cross + 2),
                    ]);
                    self.deltas.set(0, d[0] >> 2);
                    self.deltas.set(2, d[1] >> 2);
                    self.deltas.set(cross, d[2] >> 2);
                    self.deltas.set(cross + 2, d[3] >> 2);
                } else {
                    let d = self.sample_deltas::<AVX2>([sample, self.data.get(2), 0, 0]);
                    self.deltas.set(0, d[0] >> 2);
                    self.deltas.set(2, d[1] >> 2);
                }
                self.deltas.halve(&[4, 6]);
                if cross != 0 {
                    self.deltas.halve(&[cross + 4, cross + 6]);
                }
            } else {
                if cross != 0 {
                    let d = self.sample_deltas::<AVX2>([sample, other, 0, 0]);
                    self.deltas.set(0, d[0]);
                    self.deltas.set(cross, d[1]);
                } else {
                    self.deltas.set(0, self.sample_delta(sample));
                }
                self.deltas.halve(&[2, 4, 8, 20]);
                if cross != 0 {
                    self.deltas
                        .halve(&[cross + 2, cross + 4, cross + 8, cross + 20]);
                }
            }
        }
        self.count += 1;
        self.last = [sample, other];
        self.prediction
    }
}

#[derive(Clone)]
struct HighHistory {
    phases: [History; 4],
    phase: usize,
    sum: MovingSum,
}
impl HighHistory {
    fn new() -> Self {
        Self {
            phases: std::array::from_fn(|_| History::new(288)),
            phase: 0,
            sum: MovingSum::new(4),
        }
    }
    fn push(&mut self, value: i16) {
        self.sum.push(value as i32);
        self.phase = (self.phase + 1) % 4;
        let history = &mut self.phases[self.phase];
        history.advance();
        history.set(0, clamp16(self.sum.sum));
        history.halve(&[1, 2]);
    }
    fn window(&self) -> &[i16] {
        self.phases[self.phase].window()
    }
}
#[derive(Clone)]
struct High {
    data: HighHistory,
    deltas: HighHistory,
    weights: Vec<i16>,
    prediction: i32,
    count: usize,
    fast: i32,
}
impl High {
    fn new() -> Self {
        Self {
            data: HighHistory::new(),
            deltas: HighHistory::new(),
            weights: vec![0; 288],
            prediction: 99999999,
            count: 0,
            fast: 0,
        }
    }
    fn predict<const AVX2: bool>(&mut self, sample: i16, tables: &Tables) -> i32 {
        let mut amount = 0;
        if self.count != 0 {
            let error = sample as i32 - self.prediction;
            self.fast = (self.fast.wrapping_mul(63) >> 6).wrapping_add(error.wrapping_abs());
            let rate = tables.learning_rate[rounded(self.fast, 6).clamp(0, 65535) as usize] as i32;
            let index = (rounded(rate.wrapping_mul(error), 15) + 16384).clamp(0, 32767) as usize;
            amount = tables.gradient[index] as i32 * if error < 0 { -1 } else { 1 };
            self.data.push(sample);
        }
        self.prediction = rounded(
            adapt_dot::<288, AVX2>(
                &mut self.weights,
                self.deltas.window(),
                self.data.window(),
                amount,
            ) >> 5,
            10,
        );
        if self.count != 0 {
            self.deltas.push(high_delta(sample));
        }
        self.count += 1;
        self.prediction
    }
}

#[derive(Clone)]
enum Predictors {
    Simple(Vec<Simple>),
    Short(Vec<Short>),
    Lms(Vec<Lms>),
    High(Vec<High>),
}

#[derive(Clone, Copy)]
enum StageKind {
    Direct,
    Long,
    Intermediate,
    High,
    Final,
}

#[derive(Clone)]
pub(crate) struct Stage {
    predictors: Predictors,
    state: StageState,
}

#[derive(Clone)]
struct StageState {
    previous: i16,
    samples: [i16; 2],
    count: usize,
    prediction: i32,
    gains: [i32; 2],
    attenuation: i32,
    errors: MovingSum,
    gain_shift: u32,
    attenuation_shift: u32,
    minimum_error: i32,
}
impl StageState {
    fn new(kind: StageKind) -> Self {
        let gain_shift = match kind {
            StageKind::Direct => 0,
            StageKind::Long | StageKind::Intermediate => 9,
            StageKind::High | StageKind::Final => 8,
        };
        let attenuation_shift = match kind {
            StageKind::Long => 6,
            StageKind::Intermediate | StageKind::High => 7,
            _ => 0,
        };
        let gain = 1 << gain_shift;
        let attenuation = 1 << attenuation_shift;
        Self {
            previous: 0,
            samples: [0; 2],
            count: 0,
            prediction: 0,
            gains: [gain; 2],
            attenuation,
            errors: MovingSum::new(4096),
            gain_shift,
            attenuation_shift,
            minimum_error: if matches!(kind, StageKind::Long) {
                4
            } else {
                0
            },
        }
    }
    #[inline(always)]
    fn process_sample<const DECODE: bool, P, const CHANNELS: usize>(
        &mut self,
        sample: i16,
        predictors: &mut [P; CHANNELS],
        predict: &mut impl FnMut(&mut P, i16, i16) -> i32,
    ) -> i16 {
        let prev = self.previous;
        let stereo = CHANNELS == 2;
        if stereo {
            self.count += 1;
        }
        let odd = self.count & 1;
        let lane = if stereo { odd } else { 0 };
        let error = prev as i32 - self.prediction;
        let magnitude = error.abs();
        if self.gain_shift != 0 && magnitude > self.minimum_error && self.prediction != 0 {
            let g = if stereo { 1 - lane } else { 0 };
            let step = (magnitude >> 9) + 1;
            self.gains[g] = (self.gains[g]
                + if (error ^ self.prediction) < 0 {
                    -step
                } else {
                    step
                })
            .clamp(1, 1 << self.gain_shift);
        }
        let p = if stereo {
            self.samples[1 - lane] = prev;
            predict(&mut predictors[lane], self.samples[lane], prev)
        } else {
            predict(&mut predictors[0], prev, 0)
        };
        self.prediction = if self.gain_shift == 0 {
            p
        } else {
            rounded(p.wrapping_mul(self.gains[lane]), self.gain_shift)
        };
        let prediction = if self.attenuation_shift != 0 {
            if self.count.is_multiple_of(16) {
                self.errors.push(magnitude);
                let average = self.errors.sum >> 12;
                let reduction = [64, 128, 192, 256, 378, 512]
                    .iter()
                    .filter(|&&t| average > t)
                    .count() as i32;
                self.attenuation = (1 << self.attenuation_shift) - reduction;
            }
            rounded(
                self.attenuation.wrapping_mul(self.prediction),
                self.attenuation_shift,
            )
        } else {
            self.prediction
        };
        let result = (prediction as i16).wrapping_sub(sample);
        self.previous = if DECODE { result } else { sample };
        result
    }
    #[inline(always)]
    fn process_with<const DECODE: bool, P>(
        &mut self,
        samples: &mut [i16],
        predictors: &mut [P],
        predict: impl FnMut(&mut P, i16, i16) -> i32,
    ) {
        match predictors.len() {
            1 => self.process_channels::<DECODE, P, 1>(
                samples,
                predictors.try_into().unwrap(),
                predict,
            ),
            2 => self.process_channels::<DECODE, P, 2>(
                samples,
                predictors.try_into().unwrap(),
                predict,
            ),
            _ => unreachable!("LA04 supports one or two channels"),
        }
    }
    #[inline(always)]
    fn process_channels<const DECODE: bool, P, const CHANNELS: usize>(
        &mut self,
        samples: &mut [i16],
        predictors: &mut [P; CHANNELS],
        mut predict: impl FnMut(&mut P, i16, i16) -> i32,
    ) {
        for s in samples {
            *s = self.process_sample::<DECODE, P, CHANNELS>(*s, predictors, &mut predict);
        }
    }
}

impl Stage {
    fn new(predictors: Predictors, kind: StageKind) -> Self {
        Self {
            predictors,
            state: StageState::new(kind),
        }
    }
    pub(crate) fn encode(&mut self, samples: &mut [i16]) {
        self.process::<false>(samples);
    }
    fn decode(&mut self, samples: &mut [i16]) {
        self.process::<true>(samples);
    }
    fn process<const DECODE: bool>(&mut self, samples: &mut [i16]) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if has_avx2() {
            // SAFETY: the CPU and OS support AVX2. Dispatch is outside the sample loop.
            unsafe {
                self.process_avx2::<DECODE>(samples);
            }
            return;
        }
        self.process_with::<DECODE, false>(samples);
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn process_avx2<const DECODE: bool>(&mut self, samples: &mut [i16]) {
        self.process_with::<DECODE, true>(samples);
    }
    #[inline(always)]
    fn process_with<const DECODE: bool, const AVX2: bool>(&mut self, samples: &mut [i16]) {
        let tables = tables();
        // Every channel of a stage uses the same filter. Select it once for
        // the block so the sample loop has a concrete predictor type.
        match &mut self.predictors {
            Predictors::Simple(p) => {
                self.state
                    .process_with::<DECODE, _>(samples, p, |p, sample, _| p.predict(sample))
            }
            Predictors::Short(p) => {
                if p.len() == 2 {
                    self.state
                        .process_with::<DECODE, _>(samples, p, |p, sample, other| {
                            p.predict::<32, AVX2>(sample, other, tables)
                        });
                } else {
                    self.state
                        .process_with::<DECODE, _>(samples, p, |p, sample, other| {
                            p.predict::<16, AVX2>(sample, other, tables)
                        });
                }
            }
            Predictors::Lms(p) => match (p[0].weights.len(), p[0].cross, p[0].shift) {
                (512, 0, 16) => {
                    self.state
                        .process_with::<DECODE, _>(samples, p, |p, sample, other| {
                            p.predict_with::<512, 0, 16, AVX2>(sample, other, tables)
                        })
                }
                (16, 0, 16) => {
                    self.state
                        .process_with::<DECODE, _>(samples, p, |p, sample, other| {
                            p.predict_with::<16, 0, 16, AVX2>(sample, other, tables)
                        })
                }
                (32, 16, 16) => {
                    self.state
                        .process_with::<DECODE, _>(samples, p, |p, sample, other| {
                            p.predict_with::<32, 16, 16, AVX2>(sample, other, tables)
                        })
                }
                (96, 0, 13) => {
                    self.state
                        .process_with::<DECODE, _>(samples, p, |p, sample, other| {
                            p.predict_with::<96, 0, 13, AVX2>(sample, other, tables)
                        })
                }
                (192, 96, 14) => {
                    self.state
                        .process_with::<DECODE, _>(samples, p, |p, sample, other| {
                            p.predict_with::<192, 96, 14, AVX2>(sample, other, tables)
                        })
                }
                (16, 0, 11) => {
                    self.state
                        .process_with::<DECODE, _>(samples, p, |p, sample, other| {
                            p.predict_with::<16, 0, 11, AVX2>(sample, other, tables)
                        })
                }
                (32, 16, 11) => {
                    self.state
                        .process_with::<DECODE, _>(samples, p, |p, sample, other| {
                            p.predict_with::<32, 16, 11, AVX2>(sample, other, tables)
                        })
                }
                _ => unreachable!("unsupported LA04 LMS shape"),
            },
            Predictors::High(p) => {
                self.state
                    .process_with::<DECODE, _>(samples, p, |p, sample, _| {
                        p.predict::<AVX2>(sample, tables)
                    })
            }
        }
    }
}

#[derive(Clone)]
pub struct Pipeline {
    stages: Vec<Stage>,
    stereo: bool,
    correlation: i32,
}

pub(crate) fn decorrelate(samples: &mut [i16], correlation: &mut i32) {
    for frame in samples.as_chunks_mut::<2>().0 {
        let [left, right] = [frame[0], frame[1]];
        if *correlation >= 0 {
            frame[1] = left.wrapping_sub(right);
        }
        *correlation = (correlation.wrapping_mul(255) >> 8) + if left == right { 1 } else { -1 };
    }
}

impl Pipeline {
    pub fn new(channels: usize, high: bool) -> Self {
        let stereo = channels == 2;
        let mut stages = Vec::new();
        let mut add = |kind, predictors| stages.push(Stage::new(predictors, kind));
        add(
            StageKind::Direct,
            Predictors::Simple(vec![Simple::new(); channels]),
        );
        add(
            StageKind::Direct,
            Predictors::Short(vec![Short::new(stereo); channels]),
        );
        add(
            StageKind::Long,
            Predictors::Lms(vec![Lms::new(512, 0, 16); channels]),
        );
        if high {
            add(
                StageKind::Intermediate,
                Predictors::Lms(vec![
                    Lms::new(
                        if stereo { 32 } else { 16 },
                        if stereo { 16 } else { 0 },
                        16,
                    );
                    channels
                ]),
            );
            add(
                StageKind::High,
                Predictors::High(vec![High::new(); channels]),
            );
        }
        add(
            StageKind::Intermediate,
            Predictors::Lms(vec![
                Lms::new(
                    if stereo { 192 } else { 96 },
                    if stereo { 96 } else { 0 },
                    if stereo { 14 } else { 13 },
                );
                channels
            ]),
        );
        add(
            StageKind::Final,
            Predictors::Lms(vec![
                Lms::new(
                    if stereo { 32 } else { 16 },
                    if stereo { 16 } else { 0 },
                    11,
                );
                channels
            ]),
        );
        Self {
            stages,
            stereo,
            correlation: 0,
        }
    }
    pub fn encode(&mut self, samples: &mut [i16], mut observe: impl FnMut(usize, &[i16])) {
        let mut stage_index = 0;
        if self.stereo {
            decorrelate(samples, &mut self.correlation);
            observe(stage_index, samples);
            stage_index += 1;
        }
        for stage in &mut self.stages {
            stage.encode(samples);
            observe(stage_index, samples);
            stage_index += 1;
        }
    }
    pub(crate) fn into_stages(self) -> Vec<Stage> {
        self.stages
    }
    pub fn decode(&mut self, samples: &mut [i16]) {
        for stage in self.stages.iter_mut().rev() {
            stage.decode(samples);
        }
        if self.stereo {
            for frame in samples.as_chunks_mut::<2>().0 {
                if self.correlation >= 0 {
                    frame[1] = frame[0].wrapping_sub(frame[1]);
                }
                self.correlation = (self.correlation.wrapping_mul(255) >> 8)
                    + if frame[0] == frame[1] { 1 } else { -1 };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_short_deltas_match_scalar_for_all_samples() {
        let values: Vec<_> = (i16::MIN..=i16::MAX).collect();
        for &samples in values.as_chunks::<4>().0 {
            assert_eq!(
                short_deltas(samples),
                samples.map(|sample| short_delta(sample as i32))
            );
        }
    }

    #[test]
    fn vector_sample_deltas_match_scalar_for_all_samples_and_wrapping_scales() {
        let values: Vec<_> = (i16::MIN..=i16::MAX).collect();
        let mut predictor = Lms::new(32, 16, 11);
        for scale in [
            i32::MIN,
            -65537,
            -16384,
            -1,
            0,
            1,
            4096,
            16384,
            32768,
            65536,
            393216,
            i32::MAX,
        ] {
            predictor.scale = scale;
            for &samples in values.as_chunks::<4>().0 {
                let expected = samples.map(|sample| predictor.sample_delta(sample));
                assert_eq!(predictor.sample_deltas::<false>(samples), expected);
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if has_avx2() {
                    assert_eq!(
                        predictor.sample_deltas::<true>(samples),
                        expected,
                        "scale {scale}, samples {samples:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn specialized_dot_and_adaptation_match_wrapping_scalar() {
        fn check<const TAPS: usize>() {
            let mut state = 123u32;
            let values: Vec<_> = (0..TAPS + 16)
                .map(|i| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    if i % 3 == 0 {
                        i16::MIN
                    } else {
                        (state >> 16) as i16
                    }
                })
                .collect();
            for offset in 0..8 {
                let data = &values[offset..offset + TAPS];
                let deltas = &values[offset + 1..offset + TAPS + 1];
                for amount in [i32::MIN, -32768, -193, -1, 0, 1, 193, 32767, i32::MAX] {
                    let original = vec![i16::MIN; TAPS + 2];
                    let mut expected = original.clone();
                    let mut sum = 0i32;
                    for ((w, &d), &x) in expected[1..=TAPS].iter_mut().zip(deltas).zip(data) {
                        *w = w.wrapping_sub(d.wrapping_mul(amount as i16));
                        sum = sum.wrapping_add(*w as i32 * x as i32);
                    }
                    let mut actual = original.clone();
                    assert_eq!(
                        adapt_dot::<TAPS, false>(&mut actual[1..=TAPS], deltas, data, amount),
                        sum
                    );
                    assert_eq!(actual, expected);
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if has_avx2() {
                        let mut actual = original.clone();
                        assert_eq!(
                            adapt_dot::<TAPS, true>(&mut actual[1..=TAPS], deltas, data, amount),
                            sum
                        );
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
        check::<16>();
        check::<32>();
        check::<96>();
        check::<192>();
        check::<288>();
        check::<512>();
    }

    #[test]
    fn vector_limits_match_scalar_with_unaligned_tails() {
        let values: Vec<_> = (i16::MIN..=i16::MAX).collect();
        for limit in [
            0, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 24000, 32767,
        ] {
            for offset in 0..8 {
                let mut actual = values.clone();
                let mut expected = values.clone();
                let end = values.len() - offset;
                for w in &mut expected[offset..end] {
                    *w = (*w).clamp(-limit, limit);
                }
                limit_weights(&mut actual[offset..end], limit);
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx2_and_baseline_stages_match_across_history_wraps_and_clones() {
        if !has_avx2() {
            return;
        }
        let mut state = 123u32;
        let samples: Vec<_> = (0..16384)
            .map(|i| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                match i % 8 {
                    0 => i16::MIN,
                    1 => i16::MAX,
                    2 => 0,
                    _ => (state >> 16) as i16,
                }
            })
            .collect();
        for channels in [1, 2] {
            for high in [false, true] {
                let initial = Pipeline::new(channels, high).into_stages();
                let mut baseline = initial.clone();
                let mut avx2 = initial.clone();
                let mut encoded = samples.clone();
                for (index, chunk) in encoded.chunks_mut(2048).enumerate() {
                    let mut expected = chunk.to_vec();
                    for (scalar, vector) in baseline.iter_mut().zip(&mut avx2) {
                        scalar.process_with::<false, false>(&mut expected);
                        // SAFETY: the test checked CPU/OS support above.
                        unsafe { vector.process_avx2::<false>(chunk) };
                        assert_eq!(chunk, expected, "encode {channels}/{high}/{index}");
                    }
                    if index == 3 {
                        baseline = baseline.clone();
                        avx2 = avx2.clone();
                    }
                }
                let mut baseline = initial.clone();
                let mut avx2 = initial;
                for (index, chunk) in encoded.chunks_mut(2048).enumerate() {
                    let mut expected = chunk.to_vec();
                    for (scalar, vector) in baseline.iter_mut().zip(&mut avx2).rev() {
                        scalar.process_with::<true, false>(&mut expected);
                        // SAFETY: the test checked CPU/OS support above.
                        unsafe { vector.process_avx2::<true>(chunk) };
                        assert_eq!(chunk, expected, "decode {channels}/{high}/{index}");
                    }
                    if index == 3 {
                        baseline = baseline.clone();
                        avx2 = avx2.clone();
                    }
                }
                assert_eq!(encoded, samples);
            }
        }
    }

    #[test]
    fn vector_dot_matches_wrapping_scalar() {
        let mut state = 123u32;
        let mut values = vec![i16::MIN; 1024];
        values.extend((0..1024).map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as i16
        }));
        for start in [0, 1, 511, 1000] {
            for len in [0, 1, 7, 8, 15, 16, 31, 32, 96, 192, 288, 512, 1025] {
                let a = &values[start..start + len];
                let b = &values[start + 1..start + len + 1];
                let scalar = a
                    .iter()
                    .zip(b)
                    .fold(0i32, |s, (&x, &y)| s.wrapping_add(x as i32 * y as i32));
                assert_eq!(
                    dot::<0, false>(a, b),
                    scalar,
                    "offset {start}, length {len}"
                );
                assert_eq!(dot_baseline(a, b), scalar);
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if std::arch::is_x86_feature_detected!("avx2") {
                    assert_eq!(unsafe { dot_avx2::<0>(a, b) }, scalar);
                }
                let shorter = &b[..len / 2];
                let scalar = a
                    .iter()
                    .zip(shorter)
                    .fold(0i32, |s, (&x, &y)| s.wrapping_add(x as i32 * y as i32));
                assert_eq!(dot::<0, false>(a, shorter), scalar);
                assert_eq!(dot::<0, false>(shorter, a), scalar);
            }
        }
    }

    #[test]
    fn fused_adaptation_matches_separate_wrapping_operations() {
        let mut state = 123u32;
        let history: Vec<_> = (0..1030)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 16) as i16
            })
            .collect();
        for offset in [0, 1, 3] {
            for len in [
                0, 1, 7, 8, 15, 16, 31, 32, 95, 96, 97, 191, 192, 288, 512, 1025,
            ] {
                for amount in [
                    i32::MIN,
                    -65537,
                    -32768,
                    -193,
                    0,
                    193,
                    32767,
                    65536,
                    i32::MAX,
                ] {
                    let deltas = &history[offset..offset + len];
                    let data = &history[offset + 1..offset + len + 1];
                    let original = vec![i16::MIN; len + 2];
                    let mut expected = original.clone();
                    for (w, &d) in expected[1..=len].iter_mut().zip(deltas) {
                        *w = w.wrapping_sub(d.wrapping_mul(amount as i16));
                    }
                    let sum = expected[1..=len]
                        .iter()
                        .zip(data)
                        .fold(0i32, |s, (&w, &x)| s.wrapping_add(w as i32 * x as i32));
                    let mut actual = original.clone();
                    assert_eq!(
                        adapt_dot::<0, false>(&mut actual[1..=len], deltas, data, amount),
                        sum
                    );
                    assert_eq!(actual, expected);
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if has_avx2() {
                        let mut actual = original.clone();
                        assert_eq!(
                            adapt_dot::<0, true>(&mut actual[1..=len], deltas, data, amount),
                            sum
                        );
                        assert_eq!(actual, expected);
                        let mut actual = original.clone();
                        assert_eq!(
                            unsafe {
                                adapt_dot_avx2::<0>(&mut actual[1..=len], deltas, data, amount)
                            },
                            sum
                        );
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }
}
