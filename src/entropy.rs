use crate::constants::SMALL_COUNTS;

#[derive(Clone)]
pub struct RangeEncoder {
    low: u32,
    range: u32,
    pub bytes: Vec<u8>,
    count: Option<usize>,
}
impl Default for RangeEncoder {
    fn default() -> Self {
        Self {
            low: 0,
            range: u32::MAX,
            bytes: Vec::new(),
            count: None,
        }
    }
}
impl RangeEncoder {
    fn byte(&mut self, byte: u8) {
        if let Some(count) = &mut self.count {
            *count += 1;
        } else {
            self.bytes.push(byte);
        }
    }
    fn interval(&mut self, lower: u32, upper: u32, unit: u32) {
        self.low = self.low.wrapping_add(unit.wrapping_mul(lower));
        self.range = unit.wrapping_mul(upper - lower);
        loop {
            if (self.low.wrapping_add(self.range) ^ self.low) > 0xffffff {
                if self.range > 0xffff {
                    break;
                }
                self.range = self.low.wrapping_neg() & 0xffff;
            }
            self.byte((self.low >> 24) as u8);
            self.range <<= 8;
            self.low <<= 8;
        }
    }
    pub fn encode(&mut self, lower: u32, upper: u32, total: u32) {
        debug_assert!(lower < upper && upper <= total);
        self.interval(lower, upper, self.range / total);
    }
    pub fn bits(&mut self, value: u32, count: u32) {
        self.interval(value, value + 1, self.range >> count);
    }
    pub fn finish(&mut self) {
        for _ in 0..4 {
            self.byte((self.low >> 24) as u8);
            self.low <<= 8;
        }
    }
}

fn analyze_value<const RATE: u32>(state: &mut i32, value: u16) -> [u16; 2] {
    let divisor = (*state >> 6).max(1) as u32;
    let delta = ((value as i32 + 6) >> 1) - ((*state + 16) >> 5);
    *state += match RATE {
        1 => delta >> 1,
        2 => delta,
        3 => delta >> 2,
        4 => delta * 2,
        _ => unreachable!(),
    };
    let quotient = value as u32 / divisor;

    debug_assert!(divisor <= u16::MAX as u32 && quotient <= u16::MAX as u32);
    [divisor as u16, quotient as u16]
}

// Reverse cumulative counts make each interval a pair of array lookups.
// Consuming bin n only changes entries 0..=n (the common bins are small).
struct Histogram {
    tails: [u32; 193],
}
impl Histogram {
    fn new(counts: &[u32; 192]) -> Self {
        let mut tails = [0; 193];
        for i in (0..192).rev() {
            tails[i] = tails[i + 1] + counts[i];
        }
        Self { tails }
    }
    #[inline]
    fn interval(&self, bin: usize) -> (u32, u32) {
        let total = self.tails[0];
        (total - self.tails[bin], total - self.tails[bin + 1])
    }
    #[inline]
    fn find(&self, target: u32) -> usize {
        let threshold = self.tails[0] - target - 1;
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is mandatory on AArch64; all four lanes stay within tails.
        let i = unsafe {
            use std::arch::aarch64::*;
            let mut i = 0;
            let threshold = vdupq_n_u32(threshold);
            while i + 4 <= 192 {
                let found = vcleq_u32(vld1q_u32(self.tails.as_ptr().add(i + 1)), threshold);
                if vmaxvq_u32(found) != 0 {
                    // A sorted cumulative table has a suffix of matching lanes.
                    return i + (4 - vaddvq_u32(vshrq_n_u32(found, 31))) as usize;
                }
                i += 4;
            }
            i
        };
        #[cfg(any(
            target_arch = "x86_64",
            all(target_arch = "x86", target_feature = "sse2")
        ))]
        // SAFETY: SSE2 is enabled for this target. The table has 193 entries,
        // and all counts/thresholds fit in i32 (a frame has at most 61440 samples).
        let i = unsafe {
            let mut i = 0;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            let threshold = _mm_set1_epi32(threshold as i32);
            while i + 4 <= 192 {
                let tails = _mm_loadu_si128(self.tails.as_ptr().add(i + 1).cast());
                let mask =
                    _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(tails, threshold))) as u32;
                if mask != 15 {
                    return i + ((!mask & 15).trailing_zeros() as usize);
                }
                i += 4;
            }
            i
        };
        #[cfg(not(any(
            target_arch = "aarch64",
            target_arch = "x86_64",
            all(target_arch = "x86", target_feature = "sse2")
        )))]
        let i = 0;
        i + self.tails[i + 1..]
            .iter()
            .position(|&tail| tail <= threshold)
            .unwrap()
    }
    #[inline]
    fn consume(&mut self, bin: usize) {
        let len = bin + 1;
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is mandatory on AArch64. bin < 192, so even the
        // masked final four-counter load/store stays inside the 193-entry table.
        unsafe {
            use std::arch::aarch64::*;
            let mut i = 0;
            while i + 4 <= len {
                let v = vld1q_u32(self.tails.as_ptr().add(i));
                vst1q_u32(self.tails.as_mut_ptr().add(i), vsubq_u32(v, vdupq_n_u32(1)));
                i += 4;
            }
            if i < len {
                let lanes = [0, 1, 2, 3];
                let mask = vcltq_u32(vld1q_u32(lanes.as_ptr()), vdupq_n_u32((len - i) as u32));
                let v = vld1q_u32(self.tails.as_ptr().add(i));
                vst1q_u32(self.tails.as_mut_ptr().add(i), vaddq_u32(v, mask));
            }
        }
        #[cfg(any(
            target_arch = "x86_64",
            all(target_arch = "x86", target_feature = "sse2")
        ))]
        // SAFETY: SSE2 is enabled for this target. bin < 192; the masked
        // final load/store stays within tails and leaves unused lanes unchanged.
        unsafe {
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            let mut i = 0;
            let one = _mm_set1_epi32(1);
            while i + 4 <= len {
                let v = _mm_loadu_si128(self.tails.as_ptr().add(i).cast());
                _mm_storeu_si128(self.tails.as_mut_ptr().add(i).cast(), _mm_sub_epi32(v, one));
                i += 4;
            }
            if i < len {
                let mask =
                    _mm_cmpgt_epi32(_mm_set1_epi32((len - i) as i32), _mm_setr_epi32(0, 1, 2, 3));
                let v = _mm_loadu_si128(self.tails.as_ptr().add(i).cast());
                _mm_storeu_si128(
                    self.tails.as_mut_ptr().add(i).cast(),
                    _mm_add_epi32(v, mask),
                );
            }
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            target_arch = "x86_64",
            all(target_arch = "x86", target_feature = "sse2")
        )))]
        for tail in &mut self.tails[..len] {
            *tail -= 1;
        }
    }
}

pub struct EntropyEncoder {
    pub coder: RangeEncoder,
    state: [i32; 2],
    channels: usize,
}
impl EntropyEncoder {
    pub fn new(channels: usize) -> Self {
        Self {
            coder: RangeEncoder::default(),
            state: [0x4000; 2],
            channels,
        }
    }

    fn analyze(&mut self, values: &[u16], rate: u32) -> ([u32; 192], Vec<[u16; 2]>) {
        match (self.channels, rate) {
            (1, 1) => self.analyze_with::<1, 1>(values),
            (1, 2) => self.analyze_with::<1, 2>(values),
            (1, 3) => self.analyze_with::<1, 3>(values),
            (1, 4) => self.analyze_with::<1, 4>(values),
            (2, 1) => self.analyze_with::<2, 1>(values),
            (2, 2) => self.analyze_with::<2, 2>(values),
            (2, 3) => self.analyze_with::<2, 3>(values),
            (2, 4) => self.analyze_with::<2, 4>(values),
            _ => unreachable!(),
        }
    }
    fn analyze_with<const CHANNELS: usize, const RATE: u32>(
        &mut self,
        values: &[u16],
    ) -> ([u32; 192], Vec<[u16; 2]>) {
        let mut codes = vec![[0; 2]; values.len()];
        let mut counts = [0; 192];
        let mut states = self.state;
        let (input, input_remainder) = values.as_chunks::<CHANNELS>();
        let (output, output_remainder) = codes.as_chunks_mut::<CHANNELS>();
        for (input, output) in input.iter().zip(output.iter_mut()) {
            for lane in 0..CHANNELS {
                let code = analyze_value::<RATE>(&mut states[lane], input[lane]);
                counts[code[1].min(191) as usize] += 1;
                output[lane] = code;
            }
        }
        for (lane, (&value, code)) in input_remainder.iter().zip(output_remainder).enumerate() {
            *code = analyze_value::<RATE>(&mut states[lane], value);
            counts[code[1].min(191) as usize] += 1;
        }
        self.state = states;
        (counts, codes)
    }

    fn encode_values(&mut self, values: &[u16], rate: u32) {
        let (counts, codes) = self.analyze(values, rate);
        self.coder.bits(rate - 1, 2);
        let mut remaining = values.len() as u32;
        for &count in &counts {
            if remaining == 0 {
                break;
            }
            if remaining < 32 {
                let n = remaining as usize;

                let start = (n - 1) * (n + 2) / 2;
                let weights = &SMALL_COUNTS[start..start + n + 1];
                let lower = weights[..count as usize].iter().sum();
                let upper = lower + weights[count as usize];
                self.coder.encode(lower, upper, weights.iter().sum());
            } else {
                self.coder.encode(count, count + 1, remaining + 1);
            }
            remaining -= count;
            if remaining == 0 {
                break;
            }
        }
        remaining = values.len() as u32;
        let mut histogram = Histogram::new(&counts);
        for (&value, &[divisor, quotient]) in values.iter().zip(&codes) {
            let divisor = divisor as u32;
            let quotient = quotient as u32;
            let bin = quotient.min(191) as usize;
            let (lower, upper) = histogram.interval(bin);
            self.coder.encode(lower, upper, remaining);
            let (remainder, scale) = if quotient < 191 {
                (value as u32 - quotient * divisor, divisor)
            } else {
                let bits = 32 - (value as u32).leading_zeros();
                self.coder.bits(bits, 5);
                (value as u32, 1 << bits)
            };
            histogram.consume(bin);
            remaining -= 1;
            if scale > 1 {
                self.coder.encode(remainder, remainder + 1, scale);
            }
        }
    }

    fn select_rate(&self, values: &[u16]) -> u32 {
        let len = (values.len() / 2) & !3;
        let start = len / 2;
        let values = &values[start..start + len];
        let mut best = (usize::MAX, 1);
        for rate in 1..=4 {
            let mut trial = Self::new(self.channels);
            trial.state = self.state;
            trial.coder.count = Some(0);
            trial.encode_values(values, rate);
            trial.coder.finish();
            let size = trial.coder.count.unwrap();
            if size < best.0 {
                best = (size, rate);
            }
        }
        best.1
    }

    pub fn encode_frame(&mut self, samples: &[i16]) {
        let values: Vec<u16> = samples
            .iter()
            .map(|&s| {
                if s < 0 {
                    (-(s as i32) * 2 - 1) as u16
                } else {
                    (s as u16) * 2
                }
            })
            .collect();
        let silent: Vec<bool> = (0..self.channels)
            .map(|c| {
                values
                    .iter()
                    .skip(c)
                    .step_by(self.channels)
                    .all(|&v| v == 0)
            })
            .collect();
        for &flag in &silent {
            self.coder.bits(flag as u32, 1);
        }
        if silent.iter().all(|&s| s) {
            return;
        }
        let original_channels = self.channels;
        let filtered;
        let values = if self.channels == 2 && silent.iter().any(|&s| s) {
            filtered = values
                .iter()
                .skip(silent[0] as usize)
                .step_by(2)
                .copied()
                .collect::<Vec<_>>();
            self.channels = 1;
            &filtered
        } else {
            &values
        };
        let rate = self.select_rate(values);
        self.encode_values(values, rate);
        self.channels = original_channels;
    }
}

struct RangeDecoder<R> {
    input: R,
    low: u32,
    range: u32,
    code: u32,
    bytes_read: u64,
}
impl<R: std::io::Read> RangeDecoder<R> {
    fn new(input: R) -> std::io::Result<Self> {
        let mut coder = Self {
            input,
            low: 0,
            range: u32::MAX,
            code: 0,
            bytes_read: 0,
        };
        for _ in 0..4 {
            coder.code = (coder.code << 8) | coder.byte()? as u32;
        }
        Ok(coder)
    }
    fn byte(&mut self) -> std::io::Result<u8> {
        let mut byte = [0];
        self.input.read_exact(&mut byte)?;
        self.bytes_read += 1;
        Ok(byte[0])
    }

    fn value(&self, total: u32) -> std::io::Result<(u32, u32)> {
        let unit = self
            .range
            .checked_div(total)
            .filter(|&u| u != 0)
            .ok_or_else(|| crate::decoder::invalid("invalid LA04 range interval"))?;
        let value = self.code.wrapping_sub(self.low) / unit;
        if value >= total {
            return Err(crate::decoder::invalid("LA04 range value exceeds model"));
        }
        Ok((value, unit))
    }
    fn interval(&mut self, lower: u32, upper: u32, unit: u32) -> std::io::Result<()> {
        self.low = self.low.wrapping_add(unit.wrapping_mul(lower));
        self.range = unit.wrapping_mul(upper - lower);
        loop {
            if (self.low.wrapping_add(self.range) ^ self.low) > 0xffffff {
                if self.range > 0xffff {
                    break;
                }
                self.range = self.low.wrapping_neg() & 0xffff;
            }
            self.low <<= 8;
            self.range <<= 8;
            self.code = (self.code << 8) | self.byte()? as u32;
        }
        Ok(())
    }
    fn take(&mut self, total: u32) -> std::io::Result<u32> {
        let (value, unit) = self.value(total)?;
        self.interval(value, value + 1, unit)?;
        Ok(value)
    }
    fn bits(&mut self, count: u32) -> std::io::Result<u32> {
        self.take(1 << count)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct EntropyState {
    pub bytes_read: u64,
    low: u32,
    range: u32,
    code: u32,
    model: [i32; 2],
}

pub struct EntropyDecoder<R> {
    coder: RangeDecoder<R>,
    state: [i32; 2],
    channels: usize,
}
impl<R: std::io::Read> EntropyDecoder<R> {
    pub fn new(input: R, channels: usize) -> std::io::Result<Self> {
        Ok(Self {
            coder: RangeDecoder::new(input)?,
            state: [0x4000; 2],
            channels,
        })
    }
    pub fn into_inner(self) -> R {
        self.coder.input
    }
    pub(crate) fn snapshot(&self) -> EntropyState {
        EntropyState {
            bytes_read: self.coder.bytes_read,
            low: self.coder.low,
            range: self.coder.range,
            code: self.coder.code,
            model: self.state,
        }
    }
    pub(crate) fn restore(input: R, channels: usize, state: EntropyState) -> Self {
        Self {
            coder: RangeDecoder {
                input,
                low: state.low,
                range: state.range,
                code: state.code,
                bytes_read: state.bytes_read,
            },
            state: state.model,
            channels,
        }
    }
    pub fn decode_frame(&mut self, len: usize) -> std::io::Result<Vec<i16>> {
        let mut silent = [false; 2];
        for flag in &mut silent[..self.channels] {
            *flag = self.coder.bits(1)? != 0;
        }
        let active = silent[..self.channels].iter().filter(|&&s| !s).count();
        let mut samples = vec![0; len];
        if active == 0 {
            return Ok(samples);
        }
        let rate = self.coder.bits(2)? + 1;
        let count = len / self.channels * active;
        let mut counts = [0u32; 192];
        let mut remaining = count as u32;
        for frequency in &mut counts {
            if remaining == 0 {
                break;
            }
            *frequency = if remaining < 32 {
                let n = remaining as usize;
                let start = (n - 1) * (n + 2) / 2;
                let weights = &SMALL_COUNTS[start..start + n + 1];
                let total = weights.iter().sum();
                let (value, unit) = self.coder.value(total)?;
                let mut lower = 0;
                let index = weights
                    .iter()
                    .position(|&w| {
                        if value < lower + w {
                            true
                        } else {
                            lower += w;
                            false
                        }
                    })
                    .ok_or_else(|| crate::decoder::invalid("invalid LA04 histogram"))?;
                self.coder.interval(lower, lower + weights[index], unit)?;
                index as u32
            } else {
                self.coder.take(remaining + 1)?
            };
            remaining -= *frequency;
        }
        if remaining != 0 {
            return Err(crate::decoder::invalid("incomplete LA04 histogram"));
        }
        remaining = count as u32;
        let mut histogram = Histogram::new(&counts);
        for i in 0..count {
            let (target, unit) = self.coder.value(remaining)?;
            let bin = histogram.find(target);
            let (lower, upper) = histogram.interval(bin);
            self.coder.interval(lower, upper, unit)?;
            histogram.consume(bin);
            remaining -= 1;
            let state = &mut self.state[i & (active - 1)];
            let divisor = (*state >> 6).max(1) as u32;
            let value = if bin < 191 {
                bin as u32 * divisor
                    + if divisor > 1 {
                        self.coder.take(divisor)?
                    } else {
                        0
                    }
            } else {
                let bits = self.coder.bits(5)?;
                if bits > 16 {
                    return Err(crate::decoder::invalid("invalid LA04 escape width"));
                }
                self.coder.bits(bits)?
            };
            if value > u16::MAX as u32 {
                return Err(crate::decoder::invalid("LA04 residual exceeds 16 bits"));
            }
            let delta = ((value as i32 + 6) >> 1) - ((*state + 16) >> 5);
            *state += match rate {
                1 => delta >> 1,
                2 => delta,
                3 => delta >> 2,
                4 => delta * 2,
                _ => unreachable!(),
            };
            let index = if active == self.channels {
                i
            } else {
                i * 2 + silent[0] as usize
            };
            samples[index] = (value >> 1) as i16 ^ -((value & 1) as i16);
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_masked_updates_preserve_all_other_counters() {
        let counts = std::array::from_fn(|i| (i + 1) as u32);
        for bin in 0..192 {
            let mut histogram = Histogram::new(&counts);
            let mut expected = histogram.tails;
            for tail in &mut expected[..=bin] {
                *tail -= 1;
            }
            histogram.consume(bin);
            assert_eq!(histogram.tails, expected, "bin {bin}");
        }
    }

    #[test]
    fn cumulative_histogram_matches_linear_model_until_exhaustion() {
        let mut state = 123u32;
        let models = [
            std::array::from_fn(|i| (i % 7) as u32),
            std::array::from_fn(|i| if i == 191 { 61440 } else { 0 }),
            std::array::from_fn(|i| if i == 0 { 61440 } else { 0 }),
            std::array::from_fn(|i| if i % 17 == 0 { 19 } else { 0 }),
        ];
        for mut counts in models {
            let mut histogram = Histogram::new(&counts);
            let mut remaining: u32 = counts.iter().sum();
            while remaining != 0 {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let target = state % remaining;
                let mut lower = 0;
                let bin = counts
                    .iter()
                    .position(|&count| {
                        if target < lower + count {
                            true
                        } else {
                            lower += count;
                            false
                        }
                    })
                    .unwrap();
                let upper = lower + counts[bin];
                assert_eq!(histogram.find(target), bin);
                assert_eq!(histogram.find(lower), bin);
                assert_eq!(histogram.find(upper - 1), bin);
                assert_eq!(histogram.interval(bin), (lower, upper));
                histogram.consume(bin);
                counts[bin] -= 1;
                remaining -= 1;
                assert_eq!(histogram.tails[0], remaining);
                assert_eq!(histogram.tails[192], 0);
            }
            assert_eq!(histogram.tails, [0; 193]);
        }
    }

    #[test]
    fn compact_model_round_trips_extreme_residuals() {
        let frames = [
            vec![i16::MIN; 61440],
            vec![i16::MAX; 61440],
            vec![-1; 61440],
            (0..61440)
                .map(|i| match i % 4 {
                    0 => i16::MIN,
                    1 => 0,
                    2 => i16::MAX,
                    _ => 1,
                })
                .collect(),
        ];
        for channels in [1, 2] {
            let mut encoder = EntropyEncoder::new(channels);
            for frame in &frames {
                encoder.encode_frame(frame);
            }
            encoder.coder.finish();
            let mut decoder = EntropyDecoder::new(&encoder.coder.bytes[..], channels).unwrap();
            for frame in &frames {
                assert_eq!(&decoder.decode_frame(frame.len()).unwrap(), frame);
            }
        }
    }
}
