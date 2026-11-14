const TABLES: [[u32; 256]; 8] = {
    let mut tables = [[0; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = (crc >> 1) ^ (0xedb88320 & 0u32.wrapping_sub(crc & 1));
            bit += 1;
        }
        tables[0][i] = crc;
        i += 1;
    }
    let mut n = 1;
    while n < 8 {
        let mut i = 0;
        while i < 256 {
            let crc = tables[n - 1][i];
            tables[n][i] = (crc >> 8) ^ tables[0][(crc & 255) as usize];
            i += 1;
        }
        n += 1;
    }
    tables
};

#[derive(Debug, Clone)]
pub struct Crc32(u32);

impl Default for Crc32 {
    fn default() -> Self {
        Self(u32::MAX)
    }
}
impl Crc32 {
    #[inline]
    fn block(&mut self, first: u32, second: u32) {
        let first = first ^ self.0;
        self.0 = TABLES[7][first as u8 as usize]
            ^ TABLES[6][(first >> 8) as u8 as usize]
            ^ TABLES[5][(first >> 16) as u8 as usize]
            ^ TABLES[4][(first >> 24) as usize]
            ^ TABLES[3][second as u8 as usize]
            ^ TABLES[2][(second >> 8) as u8 as usize]
            ^ TABLES[1][(second >> 16) as u8 as usize]
            ^ TABLES[0][(second >> 24) as usize];
    }
    pub fn update(&mut self, bytes: &[u8]) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if bytes.len() >= 128
            && std::arch::is_x86_feature_detected!("pclmulqdq")
            && std::arch::is_x86_feature_detected!("ssse3")
        {
            // SAFETY: CPU/OS features were checked above. Only full blocks are read.
            let consumed = unsafe { self.update_pclmul(bytes) };
            self.update_scalar(&bytes[consumed..]);
            return;
        }
        self.update_scalar(bytes);
    }
    fn update_scalar(&mut self, bytes: &[u8]) {
        let (chunks, remainder) = bytes.as_chunks::<8>();
        for bytes in chunks {
            self.block(
                u32::from_le_bytes(bytes[..4].try_into().unwrap()),
                u32::from_le_bytes(bytes[4..].try_into().unwrap()),
            );
        }
        for &byte in remainder {
            self.0 = (self.0 >> 8) ^ TABLES[0][((self.0 as u8) ^ byte) as usize];
        }
    }
    pub(crate) fn update_samples(&mut self, samples: &[i16]) {
        #[cfg(target_endian = "little")]
        {
            // SAFETY: every i16 is initialized and occupies two bytes. The borrowed
            // byte slice has the same lifetime and little-endian PCM representation.
            self.update(unsafe {
                std::slice::from_raw_parts(samples.as_ptr().cast(), std::mem::size_of_val(samples))
            });
        }
        #[cfg(target_endian = "big")]
        self.update_samples_scalar(samples);
    }
    #[cfg(any(test, target_endian = "big"))]
    fn update_samples_scalar(&mut self, samples: &[i16]) {
        let (chunks, remainder) = samples.as_chunks::<4>();
        for samples in chunks {
            self.block(
                samples[0] as u16 as u32 | ((samples[1] as u16 as u32) << 16),
                samples[2] as u16 as u32 | ((samples[3] as u16 as u32) << 16),
            );
        }
        for sample in remainder {
            self.update(&sample.to_le_bytes());
        }
    }
    pub fn value(&self) -> u32 {
        self.0
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "pclmulqdq,ssse3")]
    unsafe fn update_pclmul(&mut self, bytes: &[u8]) -> usize {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        // x^n modulo the normal ZIP polynomial. Derive folding constants rather
        // than using reflected-polynomial magic numbers.
        const fn power(n: usize) -> i64 {
            let mut r = 1u32;
            let mut i = 0;
            while i < n {
                r = (r << 1) ^ (0x04c11db7 & 0u32.wrapping_sub(r >> 31));
                i += 1;
            }
            r as i64
        }

        // SAFETY: the caller checks the instruction sets and minimum length.
        // All loads are full 16-byte blocks within bytes, with no alignment requirement.
        unsafe {
            let nibbles = _mm_setr_epi8(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15);
            let mask = _mm_set1_epi8(15);
            let reverse = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            let reverse_bits = |v| {
                let lo = _mm_shuffle_epi8(nibbles, _mm_and_si128(v, mask));
                let hi = _mm_shuffle_epi8(nibbles, _mm_and_si128(_mm_srli_epi16(v, 4), mask));
                _mm_shuffle_epi8(_mm_or_si128(_mm_slli_epi16(lo, 4), hi), reverse)
            };
            let load = |offset| reverse_bits(_mm_loadu_si128(bytes.as_ptr().add(offset).cast()));
            let fold = |v, k| {
                _mm_xor_si128(
                    _mm_clmulepi64_si128(v, k, 0x00),
                    _mm_clmulepi64_si128(v, k, 0x11),
                )
            };
            let k512 = _mm_set_epi64x(const { power(576) }, const { power(512) });
            let k128 = _mm_set_epi64x(const { power(192) }, const { power(128) });
            let initial = _mm_set_epi32(self.0.reverse_bits() as i32, 0, 0, 0);
            let mut a = _mm_xor_si128(load(0), initial);
            let mut b = load(16);
            let mut c = load(32);
            let mut d = load(48);
            let mut i = 64;
            // Four independent chains hide carry-less multiplication latency.
            while i + 64 <= bytes.len() {
                a = _mm_xor_si128(fold(a, k512), load(i));
                b = _mm_xor_si128(fold(b, k512), load(i + 16));
                c = _mm_xor_si128(fold(c, k512), load(i + 32));
                d = _mm_xor_si128(fold(d, k512), load(i + 48));
                i += 64;
            }
            a = _mm_xor_si128(fold(a, k128), b);
            a = _mm_xor_si128(fold(a, k128), c);
            a = _mm_xor_si128(fold(a, k128), d);
            while i + 16 <= bytes.len() {
                a = _mm_xor_si128(fold(a, k128), load(i));
                i += 16;
            }
            // The existing slicing-by-eight kernel reduces the final 128 bits
            // and multiplies by x^32. The initial CRC is already folded into a.
            let mut final_bytes = [0u8; 16];
            _mm_storeu_si128(final_bytes.as_mut_ptr().cast(), reverse_bits(a));
            self.0 = 0;
            self.update_scalar(&final_bytes);
            i
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_crc_matches_bitwise_for_offsets_tails_and_updates() {
        let mut state = 123u32;
        let bytes: Vec<_> = (0..1024)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 24) as u8
            })
            .collect();
        for offset in 0..8 {
            for len in 0..256 {
                let data = &bytes[offset..offset + len];
                let mut expected = u32::MAX;
                for &byte in data {
                    expected ^= byte as u32;
                    for _ in 0..8 {
                        expected = (expected >> 1) ^ (0xedb88320 & 0u32.wrapping_sub(expected & 1));
                    }
                }
                let mut crc = Crc32::default();
                crc.update(data);
                assert_eq!(crc.value(), expected);
                let mut crc = Crc32::default();
                for chunk in data.chunks(19) {
                    crc.update(chunk);
                }
                assert_eq!(crc.value(), expected);
                if len % 2 == 0 {
                    let samples: Vec<_> = data
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|s| i16::from_le_bytes([s[0], s[1]]))
                        .collect();
                    let mut crc = Crc32::default();
                    for chunk in samples.chunks(13) {
                        crc.update_samples(chunk);
                    }
                    assert_eq!(crc.value(), expected);
                    let mut crc = Crc32::default();
                    crc.update_samples_scalar(&samples);
                    assert_eq!(crc.value(), expected);
                }
            }
        }
    }

    #[test]
    fn folded_crc_matches_scalar_for_long_unaligned_and_incremental_buffers() {
        let mut state = 123u32;
        let bytes: Vec<_> = (0..131104)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 24) as u8
            })
            .collect();
        for offset in 0..32 {
            for len in (0..320).chain([511, 512, 513, 1023, 1024, 4095, 65536, 131072]) {
                let data = &bytes[offset..offset + len];
                let mut expected = Crc32(0x12345678);
                expected.update_scalar(data);
                let mut actual = Crc32(0x12345678);
                actual.update(data);
                assert_eq!(
                    actual.value(),
                    expected.value(),
                    "offset {offset}, len {len}"
                );
            }
        }
        for fill in [None, Some(0), Some(255)] {
            let mut data = bytes.clone();
            if let Some(byte) = fill {
                data.fill(byte);
            }
            let mut expected = Crc32::default();
            expected.update_scalar(&data);
            for chunk_len in [1, 7, 16, 63, 64, 127, 128, 129, 255, 1024, 4096] {
                let mut actual = Crc32::default();
                for chunk in data.chunks(chunk_len) {
                    actual.update(chunk);
                }
                assert_eq!(actual.value(), expected.value(), "chunk length {chunk_len}");
            }
            let samples: Vec<_> = data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&s| i16::from_le_bytes(s))
                .collect();
            let mut actual = Crc32::default();
            actual.update_samples(&samples);
            assert_eq!(actual.value(), expected.value());
        }
    }
}
