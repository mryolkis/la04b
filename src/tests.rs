use super::*;

mod reference;

#[test]
fn all_reference_files_match() {
    for name in [
        "silence_mono",
        "edges_mono",
        "noise_stereo",
        "tone_stereo",
        "multiframe_mono",
    ] {
        for (mode, high, seeking) in [
            ("normal", false, true),
            ("high", true, true),
            ("noseek", false, false),
            ("high_noseek", true, false),
        ] {
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            let wav = std::fs::read(root.join(format!("tests/fixtures/{name}.wav"))).unwrap();
            let reference =
                std::fs::read(root.join(format!("tests/fixtures/{name}.{mode}.la"))).unwrap();
            let mut output = std::io::Cursor::new(Vec::new());
            encode(&wav[..], &mut output, Options { high, seeking }).unwrap();
            let actual = output.into_inner();
            let mismatch = actual.iter().zip(&reference).position(|(a, b)| a != b);
            assert!(
                mismatch.is_none() && actual.len() == reference.len(),
                "{name} {mode}: mismatch {mismatch:?}, sizes {} / {}",
                actual.len(),
                reference.len()
            );
            let mut restored = Vec::new();
            decode(&reference[..], &mut restored).unwrap();
            assert_eq!(restored, wav, "decoded {name} {mode}");
        }
    }
}

#[test]
fn encoder_lifecycle_crc_and_footer() {
    let mut crc = crc::Crc32::default();
    crc.update(b"123456789");
    assert_eq!(crc.value(), 0x340bc6d9);
    let mut wav = include_bytes!("../tests/fixtures/edges_mono.wav").to_vec();
    wav.extend_from_slice(b"arbitrary trailing data\x00\xff");
    let mut output = std::io::Cursor::new(Vec::new());
    let mut encoder = Encoder::new(&wav[..], &mut output, Options::default()).unwrap();
    assert_eq!(encoder.get_total_frames(), 1);
    assert_eq!(encoder.get_frame_size(), 61440);
    assert!(
        encoder
            .get_approx_compression_ratio_to_date()
            .unwrap()
            .is_infinite()
    );
    assert!(encoder.write_footer().is_err());
    encoder.create_next_frame().unwrap();
    assert!(!encoder.has_next_frame());
    assert!(encoder.create_next_frame().is_err());
    let before_footer = encoder.get_approx_compression_ratio_to_date().unwrap();
    let size = encoder.write_footer().unwrap();
    assert_eq!(encoder.write_footer().unwrap(), size);
    assert_eq!(
        encoder.get_approx_compression_ratio_to_date().unwrap(),
        before_footer
    );
    drop(encoder);
    assert_eq!(size as usize, output.get_ref().len());
    assert!(
        output
            .get_ref()
            .ends_with(b"arbitrary trailing data\x00\xff")
    );
}

#[test]
fn rejects_truncated_and_unsupported_wave() {
    let wav = include_bytes!("../tests/fixtures/edges_mono.wav");
    for length in [0, 11, 24, 43, wav.len() - 1] {
        assert!(wave::Wave::read(&wav[..length]).is_err());
    }
    let mut invalid = wav.to_vec();
    invalid[34..36].copy_from_slice(&24u16.to_le_bytes());
    assert!(wave::Wave::read(&invalid[..]).is_err());
}

#[test]
fn decoder_rejects_truncation_and_corruption() {
    let reference = include_bytes!("../tests/fixtures/edges_mono.normal.la");
    for len in 0..reference.len() {
        assert!(
            decode(&reference[..len], std::io::sink()).is_err(),
            "length {len}"
        );
    }
    for offset in [0, 12, 16, 40, 44, 45, 49, 53, reference.len() - 8] {
        let mut corrupt = reference.to_vec();
        corrupt[offset] ^= 0x80;
        assert!(
            decode(&corrupt[..], std::io::sink()).is_err(),
            "offset {offset}"
        );
    }
}

#[test]
fn round_trip_empty_and_single_silent_channel() {
    let template = include_bytes!("../tests/fixtures/tone_stereo.wav");
    for count in [0, 2, 18, 61440, 61448] {
        for silent_channel in [0, 1] {
            let mut wav = template[..44].to_vec();
            wav[4..8].copy_from_slice(&(36u32 + count * 2).to_le_bytes());
            wav[40..44].copy_from_slice(&(count * 2).to_le_bytes());
            for i in 0..count {
                let sample = if i % 2 == silent_channel {
                    0
                } else {
                    i.wrapping_mul(193) as i16
                };
                wav.extend_from_slice(&sample.to_le_bytes());
            }
            wav.extend_from_slice(b"preserved footer\0\xff");
            for high in [false, true] {
                for seeking in [false, true] {
                    let mut encoded = std::io::Cursor::new(Vec::new());
                    encode(&wav[..], &mut encoded, Options { high, seeking }).unwrap();
                    let mut restored = Vec::new();
                    decode(&encoded.into_inner()[..], &mut restored).unwrap();
                    assert_eq!(wav, restored);
                }
            }
        }
    }
}

#[test]
fn parallel_modes_match_sequential_frame_api() {
    for channels in [1u16, 2] {
        let count = 19 * FRAME_SIZE + 18;
        let mut wav = include_bytes!("../tests/fixtures/tone_stereo.wav")[..44].to_vec();
        wav[4..8].copy_from_slice(&(36 + count as u32 * 2).to_le_bytes());
        wav[22..24].copy_from_slice(&channels.to_le_bytes());
        wav[28..32].copy_from_slice(&(44100u32 * channels as u32 * 2).to_le_bytes());
        wav[32..34].copy_from_slice(&(channels * 2).to_le_bytes());
        wav[40..44].copy_from_slice(&(count as u32 * 2).to_le_bytes());
        let mut state = 123u32;
        for i in 0..count {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let sample = if i / 7000 % 3 == 0 {
                0
            } else {
                (state >> 16) as i16
            };
            wav.extend_from_slice(&sample.to_le_bytes());
        }
        for high in [false, true] {
            for seeking in [false, true] {
                let options = Options { high, seeking };
                let mut sequential = std::io::Cursor::new(Vec::new());
                let mut encoder = Encoder::new(&wav[..], &mut sequential, options).unwrap();
                while encoder.has_next_frame() {
                    encoder.create_next_frame().unwrap();
                }
                encoder.write_footer().unwrap();
                let mut parallel = std::io::Cursor::new(Vec::new());
                encode(&wav[..], &mut parallel, options).unwrap();
                assert_eq!(parallel.get_ref(), sequential.get_ref());
                let mut restored = Vec::new();
                decode(&parallel.into_inner()[..], &mut restored).unwrap();
                assert_eq!(wav, restored);
            }
        }
    }
}

#[test]
fn pipelined_encoder_stops_workers_on_output_error() {
    struct FailingOutput(std::io::Cursor<Vec<u8>>);
    impl std::io::Write for FailingOutput {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.0.get_ref().len() > 256 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "test output error",
                ));
            }
            std::io::Write::write(&mut self.0, bytes)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl std::io::Seek for FailingOutput {
        fn seek(&mut self, position: std::io::SeekFrom) -> std::io::Result<u64> {
            std::io::Seek::seek(&mut self.0, position)
        }
    }
    let mut wave =
        wave::Wave::read(&include_bytes!("../tests/fixtures/tone_stereo.wav")[..]).unwrap();
    let mut state = 123u32;
    wave.samples = (0..20 * FRAME_SIZE)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as i16
        })
        .collect();
    let result = encode_wave(
        wave,
        FailingOutput(std::io::Cursor::new(Vec::new())),
        Options {
            high: true,
            seeking: false,
        },
    );
    assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::BrokenPipe);
}
