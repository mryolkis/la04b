use super::*;

#[test]
#[ignore = "requires original DLL dumps in local/reference; see tools/README.md"]
fn lookup_tables_match_original_dll() {
    let table = tables::tables();
    let gradient = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/local/reference/gradient.bin"
    ))
    .unwrap();
    assert_eq!(
        table.gradient.iter().map(|&x| x as u8).collect::<Vec<_>>(),
        gradient
    );
    let gradient = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/local/reference/short_gradient.bin"
    ))
    .unwrap();
    assert_eq!(
        table
            .short_gradient
            .iter()
            .map(|&x| x as u8)
            .collect::<Vec<_>>(),
        gradient
    );
    let rates = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/local/reference/learning_rate.bin"
    ))
    .unwrap();
    for (i, x) in rates.as_chunks::<2>().0.iter().enumerate() {
        assert_eq!(
            table.learning_rate[i],
            u16::from_le_bytes([x[0], x[1]]),
            "rate index {i}"
        );
    }

    let scales = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/local/reference/scales.bin"
    ))
    .unwrap();
    for (i, bytes) in scales.as_chunks::<4>().0.iter().enumerate() {
        assert_eq!(
            tables::sample_scale(i as i32 - 7),
            i32::from_le_bytes(*bytes)
        );
    }
}

#[test]
#[ignore = "requires original DLL dumps in local/reference; see tools/README.md"]
fn mono_stages_match_original_dll() {
    let wav = wave::Wave::read(&include_bytes!("../../tests/fixtures/edges_mono.wav")[..]).unwrap();
    let mut samples = wav.samples;
    let mut pipeline = predictor::Pipeline::new(1, false);
    pipeline.encode(&mut samples, |stage, samples| {
        let path = format!(
            "{}/local/reference/edges_mono.normal.out{stage}.raw",
            env!("CARGO_MANIFEST_DIR")
        );
        let reference = std::fs::read(path).unwrap();
        for (i, x) in reference.as_chunks::<2>().0.iter().enumerate() {
            assert_eq!(
                samples[i],
                i16::from_le_bytes([x[0], x[1]]),
                "stage {stage}, sample {i}"
            );
        }
    });
    let mut entropy = entropy::EntropyEncoder::new(1);
    entropy.encode_frame(&samples);
    entropy.coder.finish();
    let reference = include_bytes!("../../tests/fixtures/edges_mono.normal.la");
    assert_eq!(entropy.coder.bytes, reference[53..]);
}

#[test]
#[ignore = "requires original DLL dumps in local/reference; see tools/README.md"]
fn all_stages_match_original_dll() {
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
            let mut output = std::io::Cursor::new(Vec::new());
            let mut encoder =
                Encoder::new(&wav[..], &mut output, Options { high, seeking }).unwrap();
            let total_frames = encoder.get_total_frames();
            let mut frame = 0;
            while encoder.has_next_frame() {
                encoder
                    .create_next_frame_observed(|stage, samples| {
                        let path =
                            root.join(format!("local/reference/{name}.{mode}.out{stage}.raw"));
                        let data = std::fs::read(path).unwrap();

                        if frame + 1 < total_frames {
                            return;
                        }
                        let data = &data[..samples.len() * 2];
                        for (i, x) in data.as_chunks::<2>().0.iter().enumerate() {
                            assert_eq!(
                                samples[i],
                                i16::from_le_bytes([x[0], x[1]]),
                                "{name} {mode} stage {stage}, frame {frame}, sample {i}"
                            );
                        }
                    })
                    .unwrap();
                frame += 1;
            }
        }
    }
}
