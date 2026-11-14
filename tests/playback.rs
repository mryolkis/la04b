use la04::{Decoder, FRAME_SIZE, Options, wave::Wave};
use std::io::{self, Cursor, Read, Seek, SeekFrom};

fn collect<R: Read + Seek>(decoder: &mut Decoder<R>) -> io::Result<Vec<i16>> {
    let mut pcm = Vec::new();
    loop {
        let frame = decoder.read_frame()?;
        if frame.is_empty() {
            return Ok(pcm);
        }
        assert!(frame.len() <= FRAME_SIZE);
        pcm.extend(frame);
    }
}

#[test]
fn playback_matches_original_pcm_and_seeks() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for name in [
        "silence_mono",
        "edges_mono",
        "noise_stereo",
        "tone_stereo",
        "multiframe_mono",
    ] {
        let wav =
            Wave::read(std::fs::File::open(root.join(format!("{name}.wav"))).unwrap()).unwrap();
        for mode in ["normal", "high", "noseek", "high_noseek"] {
            let mut decoder =
                Decoder::new(std::fs::File::open(root.join(format!("{name}.{mode}.la"))).unwrap())
                    .unwrap();
            let info = decoder.info();
            assert_eq!(info.sample_rate, wav.sample_rate);
            assert_eq!(info.channels, wav.channels);
            assert_eq!(collect(&mut decoder).unwrap(), wav.samples, "{name} {mode}");
            let channels = wav.channels as usize;
            let len = wav.samples.len() / channels;
            for offset in [
                len / 2,
                1.min(len),
                len,
                len.saturating_add(100),
                0,
                len.saturating_sub(1),
                0,
            ] {
                decoder.seek(offset as u64).unwrap();
                assert_eq!(
                    collect(&mut decoder).unwrap(),
                    wav.samples[offset.min(len) * channels..],
                    "{name} {mode}, seek {offset}"
                );
            }
        }
    }
}

#[test]
fn seek_across_restart_boundaries_in_both_channel_layouts() {
    for channels in [1, 2] {
        let mut samples = Vec::new();

        for i in 0..FRAME_SIZE * 39 + 128 {
            samples.push(((i * 997 + i / 13 * 31) % 65536) as i16);
        }

        let mut fixture = Wave::read(&include_bytes!("fixtures/tone_stereo.wav")[..]).unwrap();
        fixture.channels = channels;
        fixture.format[10..12].copy_from_slice(&channels.to_le_bytes());
        fixture.format[20..22].copy_from_slice(&(channels * 2).to_le_bytes());
        fixture.samples = samples;
        for high in [false, true] {
            for seeking in [false, true] {
                let mut encoded = Cursor::new(Vec::new());
                let mut encoder = la04::Encoder::from_wave(
                    Wave {
                        riff: fixture.riff,
                        format: fixture.format.clone(),
                        channels,
                        sample_rate: fixture.sample_rate,
                        samples: fixture.samples.clone(),
                        footer: vec![],
                    },
                    &mut encoded,
                    Options { high, seeking },
                )
                .unwrap();
                while encoder.has_next_frame() {
                    encoder.create_next_frame().unwrap();
                }
                encoder.write_footer().unwrap();
                drop(encoder);
                let mut decoder = Decoder::new(Cursor::new(encoded.into_inner())).unwrap();
                assert_eq!(collect(&mut decoder).unwrap(), fixture.samples);
                let c = channels as usize;
                for target in [
                    FRAME_SIZE * 19 / c - 1,
                    FRAME_SIZE * 19 / c,
                    FRAME_SIZE * 19 / c + 1,
                    FRAME_SIZE * 38 / c + 17,
                    FRAME_SIZE / c + 5,
                    0,
                ] {
                    decoder.seek(target as u64).unwrap();
                    let frame = decoder.read_frame().unwrap();
                    assert_eq!(frame, fixture.samples[target * c..target * c + frame.len()]);
                }
            }
        }
    }
}

#[test]
fn rejects_bad_crc_truncation_and_offsets() {
    let reference = include_bytes!("fixtures/edges_mono.normal.la");
    for len in 0..reference.len() {
        let result = Decoder::new(Cursor::new(&reference[..len])).and_then(|mut d| collect(&mut d));
        assert!(result.is_err(), "accepted truncated file at {len}");
    }
    let mut bad = reference.to_vec();
    bad[45] ^= 1;
    let mut decoder = Decoder::new(Cursor::new(bad)).unwrap();
    assert!(
        collect(&mut decoder)
            .unwrap_err()
            .to_string()
            .contains("CRC")
    );
    assert!(decoder.read_frame().is_err());
    assert!(decoder.seek(0).is_err());
    let mut bad = reference.to_vec();
    bad[49..53].copy_from_slice(&0u32.to_le_bytes());
    assert!(Decoder::new(Cursor::new(bad)).is_err());
}

#[test]
fn empty_file_is_seekable_and_crc_checked() {
    let mut wave = Wave::read(&include_bytes!("fixtures/edges_mono.wav")[..]).unwrap();
    wave.samples.clear();
    let mut data = Cursor::new(Vec::new());
    la04::encode_wave(wave, &mut data, Options::default()).unwrap();
    let mut decoder = Decoder::new(Cursor::new(data.into_inner())).unwrap();
    assert_eq!(decoder.info().sample_frames, 0);
    assert!(decoder.read_frame().unwrap().is_empty());
    decoder.seek(u64::MAX).unwrap();
    assert!(decoder.read_frame().unwrap().is_empty());
}

#[test]
fn opening_does_not_read_compressed_audio() {
    struct HeaderOnly(Cursor<Vec<u8>>);
    impl Read for HeaderOnly {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            if self.0.position() >= 53 {
                panic!("open read compressed audio");
            }
            self.0.read(b)
        }
    }
    impl Seek for HeaderOnly {
        fn seek(&mut self, p: SeekFrom) -> io::Result<u64> {
            self.0.seek(p)
        }
    }
    let input = HeaderOnly(Cursor::new(
        include_bytes!("fixtures/edges_mono.normal.la").to_vec(),
    ));
    assert!(Decoder::new(input).is_ok());
}

#[test]
fn can_cancel_while_discarding_silent_frames_for_a_no_seek_file() {
    let mut wave = Wave::read(&include_bytes!("fixtures/edges_mono.wav")[..]).unwrap();
    wave.samples = vec![0; FRAME_SIZE * 30];
    let mut data = Cursor::new(Vec::new());
    la04::encode_wave(
        wave,
        &mut data,
        Options {
            high: false,
            seeking: false,
        },
    )
    .unwrap();
    let mut decoder = Decoder::new(Cursor::new(data.into_inner())).unwrap();
    decoder.seek((FRAME_SIZE * 29 + 1) as u64).unwrap();
    let mut checks = 0;
    let error = decoder
        .read_frame_with_cancel(|| {
            checks += 1;
            if checks == 3 {
                Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "test cancellation",
                ))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::Interrupted);
    assert_eq!(checks, 3);
}

#[test]
fn no_seek_checkpoints_resume_compressed_file_io_without_pcm_cache() {
    use std::cell::Cell;
    use std::rc::Rc;
    struct CountedInput {
        input: Cursor<Vec<u8>>,
        bytes_read: Rc<Cell<usize>>,
    }
    impl Read for CountedInput {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            let count = self.input.read(bytes)?;
            self.bytes_read.set(self.bytes_read.get() + count);
            Ok(count)
        }
    }
    impl Seek for CountedInput {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.input.seek(position)
        }
    }
    let wave = Wave::read(&include_bytes!("fixtures/multiframe_mono.wav")[..]).unwrap();
    for data in [
        &include_bytes!("fixtures/multiframe_mono.noseek.la")[..],
        &include_bytes!("fixtures/multiframe_mono.high_noseek.la")[..],
    ] {
        let reads = Rc::new(Cell::new(0));
        let mut decoder = Decoder::new(CountedInput {
            input: Cursor::new(data.to_vec()),
            bytes_read: reads.clone(),
        })
        .unwrap();
        assert_eq!(collect(&mut decoder).unwrap(), wave.samples);
        reads.set(0);
        decoder.seek((FRAME_SIZE + 13) as u64).unwrap();
        assert_eq!(
            decoder.read_frame().unwrap(),
            wave.samples[FRAME_SIZE + 13..]
        );
        assert!(
            reads.get() > 0,
            "checkpoint seek must still read compressed input"
        );
        assert!(
            reads.get() < data.len() / 4,
            "checkpoint replayed the whole prefix"
        );
    }
}
