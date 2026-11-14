use crate::{crc::Crc32, entropy::EntropyEncoder, predictor::Pipeline, wave::Wave};
use std::io::{self, Read, Seek, SeekFrom, Write};

pub const FRAME_SIZE: usize = 61440;
pub(crate) const SEEK_INTERVAL: usize = 19;

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub high: bool,
    pub seeking: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            high: false,
            seeking: true,
        }
    }
}

pub struct Encoder<W: Write + Seek> {
    wave: Wave,
    output: W,
    options: Options,
    pipeline: Option<Pipeline>,
    entropy: Option<EntropyEncoder>,
    current_frame: usize,
    total_frames: usize,
    crc: Crc32,
    crc_offset: u64,
    footer_offset: u64,
    finished_size: Option<u64>,
}
impl<W: Write + Seek> Encoder<W> {
    pub fn new(input: impl Read, output: W, options: Options) -> io::Result<Self> {
        Self::from_wave(Wave::read(input)?, output, options)
    }

    pub fn from_wave(wave: Wave, mut output: W, options: Options) -> io::Result<Self> {
        if !(1..=2).contains(&wave.channels)
            || !wave.samples.len().is_multiple_of(wave.channels as usize)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid PCM channels or sample count",
            ));
        }
        if wave.samples.len() > i32::MAX as usize || wave.format.len() > i32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "input exceeds La 0.4b's 32-bit limits",
            ));
        }
        let total_frames = wave.samples.len().div_ceil(FRAME_SIZE);
        let mut riff = wave.riff;
        riff[..4].copy_from_slice(b"LA04");
        output.write_all(&riff)?;
        output.write_all(&(wave.format.len() as u32).to_le_bytes())?;
        output.write_all(&wave.format)?;
        output.write_all(&(wave.samples.len() as u32).to_le_bytes())?;
        output.write_all(&[(options.high as u8) * 2 + options.seeking as u8])?;
        let crc_offset = output.stream_position()?;
        output.write_all(&[0; 4])?;
        if options.seeking {
            for _ in 0..total_frames.saturating_sub(1) / SEEK_INTERVAL {
                output.write_all(&[0; 4])?;
            }
        }
        let footer_offset = output.stream_position()?;
        output.write_all(&[0; 4])?;
        Ok(Self {
            wave,
            output,
            options,
            pipeline: None,
            entropy: None,
            current_frame: 0,
            total_frames,
            crc: Crc32::default(),
            crc_offset,
            footer_offset,
            finished_size: None,
        })
    }

    pub fn has_next_frame(&self) -> bool {
        self.current_frame < self.total_frames
    }

    pub fn get_total_frames(&self) -> usize {
        self.total_frames
    }

    pub fn get_frame_size(&self) -> usize {
        FRAME_SIZE
    }

    pub fn get_approx_compression_ratio_to_date(&mut self) -> io::Result<f64> {
        Ok(self.output.stream_position()? as f64 * 50.0 / (self.current_frame * FRAME_SIZE) as f64)
    }
    fn patch_u32(&mut self, position: u64, value: u32) -> io::Result<()> {
        let current = self.output.stream_position()?;
        self.output.seek(SeekFrom::Start(position))?;
        self.output.write_all(&value.to_le_bytes())?;
        self.output.seek(SeekFrom::Start(current))?;
        Ok(())
    }
    fn position32(&mut self) -> io::Result<u32> {
        self.output
            .stream_position()?
            .try_into()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "La output exceeds 4 GiB"))
    }

    pub fn create_next_frame(&mut self) -> io::Result<()> {
        self.create_next_frame_observed(|_, _| {})
    }

    pub fn create_next_frame_observed(
        &mut self,
        observe: impl FnMut(usize, &[i16]),
    ) -> io::Result<()> {
        if self.finished_size.is_some() || !self.has_next_frame() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "no frames remain",
            ));
        }
        let restart = self.current_frame == 0
            || (self.options.seeking && self.current_frame.is_multiple_of(SEEK_INTERVAL));
        if restart {
            if self.current_frame != 0 {
                let position = self.position32()?;
                self.patch_u32(
                    self.crc_offset + (self.current_frame / SEEK_INTERVAL) as u64 * 4,
                    position,
                )?;
            }
            self.pipeline = Some(Pipeline::new(
                self.wave.channels as usize,
                self.options.high,
            ));
        }
        if self.entropy.is_none() {
            self.entropy = Some(EntropyEncoder::new(self.wave.channels as usize));
        }
        let start = self.current_frame * FRAME_SIZE;
        let end = (start + FRAME_SIZE).min(self.wave.samples.len());
        let mut samples = self.wave.samples[start..end].to_vec();
        self.crc.update_samples(&samples);
        self.pipeline
            .as_mut()
            .unwrap()
            .encode(&mut samples, observe);
        let final_frame = self.current_frame + 1 == self.total_frames
            || (self.options.seeking && (self.current_frame + 1).is_multiple_of(SEEK_INTERVAL));
        let entropy = self.entropy.as_mut().unwrap();
        entropy.encode_frame(&samples);
        if final_frame {
            entropy.coder.finish();
        }
        self.output.write_all(&entropy.coder.bytes)?;
        entropy.coder.bytes.clear();
        if final_frame {
            self.entropy = None;
        }
        self.current_frame += 1;
        Ok(())
    }

    pub fn write_footer(&mut self) -> io::Result<u64> {
        if let Some(size) = self.finished_size {
            return Ok(size);
        }
        if self.has_next_frame() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "encode all frames before writing the footer",
            ));
        }
        let position = self.position32()?;
        self.patch_u32(self.footer_offset, position)?;
        self.output.write_all(&self.wave.footer)?;
        let size = self.position32()?;
        self.patch_u32(self.crc_offset, self.crc.value())?;

        self.output.seek(SeekFrom::Start(position as u64))?;
        self.output.flush()?;
        self.finished_size = Some(size as u64);
        Ok(size as u64)
    }
    pub fn into_inner(self) -> W {
        self.output
    }

    fn encode_pipelined(&mut self) -> io::Result<()> {
        let channels = self.wave.channels as usize;
        let stages = Pipeline::new(channels, self.options.high).into_stages();
        let samples = &self.wave.samples;
        let output = &mut self.output;
        std::thread::scope(|scope| -> io::Result<()> {
            let (input, mut frames) = std::sync::mpsc::sync_channel::<Vec<i16>>(1);
            scope.spawn(move || {
                let mut correlation = 0;
                for samples in samples.chunks(FRAME_SIZE) {
                    let mut frame = samples.to_vec();
                    if channels == 2 {
                        crate::predictor::decorrelate(&mut frame, &mut correlation);
                    }
                    if input.send(frame).is_err() {
                        break;
                    }
                }
            });
            for mut stage in stages {
                let (next, received) = std::sync::mpsc::sync_channel(1);
                scope.spawn(move || {
                    for mut frame in frames {
                        stage.encode(&mut frame);
                        if next.send(frame).is_err() {
                            break;
                        }
                    }
                });
                frames = received;
            }

            self.crc.update_samples(samples);
            let mut entropy = EntropyEncoder::new(channels);
            for frame in frames {
                entropy.encode_frame(&frame);
                output.write_all(&entropy.coder.bytes)?;
                entropy.coder.bytes.clear();
            }
            entropy.coder.finish();
            output.write_all(&entropy.coder.bytes)
        })?;
        self.current_frame = self.total_frames;
        Ok(())
    }

    fn encode_all(&mut self) -> io::Result<()> {
        let workers = std::thread::available_parallelism().map_or(1, usize::from);
        let segment_size = FRAME_SIZE * SEEK_INTERVAL;
        if !self.options.seeking && workers > 1 && self.total_frames >= SEEK_INTERVAL {
            return self.encode_pipelined();
        }
        if !self.options.seeking || workers == 1 || self.wave.samples.len() <= segment_size {
            while self.has_next_frame() {
                self.create_next_frame()?;
            }
            return Ok(());
        }
        let channels = self.wave.channels as usize;
        let high = self.options.high;

        let jobs_per_batch = workers
            .saturating_mul(2)
            .min(self.total_frames.div_ceil(SEEK_INTERVAL));
        let batch_size = segment_size * jobs_per_batch;
        for start in (0..self.wave.samples.len()).step_by(batch_size) {
            let end = (start + batch_size).min(self.wave.samples.len());
            let samples = &self.wave.samples[start..end];
            let jobs = samples.len().div_ceil(segment_size);
            let next = std::sync::atomic::AtomicUsize::new(0);
            let mut segments = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..workers.min(jobs))
                    .map(|_| {
                        let next = &next;
                        scope.spawn(move || {
                            let mut completed = Vec::new();
                            loop {
                                let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if index >= jobs {
                                    break;
                                }
                                let first = index * segment_size;
                                let samples =
                                    &samples[first..(first + segment_size).min(samples.len())];
                                let mut pipeline = Pipeline::new(channels, high);
                                let mut entropy = EntropyEncoder::new(channels);
                                for frame in samples.chunks(FRAME_SIZE) {
                                    let mut frame = frame.to_vec();
                                    pipeline.encode(&mut frame, |_, _| {});
                                    entropy.encode_frame(&frame);
                                }
                                entropy.coder.finish();
                                completed.push((
                                    index,
                                    samples.len().div_ceil(FRAME_SIZE),
                                    entropy.coder.bytes,
                                ));
                            }
                            completed
                        })
                    })
                    .collect();
                if start == 0 {
                    self.crc.update_samples(&self.wave.samples);
                }
                handles
                    .into_iter()
                    .flat_map(|handle| handle.join().expect("encoder worker panicked"))
                    .collect::<Vec<_>>()
            });
            segments.sort_unstable_by_key(|segment| segment.0);
            for (_, frames, bytes) in segments {
                if self.current_frame != 0 {
                    let position = self.position32()?;
                    self.patch_u32(
                        self.crc_offset + (self.current_frame / SEEK_INTERVAL) as u64 * 4,
                        position,
                    )?;
                }
                self.output.write_all(&bytes)?;
                self.current_frame += frames;
            }
        }
        Ok(())
    }
}

pub fn encode(input: impl Read, output: impl Write + Seek, options: Options) -> io::Result<u64> {
    encode_wave(Wave::read(input)?, output, options)
}

pub fn encode_wave(wave: Wave, output: impl Write + Seek, options: Options) -> io::Result<u64> {
    let mut encoder = Encoder::from_wave(wave, output, options)?;
    encoder.encode_all()?;
    encoder.write_footer()
}
