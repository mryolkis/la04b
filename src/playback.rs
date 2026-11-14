use crate::{
    FRAME_SIZE,
    crc::Crc32,
    decoder::{Header, invalid, read_header},
    encoder::SEEK_INTERVAL,
    entropy::{EntropyDecoder, EntropyState},
    predictor::Pipeline,
};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Take};

#[derive(Clone, Copy, Debug)]
pub struct AudioInfo {
    pub channels: u16,
    pub sample_rate: u32,

    pub sample_frames: u64,
    pub high: bool,
    pub seek_table: bool,
}

const MAX_CHECKPOINTS: usize = 64;
struct Checkpoint {
    position: usize,
    entropy: EntropyState,
    pipeline: Pipeline,
}

pub struct Decoder<R> {
    header: Header,
    input: Option<R>,
    entropy: Option<EntropyDecoder<BufReader<Take<R>>>>,
    pipeline: Pipeline,
    position: usize,
    skip: usize,
    crc: Option<Crc32>,
    failed: bool,
    checkpoints: Vec<Checkpoint>,
    checkpoint_stride: usize,
}

impl<R: Read + Seek> Decoder<R> {
    pub fn new(mut input: R) -> io::Result<Self> {
        input.seek(SeekFrom::Start(0))?;
        let header = read_header(&mut input)?;
        if input.seek(SeekFrom::End(0))? < header.footer_start as u64 {
            return Err(invalid("truncated LA04 payload"));
        }
        let pipeline = Pipeline::new(header.channels, header.high);
        let checkpoint_stride = header
            .samples
            .div_ceil(FRAME_SIZE)
            .div_ceil(MAX_CHECKPOINTS)
            .max(1)
            * FRAME_SIZE;
        Ok(Self {
            header,
            input: Some(input),
            entropy: None,
            pipeline,
            position: 0,
            skip: 0,
            crc: Some(Crc32::default()),
            failed: false,
            checkpoints: Vec::new(),
            checkpoint_stride,
        })
    }

    pub fn info(&self) -> AudioInfo {
        AudioInfo {
            channels: self.header.channels as u16,
            sample_rate: self.header.sample_rate,
            sample_frames: (self.header.samples / self.header.channels) as u64,
            high: self.header.high,
            seek_table: self.header.seeking,
        }
    }

    fn take_input(&mut self) -> io::Result<R> {
        match self.entropy.take() {
            Some(entropy) => Ok(entropy.into_inner().into_inner().into_inner()),
            None => self
                .input
                .take()
                .ok_or_else(|| invalid("decoder needs reopening")),
        }
    }

    fn start_segment(&mut self, segment: usize) -> io::Result<()> {
        let mut input = self.take_input()?;
        let start = self.header.offsets[segment];
        let end = self.header.offsets[segment + 1];
        input.seek(SeekFrom::Start(start as u64))?;
        self.entropy = Some(EntropyDecoder::new(
            BufReader::new(input.take((end - start) as u64)),
            self.header.channels,
        )?);
        self.pipeline = Pipeline::new(self.header.channels, self.header.high);
        Ok(())
    }

    pub fn seek(&mut self, sample_frame: u64) -> io::Result<()> {
        if self.failed {
            return Err(invalid("decoder needs reopening after a decode error"));
        }
        let target = sample_frame.min(self.info().sample_frames) as usize * self.header.channels;
        let segment_size = SEEK_INTERVAL * FRAME_SIZE;
        let base = if target == self.header.samples {
            target
        } else if self.header.seeking {
            target / segment_size * segment_size
        } else {
            0
        };
        if !self.header.seeking && target < self.header.samples && target != 0 {
            let checkpoint = self.checkpoints.iter().rposition(|p| p.position <= target);
            let saved_position = checkpoint.map_or(0, |i| self.checkpoints[i].position);

            if self.entropy.is_some() && self.position <= target && self.position >= saved_position
            {
                self.skip = target - self.position;
                self.crc = None;
                return Ok(());
            }
            if let Some(index) = checkpoint {
                let point = &self.checkpoints[index];
                let position = point.position;
                let state = point.entropy;
                let pipeline = point.pipeline.clone();
                let offset = self.header.payload_start as u64 + state.bytes_read;
                let end = self.header.footer_start as u64;
                let mut input = self.take_input()?;
                if let Err(error) = input.seek(SeekFrom::Start(offset)) {
                    self.failed = true;
                    return Err(error);
                }
                self.entropy = Some(EntropyDecoder::restore(
                    BufReader::new(input.take(end - offset)),
                    self.header.channels,
                    state,
                ));
                self.pipeline = pipeline;
                self.position = position;
                self.skip = target - position;
                self.crc = None;
                return Ok(());
            }
        }

        if let Some(entropy) = self.entropy.take() {
            self.input = Some(entropy.into_inner().into_inner().into_inner());
        }
        self.position = base;
        self.skip = target - base;
        self.crc = (target == 0).then(Crc32::default);
        Ok(())
    }

    pub fn read_frame(&mut self) -> io::Result<Vec<i16>> {
        self.read_frame_with_cancel(|| Ok(()))
    }

    pub fn read_frame_with_cancel(
        &mut self,
        mut check: impl FnMut() -> io::Result<()>,
    ) -> io::Result<Vec<i16>> {
        if self.failed {
            return Err(invalid("decoder needs reopening after a decode error"));
        }
        let result = self.read_frame_inner(&mut check);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn read_frame_inner(
        &mut self,
        check: &mut impl FnMut() -> io::Result<()>,
    ) -> io::Result<Vec<i16>> {
        loop {
            check()?;
            if self.position == self.header.samples {
                self.check_crc()?;
                return Ok(Vec::new());
            }
            let segment_size = SEEK_INTERVAL * FRAME_SIZE;
            if self.entropy.is_none()
                || (self.header.seeking && self.position.is_multiple_of(segment_size))
            {
                let segment = if self.header.seeking {
                    self.position / segment_size
                } else {
                    0
                };
                self.start_segment(segment)?;
            }
            let count = FRAME_SIZE.min(self.header.samples - self.position);
            let mut frame = self.entropy.as_mut().unwrap().decode_frame(count)?;
            self.pipeline.decode(&mut frame);
            if let Some(crc) = &mut self.crc {
                crc.update_samples(&frame);
            }
            self.position += count;
            if !self.header.seeking
                && self.position < self.header.samples
                && self.position.is_multiple_of(self.checkpoint_stride)
                && let Err(index) = self
                    .checkpoints
                    .binary_search_by_key(&self.position, |p| p.position)
            {
                self.checkpoints.insert(
                    index,
                    Checkpoint {
                        position: self.position,
                        entropy: self.entropy.as_ref().unwrap().snapshot(),
                        pipeline: self.pipeline.clone(),
                    },
                );
            }
            if self.position == self.header.samples {
                self.check_crc()?;
            }
            let skip = self.skip.min(frame.len());
            self.skip -= skip;
            if skip == frame.len() {
                continue;
            }
            frame.drain(..skip);
            return Ok(frame);
        }
    }

    fn check_crc(&self) -> io::Result<()> {
        if self
            .crc
            .as_ref()
            .is_some_and(|crc| crc.value() != self.header.expected_crc)
        {
            return Err(invalid("LA04 PCM CRC mismatch"));
        }
        Ok(())
    }
}
