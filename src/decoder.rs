use crate::{
    FRAME_SIZE, crc::Crc32, encoder::SEEK_INTERVAL, entropy::EntropyDecoder, predictor::Pipeline,
};
use std::io::{self, Read, Write};

pub(crate) fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn read_u32(input: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0; 4];
    input.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

pub fn decode(mut input: impl Read, mut output: impl Write) -> io::Result<u64> {
    let Header {
        wav: header,
        channels,
        samples,
        seeking,
        high,
        expected_crc,
        offsets,
        payload_start,
        footer_start,
        ..
    } = read_header(&mut input)?;
    let data_len = samples as u64 * 2;
    let mut payload = Vec::new();
    input.read_to_end(&mut payload)?;
    if footer_start - payload_start > payload.len() {
        return Err(invalid("invalid LA04 footer offset"));
    }
    output.write_all(&header)?;
    let mut crc = Crc32::default();
    let segment_size = if seeking {
        SEEK_INTERVAL * FRAME_SIZE
    } else {
        samples
    };
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(offsets.len() - 1);
    for first in (0..offsets.len() - 1).step_by(workers) {
        let end = (first + workers).min(offsets.len() - 1);
        let decoded = std::thread::scope(|scope| {
            let handles: Vec<_> = (first..end)
                .map(|i| {
                    let segment =
                        &payload[offsets[i] - payload_start..offsets[i + 1] - payload_start];
                    let count = segment_size.min(samples.saturating_sub(i * segment_size));
                    scope.spawn(move || decode_segment(segment, count, channels, high))
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("decoder worker panicked"))
                .collect::<io::Result<Vec<_>>>()
        })?;
        for pcm in decoded {
            crc.update(&pcm);
            output.write_all(&pcm)?;
        }
    }
    if crc.value() != expected_crc {
        return Err(invalid("LA04 PCM CRC mismatch"));
    }
    let footer = &payload[footer_start - payload_start..];
    output.write_all(footer)?;
    output.flush()?;
    Ok(header.len() as u64 + data_len + footer.len() as u64)
}

fn decode_segment(input: &[u8], count: usize, channels: usize, high: bool) -> io::Result<Vec<u8>> {
    let mut pcm = Vec::new();
    if count == 0 {
        return Ok(pcm);
    }
    let mut entropy = EntropyDecoder::new(input, channels)?;
    let mut pipeline = Pipeline::new(channels, high);
    for start in (0..count).step_by(FRAME_SIZE) {
        let len = (count - start).min(FRAME_SIZE);
        let mut frame = entropy.decode_frame(len)?;
        pipeline.decode(&mut frame);
        #[cfg(target_endian = "little")]
        // SAFETY: initialized i16 samples occupy two bytes each and already
        // have the WAV byte order. The slice is borrowed only for this copy.
        pcm.extend_from_slice(unsafe {
            std::slice::from_raw_parts(frame.as_ptr().cast(), std::mem::size_of_val(&*frame))
        });
        #[cfg(target_endian = "big")]
        for sample in frame {
            pcm.extend_from_slice(&sample.to_le_bytes());
        }
    }
    Ok(pcm)
}

pub(crate) struct Header {
    pub wav: Vec<u8>,
    pub channels: usize,
    pub sample_rate: u32,
    pub samples: usize,
    pub seeking: bool,
    pub high: bool,
    pub expected_crc: u32,
    pub offsets: Vec<usize>,
    pub payload_start: usize,
    pub footer_start: usize,
}

pub(crate) fn read_header(mut input: impl Read) -> io::Result<Header> {
    let mut riff = [0; 12];
    input.read_exact(&mut riff)?;
    if &riff[..4] != b"LA04" || &riff[8..] != b"WAVE" {
        return Err(invalid("expected an LA04 file"));
    }
    let format_len = read_u32(&mut input)? as usize;
    let mut format = Vec::new();
    input
        .by_ref()
        .take(format_len as u64)
        .read_to_end(&mut format)?;
    if format.len() != format_len || format_len < 24 || &format[..4] != b"fmt " {
        return Err(invalid("invalid LA04 format header"));
    }
    let samples = read_u32(&mut input)? as usize;
    let mut flags = [0];
    input.read_exact(&mut flags)?;
    if flags[0] & !3 != 0 || samples > i32::MAX as usize {
        return Err(invalid("unsupported LA04 flags or sample count"));
    }
    let seeking = flags[0] & 1 != 0;
    let high = flags[0] & 2 != 0;
    let expected_crc = read_u32(&mut input)?;
    let frames = samples.div_ceil(FRAME_SIZE);
    let seek_count = if seeking {
        frames.saturating_sub(1) / SEEK_INTERVAL
    } else {
        0
    };
    let payload_start = 16usize
        .checked_add(format_len)
        .and_then(|n| n.checked_add(13 + seek_count * 4))
        .ok_or_else(|| invalid("LA04 header exceeds address space"))?;
    let mut offsets = vec![payload_start];
    for _ in 0..seek_count {
        offsets.push(read_u32(&mut input)? as usize);
    }
    let footer_start = read_u32(&mut input)? as usize;
    offsets.push(footer_start);
    if footer_start < payload_start || offsets.windows(2).any(|o| o[0] > o[1]) {
        return Err(invalid("invalid LA04 seek or footer offsets"));
    }
    riff[..4].copy_from_slice(b"RIFF");

    let mut header = riff.to_vec();
    header.extend_from_slice(&format);
    header.extend_from_slice(b"data\0\0\0\0");
    let wave = crate::wave::Wave::read(&header[..])?;
    let channels = wave.channels as usize;
    if !samples.is_multiple_of(channels) {
        return Err(invalid("incomplete LA04 stereo sample frame"));
    }
    let data_len = (samples as u32)
        .checked_mul(2)
        .ok_or_else(|| invalid("LA04 PCM exceeds WAV size limit"))?;
    let end = header.len();
    header[end - 4..].copy_from_slice(&data_len.to_le_bytes());
    if wave.sample_rate == 0 {
        return Err(invalid("invalid LA04 sample rate"));
    }
    Ok(Header {
        wav: header,
        channels,
        sample_rate: wave.sample_rate,
        samples,
        seeking,
        high,
        expected_crc,
        offsets,
        payload_start,
        footer_start,
    })
}
