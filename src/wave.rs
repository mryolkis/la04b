use std::io::{self, Read};

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn u16le(bytes: &[u8]) -> u16 {
    u16::from_le_bytes(bytes.try_into().unwrap())
}
fn u32le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().unwrap())
}

pub struct Wave {
    pub riff: [u8; 12],
    pub format: Vec<u8>,
    pub channels: u16,
    pub sample_rate: u32,
    pub samples: Vec<i16>,
    pub footer: Vec<u8>,
}

impl Wave {
    pub fn read(mut input: impl Read) -> io::Result<Self> {
        let mut riff = [0; 12];
        input.read_exact(&mut riff)?;
        if &riff[..4] != b"RIFF" || &riff[8..] != b"WAVE" {
            return Err(invalid("expected a RIFF/WAVE file"));
        }
        let mut format = Vec::new();
        let (mut channels, mut sample_rate) = (0, 0);
        let data_len;
        loop {
            let mut head = [0; 8];
            input.read_exact(&mut head)?;
            let len = u32le(&head[4..]) as usize;
            if &head[..4] == b"data" {
                if channels == 0 {
                    return Err(invalid("missing fmt chunk"));
                }
                data_len = len;
                break;
            }
            let start = format.len();
            format.extend_from_slice(&head);

            let count = input
                .by_ref()
                .take((len + (len & 1)) as u64)
                .read_to_end(&mut format)?;
            if count != len + (len & 1) {
                return Err(invalid("truncated WAVE chunk"));
            }
            if &head[..4] == b"fmt " {
                if start != 0 || len < 16 {
                    return Err(invalid("invalid fmt chunk"));
                }
                let f = &format[8..];
                channels = u16le(&f[2..4]);
                sample_rate = u32le(&f[4..8]);
                if u16le(&f[..2]) != 1 || u16le(&f[14..16]) != 16 {
                    return Err(invalid("La 0.4b requires 16-bit PCM"));
                }
                if !(1..=2).contains(&channels) || u16le(&f[12..14]) != channels * 2 {
                    return Err(invalid("La 0.4b requires mono or stereo PCM"));
                }
            } else if start == 0 {
                return Err(invalid("expected fmt as the first WAVE chunk"));
            }
        }
        if data_len % (channels as usize * 2) != 0 {
            return Err(invalid("incomplete PCM sample frame"));
        }
        let mut data = Vec::new();
        input
            .by_ref()
            .take(data_len as u64)
            .read_to_end(&mut data)?;
        if data.len() != data_len {
            return Err(invalid("truncated PCM data"));
        }
        let samples = data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        let mut footer = Vec::new();
        input.read_to_end(&mut footer)?;
        Ok(Self {
            riff,
            format,
            channels,
            sample_rate,
            samples,
            footer,
        })
    }
}
