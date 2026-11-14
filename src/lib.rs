mod constants;
pub mod crc;
mod decoder;
mod encoder;
mod entropy;
mod playback;
mod predictor;
mod tables;
pub mod wave;

pub use decoder::decode;
pub use encoder::{Encoder, FRAME_SIZE, Options, encode, encode_wave};
pub use playback::{AudioInfo, Decoder};

#[cfg(test)]
mod tests;
