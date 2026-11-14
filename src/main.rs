use la04::{Encoder, Options};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut options = Options::default();
    let mut overwrite = false;
    let mut console_in = false;
    let mut output_raw = false;
    let mut decode = false;
    let mut paths = Vec::new();
    let mut flags = true;
    for arg in std::env::args_os().skip(1) {
        match arg.to_str() {
            Some("--") if flags => flags = false,
            Some("-high" | "--high") if flags => options.high = true,
            Some("-noseek" | "--no-seek") if flags => options.seeking = false,
            Some("-overwrite" | "--overwrite") if flags => overwrite = true,
            Some("-cin" | "--stdin") if flags => console_in = true,
            Some("-outputraw" | "--output-raw") if flags => output_raw = true,
            Some("-d" | "--decode") if flags => decode = true,
            Some("-h" | "--help") if flags => {
                println!(
                    "La 0.4b native Rust encoder/decoder\nUsage: la-rs [-high] [-noseek] [-overwrite] [-cin] [-outputraw] input.wav [output.la]\n       la-rs -d [-overwrite] input.la [output.wav]\nUse - for input from stdin. Encoding output must be a seekable file.\n-outputraw writes each transformed stage as output.la.stageN.raw."
                );
                return Ok(());
            }
            Some(s) if flags && s.starts_with('-') && s != "-" => {
                return Err(format!("unknown option: {s}").into());
            }
            _ => paths.push(PathBuf::from(arg)),
        }
    }
    if paths.is_empty() || paths.len() > 2 {
        return Err(
            "usage: la-rs [-high] [-noseek] [-overwrite] [-cin] input.wav [output.la]".into(),
        );
    }
    let input = &paths[0];
    let output = paths
        .get(1)
        .cloned()
        .unwrap_or_else(|| input.with_extension(if decode { "wav" } else { "la" }));
    if !console_in
        && input.as_os_str() != "-"
        && output.exists()
        && input.canonicalize()? == output.canonicalize()?
    {
        return Err("input and output must be different files".into());
    }

    if decode {
        if output_raw {
            return Err("-outputraw is only supported for encoding".into());
        }
        let start = std::time::Instant::now();

        let mut restored = Vec::new();
        let size = if console_in || input.as_os_str() == "-" {
            la04::decode(io::stdin().lock(), &mut restored)?
        } else {
            la04::decode(File::open(input)?, &mut restored)?
        };
        let mut file = if overwrite {
            File::create(&output)?
        } else {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&output)?
        };
        file.write_all(&restored)?;
        eprintln!(
            "Decoded: {size} bytes, {:.2}s",
            start.elapsed().as_secs_f64()
        );
        return Ok(());
    }
    let wave = if console_in || input.as_os_str() == "-" {
        la04::wave::Wave::read(io::stdin().lock())?
    } else {
        la04::wave::Wave::read(File::open(input)?)?
    };
    let source_size =
        20 + wave.format.len() as u64 + wave.samples.len() as u64 * 2 + wave.footer.len() as u64;
    let file = if overwrite {
        File::create(&output)?
    } else {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)?
    };
    if !output_raw {
        let start = std::time::Instant::now();
        let size = la04::encode_wave(wave, file, options)?;
        eprintln!(
            "Finished: {} bytes, {:.1}% of WAV, {:.2}s",
            size,
            size as f64 * 100.0 / source_size as f64,
            start.elapsed().as_secs_f64()
        );
        return Ok(());
    }
    let mut encoder = Encoder::from_wave(wave, file, options)?;
    let total = encoder.get_total_frames();
    let start = std::time::Instant::now();
    let mut raw_files: Vec<File> = Vec::new();
    for frame in 0..total {
        let mut raw_error = None;
        encoder.create_next_frame_observed(|stage, samples| {
            if !output_raw || raw_error.is_some() {
                return;
            }
            let result = (|| -> io::Result<()> {
                if raw_files.len() <= stage {
                    let mut name = output.as_os_str().to_os_string();
                    name.push(format!(".stage{stage}.raw"));
                    let file = if overwrite {
                        File::create(name)?
                    } else {
                        OpenOptions::new().write(true).create_new(true).open(name)?
                    };
                    raw_files.push(file);
                }
                let mut data = Vec::with_capacity(samples.len() * 2);
                for sample in samples {
                    data.extend_from_slice(&sample.to_le_bytes());
                }
                raw_files[stage].write_all(&data)
            })();
            if let Err(error) = result {
                raw_error = Some(error);
            }
        })?;
        if let Some(error) = raw_error {
            return Err(error.into());
        }
        eprint!("\rEncoding: {}/{} frames", frame + 1, total);
    }
    let size = encoder.write_footer()?;
    eprintln!(
        "\rFinished: {} bytes, {:.1}% of WAV, {:.2}s",
        size,
        size as f64 * 100.0 / source_size as f64,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("la-rs: {error}");
        std::process::exit(1);
    }
}
