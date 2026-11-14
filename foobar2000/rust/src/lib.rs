use la04::Decoder;
use std::ffi::{c_char, c_void};
use std::io::{self, Read, Seek, SeekFrom};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

#[repr(C)]
#[derive(Clone, Copy)]
struct LaIo {
    context: *mut c_void,
    read: unsafe extern "C" fn(*mut c_void, *mut u8, usize) -> isize,
    seek: unsafe extern "C" fn(*mut c_void, u64) -> i32,
    size: u64,
    check: unsafe extern "C" fn(*mut c_void) -> i32,
}

struct Reader {
    io: LaIo,
    position: u64,
}
impl Read for Reader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = unsafe { (self.io.read)(self.io.context, buffer.as_mut_ptr(), buffer.len()) };
        if count < 0 || count as usize > buffer.len() {
            return Err(io::Error::other("foobar2000 read failed"));
        }
        self.position += count as u64;
        Ok(count as usize)
    }
}
impl Seek for Reader {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let position = match from {
            SeekFrom::Start(n) => Some(n),
            SeekFrom::Current(n) => self.position.checked_add_signed(n),
            SeekFrom::End(n) => self.io.size.checked_add_signed(n),
        }
        .filter(|&n| n <= self.io.size)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid seek offset"))?;
        if unsafe { (self.io.seek)(self.io.context, position) } != 0 {
            return Err(io::Error::other("foobar2000 seek failed"));
        }
        self.position = position;
        Ok(position)
    }
}

#[repr(C)]
#[derive(Default)]
struct LaInfo {
    sample_frames: u64,
    sample_rate: u32,
    channels: u16,
    high: u8,
    seek_table: u8,
}

struct LaDecoder {
    io: LaIo,
    decoder: Decoder<Reader>,
    frame: Vec<i16>,
    poisoned: bool,
}

const ERROR_SIZE: usize = 256;
unsafe fn error(out: *mut c_char, message: &str) {
    unsafe {
        let bytes = message.as_bytes();
        let count = bytes.len().min(ERROR_SIZE - 1);
        ptr::copy_nonoverlapping(bytes.as_ptr(), out.cast(), count);
        *out.add(count) = 0;
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn la_open(io: LaIo, info: *mut LaInfo, err: *mut c_char) -> *mut LaDecoder {
    unsafe {
        match catch_unwind(AssertUnwindSafe(|| {
            let decoder = Decoder::new(Reader { io, position: 0 })?;
            let metadata = decoder.info();
            *info = LaInfo {
                sample_frames: metadata.sample_frames,
                sample_rate: metadata.sample_rate,
                channels: metadata.channels,
                high: metadata.high as u8,
                seek_table: metadata.seek_table as u8,
            };
            Ok::<_, io::Error>(Box::into_raw(Box::new(LaDecoder {
                decoder,
                io,
                frame: Vec::new(),
                poisoned: false,
            })))
        })) {
            Ok(Ok(handle)) => handle,
            Ok(Err(e)) => {
                error(err, &e.to_string());
                ptr::null_mut()
            }
            Err(_) => {
                error(err, "Rust decoder panic while opening LA04");
                ptr::null_mut()
            }
        }
    }
}

unsafe fn operation(
    handle: *mut LaDecoder,
    err: *mut c_char,
    action: impl FnOnce(&mut LaDecoder) -> io::Result<i32>,
) -> i32 {
    unsafe {
        let handle = &mut *handle;
        if handle.poisoned {
            error(err, "decoder needs reopening after an error");
            return -1;
        }
        match catch_unwind(AssertUnwindSafe(|| action(handle))) {
            Ok(Ok(result)) => result,
            failure => {
                handle.poisoned = true;
                match failure {
                    Ok(Err(e)) => error(err, &e.to_string()),
                    _ => error(err, "Rust decoder panic while decoding LA04"),
                }
                -1
            }
        }
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn la_read(
    handle: *mut LaDecoder,
    samples: *mut *const i16,
    count: *mut usize,
    err: *mut c_char,
) -> i32 {
    unsafe {
        *samples = ptr::null();
        *count = 0;
        operation(handle, err, |handle| {
            let io = handle.io;
            handle.frame = handle.decoder.read_frame_with_cancel(|| {
                if (io.check)(io.context) == 0 {
                    Ok(())
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "decoding cancelled",
                    ))
                }
            })?;
            *count = handle.frame.len();
            *samples = handle.frame.as_ptr();
            Ok(i32::from(!handle.frame.is_empty()))
        })
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn la_seek(handle: *mut LaDecoder, sample_frame: u64, err: *mut c_char) -> i32 {
    unsafe {
        operation(handle, err, |handle| {
            handle.frame.clear();
            handle.decoder.seek(sample_frame)?;
            Ok(0)
        })
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn la_close(handle: *mut LaDecoder) {
    unsafe {
        if !handle.is_null() {
            drop(Box::from_raw(handle));
        }
    }
}
