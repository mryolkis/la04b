use std::sync::OnceLock;

pub struct Tables {
    pub learning_rate: Box<[u16; 65536]>,
    pub gradient: Box<[i8; 32768]>,
    pub short_gradient: Box<[i8; 16384]>,
}
pub fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| Tables {
        learning_rate: (0..65536)
            .map(|i| {
                let x = i as f64;
                let divisor =
                    ((x * ((i / 25 + 3) as f64).ln() * (x + 3.0).ln() * 0.2857142857142857 + 35.0)
                        * 0.05)
                        .max(1.0);
                (32768.0 / divisor) as u16
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
            .try_into()
            .unwrap(),

        gradient: (-16384i32..16384)
            .map(|i| {
                let a = i.abs();
                if a == 0 {
                    1
                } else {
                    (((a / 8) as f64 + std::f64::consts::E).ln() * ((a * 5) as f64).ln()) as i32
                        as i8
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
            .try_into()
            .unwrap(),

        short_gradient: (-8192i32..8192)
            .map(|i| {
                if i == 0 {
                    1
                } else {
                    ((i.abs() as f64 * 1.25).ln().powi(2) + 1.0) as i32 as i8
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
            .try_into()
            .unwrap(),
    })
}

pub fn short_delta(value: i32) -> i16 {
    let magnitude = match value.abs() {
        0..=1024 => 1,
        1025..=3000 => 2,
        3001..=5000 => 3,
        _ => 4,
    };
    (-value.signum() * magnitude) as i16
}
pub fn delta(value: i32) -> i16 {
    let a = value.abs();
    let magnitude = match a {
        0..=400 => 16,
        401..=700 => 32,
        701..=1224 => 48,
        1225..=4096 => 64,
        4097..=16384 => 80,
        _ => 96,
    };
    let mut d = -value.signum() * magnitude;
    if a < 16 {
        d /= 2;
    }
    if a < 4 {
        d /= 2;
    }
    d as i16
}
pub fn high_delta(value: i16) -> i16 {
    let value = value as i32;
    let magnitude = match value.abs() {
        0..=400 => 4,
        401..=700 => 8,
        701..=1224 => 12,
        1225..=4096 => 16,
        4097..=16384 => 20,
        _ => 24,
    };
    (-value.signum() * magnitude) as i16
}

pub fn sample_scale(average: i32) -> i32 {
    ((36.0 / (average as f64 * 0.20833333333333334 + 1.5)).max(1.0) * 16384.0) as i32
}
