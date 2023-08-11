// Perlin noise implementation inspired by Adrian Biagioli's C# version.
// Original blog post: https://adrianb.io/2014/08/09/perlinnoise.html

const REPEAT: f64 = 0.0;

/*
Hash lookup table as defined by Ken Perlin:
This is a randomly arranged array of all numbers from 0-255 inclusive.
*/
const PERMUTATION: [usize; 256] = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69,
    142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219,
    203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230,
    220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76,
    132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173,
    186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206,
    59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163,
    70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
    178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162,
    241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204,
    176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141,
    128, 195, 78, 66, 215, 61, 156, 180,
];

// Doubled permutation to avoid overflow
fn generate_permutation() -> [usize; 512] {
    let mut p: [usize; 512] = [0; 512];
    for i in 0..512 {
        p[i] = PERMUTATION[i % 255];
    }
    p
}

lazy_static! {
    static ref P: [usize; 512] = generate_permutation();
}

// Utility function to perform linear interpolation
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/*
Fade function as defined by Ken Perlin:
This function eases coordinate values towards integral values, resulting in a smoother final output.
The function is defined as: 6t^5 - 15t^4 + 10t^3.
*/
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

// Utility function to calculate the gradient of the cube corner
fn grad(hash: usize, x: f64, y: f64, z: f64) -> f64 {
    let h = hash & 15; // Extract the first 4 bits of the hashed value (15 == 0b1111)

    let u = if h < 8 { x } else { y }; // If the most significant bit (MSB) of the hash is 0, then set u = x; otherwise, set u = y.

    let v = if h < 4 {
        y // If the first and second significant bits are 0, set v = y.
    } else if h == 12 || h == 14 {
        x // If the first and second significant bits are 1, set v = x.
    } else {
        z // If the first and second significant bits are not equal (0/1, 1/0), set v = z.
    };

    // Use the last 2 bits to decide if u and v are positive or negative. Then return their sum.
    (if (h & 1) == 0 { u } else { -u }) + (if (h & 2) == 0 { v } else { -v })
}

// // An alternate implementation of the grad function.
// // Source: http://riven8192.blogspot.com/2010/08/calculate-perlinnoise-twice-as-fast.html
// fn grad(hash: usize, x: f64, y: f64, z: f64) -> f64 {
//     match hash & 0xF {
//         0x0 => x + y,
//         0x1 => -x + y,
//         0x2 => x - y,
//         0x3 => -x - y,
//         0x4 => x + z,
//         0x5 => -x + z,
//         0x6 => x - z,
//         0x7 => -x - z,
//         0x8 => y + z,
//         0x9 => -y + z,
//         0xA => y - z,
//         0xB => -y - z,
//         0xC => y + x,
//         0xD => -y + z,
//         0xE => y - x,
//         0xF => -y - z,
//         _ => unreachable!(), // This case should never happen
//     }
// }

fn inc(mut num: usize) -> usize {
    num = num + 1;
    if REPEAT > 0.0 {
        num %= REPEAT as usize;
    }
    num
}

// The perlin function with three-dimensional coordinates
fn perlin(mut x: f64, mut y: f64, mut z: f64) -> f64 {
    if REPEAT > 0.0 {
        x = x % REPEAT;
        y = y % REPEAT;
        z = z % REPEAT;
    }

    let xi = x as usize & 255;
    let yi = y as usize & 255;
    let zi = z as usize & 255;

    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    let aaa = P[P[P[xi] + yi] + zi];
    let aba = P[P[P[xi] + inc(yi)] + zi];
    let aab = P[P[P[xi] + yi] + inc(zi)];
    let abb = P[P[P[xi] + inc(yi)] + inc(zi)];
    let baa = P[P[P[inc(xi)] + yi] + zi];
    let bba = P[P[P[inc(xi)] + inc(yi)] + zi];
    let bab = P[P[P[inc(xi)] + yi] + inc(zi)];
    let bbb = P[P[P[inc(xi)] + inc(yi)] + inc(zi)];

    let x1 = lerp(grad(aaa, xf, yf, zf), grad(baa, xf - 1.0, yf, zf), u);
    let x2 = lerp(
        grad(aba, xf, yf - 1.0, zf),
        grad(bba, xf - 1.0, yf - 1.0, zf),
        u,
    );
    let y1 = lerp(x1, x2, v);

    let x1 = lerp(
        grad(aab, xf, yf, zf - 1.0),
        grad(bab, xf - 1.0, yf, zf - 1.0),
        u,
    );
    let x2 = lerp(
        grad(abb, xf, yf - 1.0, zf - 1.0),
        grad(bbb, xf - 1.0, yf - 1.0, zf - 1.0),
        u,
    );
    let y2 = lerp(x1, x2, v);

    (lerp(y1, y2, w) + 1.0) / 2.0
}

// Sine color map function
fn sine_map(value: f64, color_period: f64) -> f64 {
    let mapped_value = (value * 2.0 * std::f64::consts::PI / color_period).sin();
    mapped_value
}

// The perlin function with three-dimensional coordinates and multiple octaves
pub fn perlin_with_octaves(
    x: f64,
    y: f64,
    t: f64,
    num_octaves: usize,
    wavelength_x: f64,
    wavelength_y: f64,
    wavelength_z: f64,
    color_period: f64,
    epsilon: f64,
) -> f64 {
    let mut noise = 0.0;

    for l in 0..num_octaves {
        let frequency_x = 2.0_f64.powf(l as f64) / wavelength_x;
        let frequency_y = 2.0_f64.powf(l as f64) / wavelength_y;
        let frequency_z = 2.0_f64.powf(l as f64) / wavelength_z;

        noise += perlin(x * frequency_x, y * frequency_y, t * frequency_z);
    }

    // Apply sine color map
    noise = sine_map(noise, color_period);

    // Scale to the range [0, 1]
    noise = (noise + 1.0) / 2.0;

    // Scale to the range [0, 10]
    noise *= epsilon;

    noise
}

#[cfg(test)]
mod tests {
    use super::perlin_with_octaves;
    use std::{fs::File, io::Write};

    #[test]
    fn test_perlin_noise_generation() {
        // Define the size of the grid and the step size
        let grid_size = 64;
        let step_size = 0.1;

        // Open the output file
        let mut file =
            File::create("/home/ala22014/u3d/python/perlin-tests/perlin-noise-3d-data.csv")
                .expect("Failed to create file");

        // Write the header row to the CSV file
        writeln!(file, "x,y,z,perlin_value").expect("Failed to write to file");

        // Generate and write the Perlin noise data to the CSV file
        for x in 0..grid_size {
            for y in 0..grid_size {
                for z in 0..grid_size {
                    let perlin_value = perlin_with_octaves(
                        x as f64 * step_size,
                        y as f64 * step_size,
                        z as f64 * step_size,
                        8,
                        4.0,
                        2.0,
                        8.0,
                        2.0,
                    );
                    let line = format!("{},{},{},{}", x, y, z, perlin_value);
                    writeln!(file, "{}", line).expect("Failed to write to file");
                }
            }
        }
    }
}
