
// pub mod input_module {
pub struct Input {
    pub x1: f32,
    pub x2: f32,
    pub x3: f32,
    pub x4: f32,
}

impl Input {
    pub fn to_vec(&self) -> Vec<f32> {
        vec![self.x1, self.x2, self.x3, self.x4]
    }

    pub fn to_matrix(&self) -> Vec<Vec<f32>> {
        vec![self.to_vec()]
    }
}

pub struct Labels {
    pub x1: f32,
    pub x2: f32,
    pub x3: f32,
    pub x4: f32,
}

impl Labels {
    pub fn to_vec(&self) -> Vec<f32> {
        vec![self.x1, self.x2, self.x3, self.x4]
    }
}

pub fn normalize_input(input: &Input) -> Input {
    let values = input.to_vec();
    let n = values.len() as f32;
    
    let mean: f32 = values.iter().sum::<f32>() / n;
    let variance: f32 = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / n;
    let std = variance.sqrt() + 1e-8;
    
    Input {
        x1: (input.x1 - mean) / std,
        x2: (input.x2 - mean) / std,
        x3: (input.x3 - mean) / std,
        x4: (input.x4 - mean) / std,
    }
}

pub fn minmax_normalize(input: &Input) -> Input {
    let values = input.to_vec();
    let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;
    
    Input {
        x1: (input.x1 - min_val) / range,
        x2: (input.x2 - min_val) / range,
        x3: (input.x3 - min_val) / range,
        x4: (input.x4 - min_val) / range,
    }
}
// }