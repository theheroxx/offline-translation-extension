use burn::optim::{AdamConfig, AdamWConfig, SgdConfig};

pub fn sgd_config() -> SgdConfig {
    SgdConfig::new()
}


pub fn adam_config() -> AdamConfig {
    AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-5)
}


pub fn adamw_config(weight_decay: f32) -> AdamWConfig {
    AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-5)
        .with_weight_decay(weight_decay)
}