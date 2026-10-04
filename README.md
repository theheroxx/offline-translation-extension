Quick Start

1. Clone the repository
git clone https://github.com/theheroxx/offline-translation-extension.git
cd offline-translation-extension

2. Verify Rust
rustc --version
cargo --version

3. Verify NVIDIA GPU
nvidia-smi

4. Verify CUDA
nvcc --version

5. Build the project
cargo check

If the project compiles successfully:

cargo build

6. Start training
cargo run

The program initializes the CUDA device and starts the Transformer training pipeline.