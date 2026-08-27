use std::fs;
use std::io;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TranslationPair {
    pub source: String,
    pub target: String,
}


pub fn load_dataset<P: AsRef<Path>>(
    data_dir: P,
) -> io::Result<Vec<TranslationPair>> {
    let data_dir = data_dir.as_ref();

    let mut pairs = Vec::new();

    let mut index = 1;

    loop {
        let source_path = data_dir.join(format!("{}.txt", index));
        let target_path = data_dir.join(format!("{}p.txt", index));

        // Stop when the next source file does not exist.
        if !source_path.exists() {
            break;
        }

        if !target_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "Missing target file for {}: {}",
                    index,
                    target_path.display()
                ),
            ));
        }

        let source = fs::read_to_string(&source_path)?;
        let target = fs::read_to_string(&target_path)?;

        pairs.push(TranslationPair {
            source: source.trim().to_string(),
            target: target.trim().to_string(),
        });

        index += 1;
    }

    Ok(pairs)
}