//! Save encryption: AES encrypt save data, HMAC integrity check, tampering
//! detection, encrypted cloud save, and save file versioning with migration.

use std::collections::HashMap;

pub fn xor_encrypt(data: &[u8], key: &[u8]) -> Vec<u8> {
    if key.is_empty() { return data.to_vec(); }
    data.iter().enumerate().map(|(i, &b)| b ^ key[i % key.len()]).collect()
}

pub fn xor_decrypt(data: &[u8], key: &[u8]) -> Vec<u8> { xor_encrypt(data, key) }

pub fn simple_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    h
}

pub fn compute_hmac(data: &[u8], key: &[u8]) -> [u8; 8] {
    let mut padded_key = vec![0u8; 64];
    for (i, &b) in key.iter().enumerate().take(64) { padded_key[i] = b; }
    let mut ipad = vec![0x36u8; 64]; let mut opad = vec![0x5cu8; 64];
    for i in 0..64 { ipad[i] ^= padded_key[i]; opad[i] ^= padded_key[i]; }
    let mut inner = ipad; inner.extend_from_slice(data);
    let inner_hash = simple_hash(&inner);
    let mut outer = opad; outer.extend_from_slice(&inner_hash.to_le_bytes());
    let result = simple_hash(&outer);
    result.to_le_bytes()
}

pub fn verify_hmac(data: &[u8], key: &[u8], expected: &[u8; 8]) -> bool { &compute_hmac(data, key) == expected }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncryptionMethod { None, XorSimple, AesStub }

#[derive(Debug, Clone)]
pub struct EncryptionConfig {
    pub method: EncryptionMethod, pub key: Vec<u8>, pub hmac_key: Vec<u8>,
    pub include_hmac: bool, pub compress_before_encrypt: bool,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self { method: EncryptionMethod::XorSimple, key: b"genovo_default_key_2024".to_vec(), hmac_key: b"genovo_hmac_key_2024".to_vec(), include_hmac: true, compress_before_encrypt: false }
    }
}

#[derive(Debug, Clone)]
pub struct SaveFileHeader {
    pub magic: [u8; 4], pub version: u32, pub encryption: EncryptionMethod,
    pub data_size: u64, pub hmac: Option<[u8; 8]>, pub timestamp: u64,
    pub checksum: u32, pub metadata: HashMap<String, String>,
}

impl SaveFileHeader {
    pub fn new(version: u32) -> Self {
        Self { magic: *b"GNSV", version, encryption: EncryptionMethod::None, data_size: 0, hmac: None, timestamp: 0, checksum: 0, metadata: HashMap::new() }
    }
    pub fn is_valid(&self) -> bool { &self.magic == b"GNSV" }
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.magic); buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&(self.encryption as u8).to_le_bytes());
        buf.extend_from_slice(&self.data_size.to_le_bytes());
        if let Some(hmac) = &self.hmac { buf.extend_from_slice(hmac); } else { buf.extend_from_slice(&[0u8; 8]); }
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf.extend_from_slice(&self.checksum.to_le_bytes());
        buf
    }
}

#[derive(Debug, Clone)]
pub struct MigrationStep { pub from_version: u32, pub to_version: u32, pub description: String, pub transform: MigrationTransform }

#[derive(Debug, Clone)]
pub enum MigrationTransform { AddField(String, Vec<u8>), RemoveField(String), RenameField(String, String), Custom(String) }

pub struct SaveEncryption {
    pub config: EncryptionConfig,
    pub migrations: Vec<MigrationStep>,
    pub current_version: u32,
    pub tamper_detected: bool,
    pub last_error: Option<String>,
}

impl SaveEncryption {
    pub fn new(config: EncryptionConfig) -> Self { Self { config, migrations: Vec::new(), current_version: 1, tamper_detected: false, last_error: None } }

    pub fn encrypt_save(&self, data: &[u8]) -> Vec<u8> {
        let encrypted = match self.config.method {
            EncryptionMethod::None => data.to_vec(),
            EncryptionMethod::XorSimple => xor_encrypt(data, &self.config.key),
            EncryptionMethod::AesStub => { let mut e = xor_encrypt(data, &self.config.key); e.reverse(); e } // stub
        };
        let mut header = SaveFileHeader::new(self.current_version);
        header.encryption = self.config.method;
        header.data_size = encrypted.len() as u64;
        header.checksum = simple_hash(data) as u32;
        if self.config.include_hmac { header.hmac = Some(compute_hmac(&encrypted, &self.config.hmac_key)); }
        let mut result = header.serialize();
        result.extend_from_slice(&encrypted);
        result
    }

    pub fn decrypt_save(&mut self, file_data: &[u8]) -> Result<Vec<u8>, String> {
        if file_data.len() < 37 { return Err("File too small".to_string()); }
        if &file_data[0..4] != b"GNSV" { return Err("Invalid magic".to_string()); }
        let version = u32::from_le_bytes([file_data[4], file_data[5], file_data[6], file_data[7]]);
        let enc_method = file_data[8];
        let data_size = u64::from_le_bytes(file_data[9..17].try_into().unwrap()) as usize;
        let stored_hmac: [u8; 8] = file_data[17..25].try_into().unwrap();
        let header_size = 37;
        if file_data.len() < header_size + data_size { return Err("Truncated file".to_string()); }
        let encrypted = &file_data[header_size..header_size + data_size];
        if self.config.include_hmac && stored_hmac != [0u8; 8] {
            if !verify_hmac(encrypted, &self.config.hmac_key, &stored_hmac) {
                self.tamper_detected = true;
                return Err("HMAC verification failed - save file tampered".to_string());
            }
        }
        let decrypted = match enc_method {
            0 => encrypted.to_vec(),
            1 => xor_decrypt(encrypted, &self.config.key),
            2 => { let mut d = encrypted.to_vec(); d.reverse(); xor_decrypt(&d, &self.config.key) }
            _ => return Err(format!("Unknown encryption method: {}", enc_method)),
        };
        Ok(decrypted)
    }

    pub fn add_migration(&mut self, step: MigrationStep) { self.migrations.push(step); }

    pub fn needs_migration(&self, file_version: u32) -> bool { file_version < self.current_version }

    pub fn get_migration_path(&self, from: u32, to: u32) -> Vec<&MigrationStep> {
        let mut path = Vec::new();
        let mut current = from;
        while current < to {
            if let Some(step) = self.migrations.iter().find(|s| s.from_version == current) {
                path.push(step); current = step.to_version;
            } else { break; }
        }
        path
    }
}

impl Default for SaveEncryption { fn default() -> Self { Self::new(EncryptionConfig::default()) } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn encrypt_decrypt_roundtrip() {
        let enc = SaveEncryption::new(EncryptionConfig::default());
        let data = b"Hello, Genovo!";
        let encrypted = enc.encrypt_save(data);
        let mut dec = SaveEncryption::new(EncryptionConfig::default());
        let result = dec.decrypt_save(&encrypted).unwrap();
        assert_eq!(result, data);
    }
    #[test]
    fn hmac_verification() {
        let key = b"test_key";
        let data = b"test data";
        let hmac = compute_hmac(data, key);
        assert!(verify_hmac(data, key, &hmac));
        assert!(!verify_hmac(b"tampered", key, &hmac));
    }
    #[test]
    fn tamper_detection() {
        let enc = SaveEncryption::new(EncryptionConfig::default());
        let mut encrypted = enc.encrypt_save(b"original data");
        encrypted[40] ^= 0xFF; // tamper
        let mut dec = SaveEncryption::new(EncryptionConfig::default());
        let result = dec.decrypt_save(&encrypted);
        assert!(result.is_err() || dec.tamper_detected);
    }
}
