#[cfg(unix)]
fn main() {
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let launcher = manifest_dir.join("lplearn.data/scripts/lplearn");
    let metadata = std::fs::metadata(&launcher)
        .unwrap_or_else(|error| panic!("cannot inspect {}: {error}", launcher.display()));
    let permissions = std::fs::Permissions::from_mode(metadata.permissions().mode() | 0o111);
    std::fs::set_permissions(&launcher, permissions)
        .unwrap_or_else(|error| panic!("cannot make {} executable: {error}", launcher.display()));
    println!("cargo:rerun-if-changed=lplearn.data/scripts/lplearn");
}

#[cfg(not(unix))]
fn main() {
    println!("cargo:rerun-if-changed=lplearn.data/scripts/lplearn");
}
