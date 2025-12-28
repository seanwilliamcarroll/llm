#!/usr/bin/env bash
set -euo pipefail

HOOK_DIR="$(git rev-parse --git-dir)/hooks"
HOOK_PATH="$HOOK_DIR/pre-commit"

mkdir -p "$HOOK_DIR"

cat > "$HOOK_PATH" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Colors for readability
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
NC="\033[0m"

log() {
  echo -e "${GREEN}==>${NC} $1"
}

warn() {
  echo -e "${YELLOW}==>${NC} $1"
}

fail() {
  echo -e "${RED}ERROR:${NC} $1"
  exit 1
}

# Ensure we're at repo root
cd "$(git rev-parse --show-toplevel)"

########################################
# Rust toolchain sanity checks
########################################

if ! command -v cargo >/dev/null; then
  fail "cargo not found"
fi

########################################
# rustfmt
########################################
log "Running rustfmt..."
cargo fmt --all -- --check

########################################
# clippy (warnings = errors)
########################################
log "Running clippy..."
cargo clippy --all-targets --all-features -- -D warnings

########################################
# Debug build
########################################
log "Building (debug)..."
cargo build --workspace

########################################
# Release build
########################################
log "Building (release)..."
cargo build --workspace --release

########################################
# Unused dependencies (cargo-udeps)
########################################
if ! cargo +nightly udeps --help >/dev/null 2>&1; then
  warn "cargo-udeps not found. Installing nightly + cargo-udeps..."
  rustup toolchain install nightly
  cargo +nightly install cargo-udeps
fi

log "Checking unused dependencies..."
cargo +nightly udeps --workspace

########################################
# Docs build
########################################
log "Building docs..."
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace

########################################
# check formatting of Cargo.toml
########################################
if ! command -v taplo >/dev/null 2>&1; then
  warn "taplo-cli not found. Installing..."
  cargo install taplo-cli --locked
else
  log "taplo-cli found, skipping installation."
fi

log "Checking Cargo.toml formatting..."
taplo fmt --check

log "All checks passed 🎉"
EOF

chmod +x "$HOOK_PATH"

echo "✅ Pre-commit hook installed successfully."
