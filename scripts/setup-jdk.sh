#!/usr/bin/env bash
# Script: setup-jdk.sh
# Purpose: detect installed JDK, print instructions to set JAVA_HOME, and optionally set JAVA_HOME for this shell.

set -euo pipefail

print_help() {
  cat <<'EOF'
Usage: setup-jdk.sh [--auto-export]

Detects JAVA_HOME and java executable. If not found, prints instructions to install an OpenJDK
and how to set JAVA_HOME. If --auto-export is provided, the script will set JAVA_HOME and PATH
for the current shell (only affects this shell session).
EOF
}

AUTO_EXPORT=false
while [ ${#} -gt 0 ]; do
  case "$1" in
    --auto-export) AUTO_EXPORT=true; shift ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "Unknown option: $1"; print_help; exit 1 ;;
  esac
done

COMMON_JAVA_PATHS=(
  "/usr/lib/jvm/java-17-openjdk-amd64"
  "/usr/lib/jvm/java-11-openjdk-amd64"
  "/usr/lib/jvm/java-21-openjdk-amd64"
  "/usr/lib/jvm/java-17-openjdk"
  "/usr/lib/jvm/java-11-openjdk"
  "/usr/lib/jvm/jdk-17"
  "/usr/lib/jvm/openjdk-17"
  "/usr/lib/jvm/openjdk-21"
  "/opt/java/openjdk"
)

# check whether a java binary is present
if command -v java >/dev/null 2>&1; then
  JAVA_BIN_PATH=$(command -v java)
  echo "Found java executable at: $JAVA_BIN_PATH"
  if [ -n "${JAVA_HOME:-}" ]; then
    echo "JAVA_HOME is currently set: $JAVA_HOME"
  else
    # try to deduce JAVA_HOME from java command path
    JAVA_HOME_DERIVED=$(dirname $(dirname "$JAVA_BIN_PATH"))
    echo "Suggested JAVA_HOME value: $JAVA_HOME_DERIVED"
    if $AUTO_EXPORT; then
      export JAVA_HOME="$JAVA_HOME_DERIVED"
      export PATH="$JAVA_HOME/bin:$PATH"
      echo "JAVA_HOME exported for this session"
    fi
  fi
  java -version || true
  exit 0
fi

# If no java in PATH, search for common JDK paths
for p in "${COMMON_JAVA_PATHS[@]}"; do
  if [ -x "$p/bin/java" ]; then
    echo "Found java at: $p/bin/java"
    echo "Please set JAVA_HOME to: $p"
    if $AUTO_EXPORT; then
      export JAVA_HOME="$p"
      export PATH="$JAVA_HOME/bin:$PATH"
      echo "JAVA_HOME exported for this session"
    fi
    echo "java -version:"
    $p/bin/java -version || true
    exit 0
  fi
done

# Fallback: print instructions
cat <<'EOF'
ERROR: No Java binary found in PATH or in common locations.

To install OpenJDK (Debian/Ubuntu), run:
  sudo apt update
  sudo apt install -y openjdk-17-jdk

After install, set JAVA_HOME in your shell profile (e.g., ~/.bashrc):
  export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
  export PATH=$JAVA_HOME/bin:$PATH

Or use SDKMAN/jenv per your preference.

You can then re-run this script with --auto-export to set variables for this session.
EOF

exit 1
