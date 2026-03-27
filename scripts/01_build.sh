#!/bin/bash
# Step 1: Build the Java fat jar
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
echo "Building forge jar..."
mvn package -pl forge-gui-desktop -am -Denforcer.skip=true -Dcheckstyle.skip=true -DskipTests -q
echo "Build complete."
