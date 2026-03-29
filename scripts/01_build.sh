#!/bin/bash
# Step 1: Build the Java fat jar
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rl_common.sh"

cd "$PROJECT_ROOT"
echo "Building forge jar..."
mvn package -pl forge-gui-desktop -am -Denforcer.skip=true -Dcheckstyle.skip=true -DskipTests -q
JAR_PATH="$(rl_resolve_jar)"
echo "Build complete."
echo "Jar: $JAR_PATH"
