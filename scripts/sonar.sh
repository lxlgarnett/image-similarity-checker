#!/bin/sh
# SonarQube 本地扫描：读取 token →（运行测试生成覆盖率）→ 上传到 http://192.168.11.4:29000
# token 读取顺序（后者覆盖前者）：~/.config/sonarqube/env → ./.env.sonar → 已有的环境变量
# 版本号取当前提交短 SHA：新代码基线 = 上一次扫描的提交，建议 commit 之后再扫描。
set -e
cd "$(dirname "$0")/.."

_preset_token="${SONAR_TOKEN:-}"
for f in "$HOME/.config/sonarqube/env" ./.env.sonar; do
  if [ -f "$f" ]; then set -a; . "$f"; set +a; fi
done
[ -n "$_preset_token" ] && SONAR_TOKEN="$_preset_token"
if [ -z "${SONAR_TOKEN:-}" ]; then
  echo "SONAR_TOKEN 未设置：请写入 ~/.config/sonarqube/env（或本仓库的 .env.sonar）" >&2
  exit 1
fi
export SONAR_TOKEN

EXTRA=""
# 暂无测试，不上报覆盖率

# shellcheck disable=SC2086
npx --yes @sonar/scan@5 -Dsonar.projectVersion="$(git rev-parse --short HEAD)" $EXTRA
