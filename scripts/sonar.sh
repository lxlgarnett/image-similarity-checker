#!/bin/sh
# SonarQube 本地扫描：读取配置 →（运行测试生成覆盖率）→ 上传到 $SONAR_HOST_URL
# 配置读取顺序（后者覆盖前者）：~/.config/sonarqube/env → ./.env.sonar → 已有的环境变量
# 需要的变量：SONAR_HOST_URL（服务器地址）、SONAR_TOKEN
# 版本号取当前提交短 SHA：新代码基线 = 上一次扫描的提交，建议 commit 之后再扫描。
set -e
cd "$(dirname "$0")/.."

_preset_token="${SONAR_TOKEN:-}"
_preset_host="${SONAR_HOST_URL:-}"
for f in "$HOME/.config/sonarqube/env" ./.env.sonar; do
  if [ -f "$f" ]; then set -a; . "$f"; set +a; fi
done
[ -n "$_preset_token" ] && SONAR_TOKEN="$_preset_token"
[ -n "$_preset_host" ] && SONAR_HOST_URL="$_preset_host"
for v in SONAR_HOST_URL SONAR_TOKEN; do
  eval "_val=\${$v:-}"
  if [ -z "$_val" ]; then
    echo "$v 未设置：请写入 ~/.config/sonarqube/env（或本仓库的 .env.sonar）" >&2
    exit 1
  fi
done
export SONAR_TOKEN SONAR_HOST_URL

EXTRA=""
# 暂无测试，不上报覆盖率

# shellcheck disable=SC2086
npx --yes @sonar/scan@5 -Dsonar.projectVersion="$(git rev-parse --short HEAD)" $EXTRA
