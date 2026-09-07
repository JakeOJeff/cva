#!/bin/sh
set -e

# /data is a mounted volume; its ownership comes from the host, so fix it here
# rather than at build time.
mkdir -p /data/uploads /data/results /data/huggingface
chown -R cva:cva /data 2>/dev/null || true

if [ -z "$CVA_PASSWORD" ]; then
  echo "  [!] CVA_PASSWORD is unset - this server has NO authentication."
  echo "      Do not expose it publicly like this."
fi
if [ -z "$HF_TOKEN" ]; then
  echo "  [!] HF_TOKEN is unset - diarization will fail. See the README."
fi

exec su -s /bin/sh -c "exec $*" cva
