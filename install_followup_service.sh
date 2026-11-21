#!/usr/bin/env bash
# Run from the user's normal terminal, where the user systemd manager is accessible.
set -Eeuo pipefail
cd "$(dirname "$0")"
mkdir -p "$HOME/.config/systemd/user" "$PWD/logs"
if systemctl --user is-active --quiet qkd-followup.service; then
    echo 'qkd-followup is already active; leaving it running.'
    exit 0
fi
install -m 644 systemd/qkd-followup.service "$HOME/.config/systemd/user/qkd-followup.service"
systemctl --user daemon-reload
systemctl --user enable --now qkd-followup.service
systemctl --user status qkd-followup.service --no-pager
