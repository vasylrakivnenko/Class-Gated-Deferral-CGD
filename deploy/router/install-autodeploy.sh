#!/usr/bin/env bash
# One-time bootstrap on the box. Installs the timer that keeps router.zadum.ai
# on origin/main. Run once as root:
#
#     cd /root/projects/zadumai && git pull --ff-only && bash deploy/router/install-autodeploy.sh
#
# After this, every push to main reaches the box within a minute. Nothing else
# needs doing on a change: the script pulls, restarts, health-checks, and rolls
# back if the restarted service does not answer.
set -euo pipefail
REPO=/root/projects/zadumai

[ -d "$REPO/.git" ] || { echo "no checkout at $REPO" >&2; exit 1; }
chmod +x "$REPO/deploy/router/deploy.sh"
install -m 644 "$REPO/deploy/router/zadum-deploy.service" /etc/systemd/system/
install -m 644 "$REPO/deploy/router/zadum-deploy.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now zadum-deploy.timer

echo
echo "auto-deploy is on. Useful afterwards:"
echo "  systemctl list-timers zadum-deploy    # when it next runs"
echo "  journalctl -u zadum-deploy -f         # what it did"
echo "  systemctl start zadum-deploy          # deploy right now, don't wait"
echo "  systemctl disable --now zadum-deploy.timer   # turn it off"
