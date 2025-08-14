#!/usr/bin/env bash
# Mount LSDF-Share an einen Zielordner – ohne prefixpath, mit Fallback via Bind-Mount.
# Nutzung: bash mount_lsdf.sh
set -euo pipefail

# ===== Konfiguration (bei Bedarf anpassen) =====
MOUNT_DIR="${MOUNT_DIR:-/home/ws/hp5743/projects/forecasts_pinter/pinter_paper_2/GermanBuildingData/02_forecast/mount}"
SHARE="//os.lsdf.kit.edu/kit"
SUB="iai/projects/iai-aida/Daten_Pinter/forecasts_interval_scheduling"
SMB_VERS="${SMB_VERS:-3.0}"  # ggf. 2.1 oder 2.0
BASE="${MOUNT_DIR}.base"     # Root-Share-Mount für Bind-Mount-Fallback

# ===== Sudo-Check =====
if [[ $EUID -ne 0 ]]; then
  echo "Sudo-Rechte erforderlich."
  sudo -v || { echo "Sudo fehlgeschlagen."; exit 1; }
fi

# ===== Zugangsdaten abfragen =====
read -p "LSDF Benutzername: " username
read -s -p "LSDF Passwort: " password; echo
read -p "Domain (optional, z.B. KIT) [leer lassen falls keine]: " domain; domain=${domain:-}

# ===== Mountpoints vorbereiten =====
sudo umount "$MOUNT_DIR" 2>/dev/null || true
sudo umount "$BASE" 2>/dev/null || true
sudo mkdir -p "$MOUNT_DIR" "$BASE"

# ===== temporäre Credentials-Datei =====
cred="$(mktemp)"
chmod 600 "$cred"
{
  printf 'username=%s\n' "$username"
  printf 'password=%s\n' "$password"
  [[ -n "$domain" ]] && printf 'domain=%s\n' "$domain"
} > "$cred"
cleanup() { shred -u "$cred" 2>/dev/null || rm -f "$cred"; }
trap cleanup EXIT

OPTS="rw,vers=${SMB_VERS},credentials=${cred},uid=$(id -u),gid=$(id -g),file_mode=0644,dir_mode=0755,iocharset=utf8,sec=ntlmssp"

echo "==> Versuche direkten UNC-Subpfad zu mounten (ohne trailing slash)..."
if sudo mount -t cifs "${SHARE}/${SUB}" "$MOUNT_DIR" -o "$OPTS"; then
  echo "UNC-Mount erfolgreich: ${SHARE}/${SUB} -> $MOUNT_DIR"
else
  rc=$?
  echo "UNC-Mount fehlgeschlagen (rc=$rc). Fallback: Root-Share + Bind-Mount..."
  echo "-> Root-Share mounten: ${SHARE} -> $BASE"
  sudo mount -t cifs "${SHARE}" "$BASE" -o "$OPTS"

  if [[ ! -d "$BASE/$SUB" ]]; then
    echo "Unterpfad existiert nicht: $BASE/$SUB"
    echo "Bitte Share/Teilpfad prüfen."
    exit 2
  fi

  echo "-> Bind-Mount: $BASE/$SUB -> $MOUNT_DIR"
  sudo mount --bind "$BASE/$SUB" "$MOUNT_DIR"
fi

# ===== Verifikation & Schreibtest =====
echo "==> Prüfe Mount:"
mount | grep -F "$MOUNT_DIR" || true
echo "Inhalt (erste Einträge):"
ls -la "$MOUNT_DIR" | head || true

echo "==> Schreibtest:"
if : >"$MOUNT_DIR/.writetest" 2>/dev/null; then
  rm -f "$MOUNT_DIR/.writetest"
  echo "OK: $MOUNT_DIR ist für UID $(id -u) schreibbar."
else
  echo "Hinweis: $MOUNT_DIR ist nicht schreibbar für UID $(id -u)."
  echo "→ Serverrechte prüfen oder mit anderen Mount-Optionen (z.B. andere Domain) erneut versuchen."
fi

echo "Fertig."
