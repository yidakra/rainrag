#!/usr/bin/env bash
# Set up the mutual-TLS origin listener the external gateway connects to.
#
# Why this exists: after access moved to rag.tvrain.io the gateway-to-origin
# hop was plain HTTP on port 8080, carrying the login form in cleartext on the
# office LAN (CWE-319, raised by review on PR #88). Port 8443 replaces it with
# TLS that also *authenticates* the gateway: nginx refuses any client that does
# not present a certificate signed by this CA, so the allow-by-source-address
# rule stops being the only control.
#
# This script never generates the gateway's private key. Alexander sends a CSR,
# we sign it, and only the certificate travels back. A private key must not be
# transmitted, and there is no reason for it to leave his host.
#
#   sudo bash deploy/nginx/origin-mtls-setup.sh init     # CA + origin cert
#   sudo bash deploy/nginx/origin-mtls-setup.sh sign FILE.csr   # sign his CSR
#   sudo bash deploy/nginx/origin-mtls-setup.sh selftest        # our own probe cert
#
set -euo pipefail

DIR=/etc/ssl/rainrag-origin
ORIGIN_CN=172.16.52.220
DAYS_CA=3650
DAYS_LEAF=825

require_root() { [ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }; }

init() {
  install -d -m 755 "$DIR"
  install -d -m 700 "$DIR/private"
  if [ -f "$DIR/ca.crt" ]; then
    echo "CA already present at $DIR/ca.crt, leaving it alone"
  else
    openssl req -x509 -newkey rsa:4096 -nodes -sha256 -days "$DAYS_CA" \
      -keyout "$DIR/private/ca.key" -out "$DIR/ca.crt" \
      -subj "/CN=RainRAG origin CA" 2>/dev/null
    chmod 600 "$DIR/private/ca.key"
    echo "created CA $DIR/ca.crt"
  fi
  if [ -f "$DIR/origin.crt" ]; then
    echo "origin cert already present, leaving it alone"
  else
    openssl req -newkey rsa:2048 -nodes -sha256 \
      -keyout "$DIR/private/origin.key" -out "$DIR/origin.csr" \
      -subj "/CN=$ORIGIN_CN" 2>/dev/null
    openssl x509 -req -in "$DIR/origin.csr" -CA "$DIR/ca.crt" -CAkey "$DIR/private/ca.key" \
      -CAcreateserial -days "$DAYS_LEAF" -sha256 -out "$DIR/origin.crt" \
      -extfile <(printf 'subjectAltName=IP:%s\nextendedKeyUsage=serverAuth\n' "$ORIGIN_CN") 2>/dev/null
    chmod 600 "$DIR/private/origin.key"
    rm -f "$DIR/origin.csr"
    echo "created origin cert $DIR/origin.crt"
  fi
  echo
  echo "Hand this CA certificate to the gateway owner (public, safe to paste):"
  echo "---"
  cat "$DIR/ca.crt"
}

sign() {
  local csr=${1:?usage: sign FILE.csr}
  [ -f "$csr" ] || { echo "no such CSR: $csr" >&2; exit 1; }
  local subject
  subject=$(openssl req -in "$csr" -noout -subject)
  echo "signing: $subject"
  openssl x509 -req -in "$csr" -CA "$DIR/ca.crt" -CAkey "$DIR/private/ca.key" \
    -CAcreateserial -days "$DAYS_LEAF" -sha256 -out "${csr%.csr}.crt" \
    -extfile <(printf 'extendedKeyUsage=clientAuth\n') 2>/dev/null
  echo "wrote ${csr%.csr}.crt -- send back only this file, never a key"
}

selftest() {
  # A throwaway client cert so the listener can be proven before anyone relies
  # on it: with it a request must succeed, without it nginx must refuse.
  local t=${DIR}/selftest
  install -d -m 700 "$t"
  openssl req -newkey rsa:2048 -nodes -sha256 -keyout "$t/c.key" -out "$t/c.csr" \
    -subj "/CN=selftest" 2>/dev/null
  openssl x509 -req -in "$t/c.csr" -CA "$DIR/ca.crt" -CAkey "$DIR/private/ca.key" \
    -CAcreateserial -days 1 -sha256 -out "$t/c.crt" \
    -extfile <(printf 'extendedKeyUsage=clientAuth\n') 2>/dev/null
  chmod 644 "$t/c.crt" "$t/c.key"
  echo "$t/c.crt $t/c.key"
}

require_root
case "${1:-}" in
  init) init ;;
  sign) shift; sign "$@" ;;
  selftest) selftest ;;
  *) echo "usage: $0 {init|sign FILE.csr|selftest}" >&2; exit 1 ;;
esac
